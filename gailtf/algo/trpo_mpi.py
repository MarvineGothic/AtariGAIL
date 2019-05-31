import tensorflow as tf
import time
from gailtf.baselines.common import explained_variance, zipsame, dataset, fmt_row
from collections import deque
from gailtf.baselines.common.mpi_adam import MpiAdam
from gailtf.baselines.common.cg import cg
from contextlib import contextmanager
from gailtf.common.statistics import stats
from gailtf.agent.run_atari import *


def learn(args,
          env,
          policy_func,
          discriminator,
          expert_dataset,
          pretrained_weight, *,
          pretrained=False,
          g_step=3,
          d_step=1,
          timesteps_per_batch=1024,  # what to train on
          max_kl=0.01,
          cg_iters=10,
          cg_damping=1e-2,
          gamma=0.995,
          lam=0.97,  # advantage estimation
          entcoeff=0.0,
          vf_iters=3,
          vf_stepsize=3e-4,
          d_stepsize=3e-4,
          max_timesteps=0,
          max_episodes=0,
          max_iters=0,  # time constraint
          callback=None,
          save_per_iter=100,
          ckpt_dir=None,
          log_dir=None,
          load_model_path=None,
          task_name=None,
          visualize=False
          ):
    # ============================================ INIT FROM ARGS ======================================================
    pretrained = args.pretrained
    g_step = args.g_step
    d_step = args.d_step
    timesteps_per_batch = args.trajectoriesPerBatch
    max_kl = args.max_kl
    max_iters = args.num_timesteps
    entcoeff = args.policy_entcoeff
    ckpt_dir = args.checkpoint_dir
    log_dir = args.log_dir
    save_per_iter = args.save_per_iter
    load_model_path = args.load_model_path
    task_name = args.task_name
    visualize = args.visualize
    max_to_keep = args.max_to_keep

    # ==================================================================================================================
    logger.log("Task name: " + task_name)
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # sns.set(color_codes=True)
    # Setup losses and stuff
    # ----------------------------------------
    pi = policy_func(args, "pi", env, reuse=(pretrained_weight != None))
    oldpi = policy_func(args, "oldpi", env, reuse=(pretrained_weight != None))
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    entbonus = entcoeff * meanent

    vferr = U.mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))  # advantage * pnew / pold
    surrgain = U.mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    d_adam = MpiAdam(discriminator.get_trainable_variables())
    #d_adam = tf.train.RMSPropOptimizer(learning_rate=d_stepsize)
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start + sz], shape))
        start += sz
    # pylint: disable=E1111
    gvp = tf.add_n([U.sum(g * tangent) for (g, tangent) in zipsame(klgrads, tangents)])
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
                                                    for (oldv, newv) in
                                                    zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    writer = U.FileWriter(log_dir)
    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    d_adam.sync()
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi,
                                     env,
                                     discriminator,
                                     timesteps_per_batch,
                                     stochastic=True,
                                     visualize=visualize)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40)  # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40)  # rolling buffer for episode rewards
    true_rewbuffer = deque(maxlen=40)

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0]) == 1

    g_loss_stats = stats(loss_names)
    d_loss_stats = stats(discriminator.loss_name)
    ep_stats = stats(["True_rewards", "Rewards", "Episode_length"])
    # if provide pretrained weight
    if pretrained_weight is not None:
        logger.log("Load pretrained weights...")
        U.load_state(pretrained_weight, var_list=pi.get_variables())
    # if provieded model path
    elif load_model_path is not None:
        logger.log("Load model...")
        iters_so_far = int(load_model_path.split("-")[-1]) + 1
        U.load_state(load_model_path)

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break

        # Save model
        if iters_so_far % save_per_iter == 0 and ckpt_dir is not None:
            U.save_state(os.path.join(ckpt_dir, task_name), counter=iters_so_far, max_to_keep=max_to_keep)

        logger.log("********** Iteration %i ************" % iters_so_far)

        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        # ------------------ Update G ------------------
        logger.log("Optimizing Policy...")
        for _ in range(g_step):
            with timed("sampling"):
                seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            if seg["save"]:
                itr = str(round(iters_so_far / 100))
                U.save_state(os.path.join(ckpt_dir + "_score", task_name + "_score"),
                             counter=iters_so_far,
                             max_to_keep=max_to_keep)

            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"]  # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob)  # update running mean/std for policy

            args = seg["ob"], seg["ac"], atarg
            fvpargs = [arr[::5] for arr in args]

            logger.log("set old parameter values to new parameter values")
            assign_old_eq_new()  # set old parameter values to new parameter values

            with timed("cg"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            g = allmean(g)

            if np.allclose(g, 0):
                logger.log("Got zero gradient. Still try updating")

            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank == 0)
            assert np.isfinite(stepdir).all()
            shs = .5 * stepdir.dot(fisher_vector_product(stepdir))

            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f" % (expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step. Setting theta to previous")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum()))  # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

            with timed("vf"):
                for _ in range(vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                                                             include_final_partial_batch=False, batch_size=128):
                        if hasattr(pi, "ob_rms"):
                            pi.ob_rms.update(mbob)  # update running mean/std for policy
                        g = allmean(compute_vflossandgrad(mbob, mbret))
                        vfadam.update(g, vf_stepsize)

        g_losses = meanlosses
        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_name))
        ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob))
        batch_size = len(ob) // d_step
        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
        for ob_batch, ac_batch in dataset.iterbatches((ob, ac), include_final_partial_batch=False,
                                                      batch_size=batch_size):

            ob_expert, ac_expert = expert_dataset.get_next_batch(len(ob_batch))
            # update running mean/std for discriminator
            if hasattr(discriminator, "obs_rms"):
                discriminator.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

            *newlosses, g = discriminator.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
            d_adam.update(allmean(g), d_stepsize)
            # d_adam.minimize(discriminator.total_loss, var_list=discriminator.get_trainable_variables())

            exp_logits, gen_logits = discriminator.get_logits(ob_batch, ac_batch, ob_expert, ac_expert)

            try:
                if iters_so_far == 0:
                    logger.log("Saving expert plot...")
                    save_expert_data_set_plot(expert_dataset, discriminator, task_name)
                if iters_so_far % save_per_iter == 0:
                    logger.log("Saving probability plots...")
                    save_plots(exp_logits, gen_logits, iters_so_far, 0, task_name)
            except Exception as e:
                logger.log("Couldn't save plots. Exception: " + str(e))
                pass

            d_losses.append(newlosses)
        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews, true_rets = map(flatten_lists, zip(*listoflrpairs))
        true_rewbuffer.extend(true_rets)
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(true_rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank == 0:
            logger.dump_tabular()
            g_loss_stats.add_all_summary(writer, g_losses, iters_so_far)
            d_loss_stats.add_all_summary(writer, np.mean(d_losses, axis=0), iters_so_far)
            ep_stats.add_all_summary(writer, [np.mean(true_rewbuffer), np.mean(rewbuffer),
                                              np.mean(lenbuffer)], iters_so_far)
