import tensorflow as tf
import gailtf.baselines.common.tf_util as U
from gailtf.baselines import logger
from tqdm import tqdm
from gailtf.baselines.common.mpi_adam import MpiAdam
import tempfile, os
from gailtf.common.statistics import stats


def evaluate(args, env, policy_func, load_model_path, stochastic_policy=False, number_trajs=10):
    from gailtf.algo.trpo_mpi import traj_episode_generator
    pi = policy_func(args, "pi", env)  # Construct network for new policy
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    ep_gen = traj_episode_generator(pi, env, 1024, stochastic=stochastic_policy)
    U.load_state(load_model_path)
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = ep_gen.__next__()
        ep_len, ep_ret = traj['ep_len'], traj['ep_ret']
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    print("Average length:", sum(len_list) / len(len_list))
    print("Average return:", sum(ret_list) / len(ret_list))


def learn(args,
          env,
          policy_func,
          dataset,
          optim_batch_size=128,
          adam_epsilon=1e-5,
          optim_stepsize=3e-4):

    # ============================== INIT FROM ARGS ==================================
    max_iters = args.BC_max_iter
    pretrained = args.pretrained
    ckpt_dir = args.checkpoint_dir
    log_dir = args.log_dir
    task_name = args.task_name

    val_per_iter = int(max_iters / 10)
    pi = policy_func(args, "pi", env)  # Construct network for new policy
    oldpi = policy_func(args, "oldpi", env)
    # placeholder
    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac - pi.ac))
    var_list = pi.get_trainable_variables()
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob, ac, stochastic], [loss] + [U.flatgrad(loss, var_list)])

    if not pretrained:
        writer = U.FileWriter(log_dir)
        ep_stats = stats(["Loss"])
    U.initialize()
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    for iter_so_far in tqdm(range(int(max_iters + 1))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        loss, g = lossandgrad(ob_expert, ac_expert, True)
        adam.update(g, optim_stepsize)
        if not pretrained:
            ep_stats.add_all_summary(writer, [loss], iter_so_far)
        if iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            loss, g = lossandgrad(ob_expert, ac_expert, False)
            logger.log("Validation:")
            logger.log("Loss: %f" % loss)
            if not pretrained:
                U.save_state(os.path.join(ckpt_dir, task_name), counter=iter_so_far)
    if pretrained:
        savedir_fname = tempfile.TemporaryDirectory().name
        U.save_state(savedir_fname, max_to_keep=args.max_to_keep)
        return savedir_fname
