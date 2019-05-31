#!/usr/bin/env python
# https://github.com/andrewliao11/gail-tf
import logging
import os.path as osp
import sys
from gym.spaces import Discrete, MultiDiscrete

from gailtf.baselines import bench
from gailtf.common.common import *
from environments.atari.wrappers.actionWrapper import ActionWrapper
from gailtf.baselines.common import set_global_seeds
from gailtf.baselines.common.atari_wrappers import make_atari
from gailtf.dataset.mujoco import Mujoco_Dset

PI = 'pi'


def train(args):
    global env

    if args.expert_path is not None:
        assert osp.exists(args.expert_path)
    if args.load_model_path is not None:
        assert osp.exists(args.load_model_path + '.meta')
        args.pretrained = False

    printArgs(args)

    # ================================================ ENVIRONMENT =====================================================
    isNes = False
    U.make_session(num_cpu=args.num_cpu).__enter__()
    set_global_seeds(args.seed)

    if args.networkName == "MLP":
        env = gym.make(args.env_id)
        env = ActionWrapper(env, args.discrete)
    elif args.networkName == "CNN":
        env = make_atari(args.env_id)
        env = ActionWrapper(env, args.discrete)
        if args.deepmind:
            from gailtf.baselines.common.atari_wrappers import wrap_deepmind
            env = wrap_deepmind(env, False)

    env.metadata = 0
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)

    discrete = (".D." if args.discrete else ".MD")
    # ============================================== PLAY AGENT ========================================================
    # ==================================================================================================================
    if args.task == 'play_agent':
        logger.log("Playing agent...")
        from environments.atari.atari_agent import playAtari
        agent = policy_fn(args, PI, env, reuse=False)
        playAtari(env, agent, U, modelPath=args.load_model_path, fps=60, stochastic=args.stochastic_policy, zoom=2, delay=10)
        env.close()
        sys.exit()

    # ========================================== SAMPLE TRAJECTORY FROM RL =============================================
    # ==================================================================================================================

    if args.task == 'RL_expert':
        logger.log("Sampling trajectory...")
        stoch = 'stochastic.' if args.stochastic_policy else 'deterministic.'
        taskName = stoch + "" + args.alg + "." + args.env_id + discrete + "." + str(args.maxSampleTrajectories)
        currentPolicy = policy_fn(args,
                                  PI,
                                  env,
                                  reuse=False)
        episodesGenerator = traj_episode_generator(currentPolicy,
                                                   env,
                                                   args.trajectoriesPerBatch,
                                                   stochastic=args.stochastic_policy,
                                                   render=args.visualize,
                                                   downsample=args.downsample)
        sample_trajectory(args.load_model_path,
                          episodesGenerator,
                          taskName,
                          args.stochastic_policy,
                          max_sample_traj=args.maxSampleTrajectories)
        sys.exit()

    # ======================================== SAMPLE TRAJECTORY FROM HUMAN ============================================
    # ==================================================================================================================
    if args.task == 'human_expert':
        logger.log("Human plays...")
        taskName = "human." + args.env_id + "_" + args.networkName + "." + "50"
        args.checkpoint_dir = osp.join(args.checkpoint_dir, taskName)
        taskName = taskName + ".pkl"

        if isNes:
            from environments.nes_py.app.play_human import playNES
            sampleTrajectories = playNES(env, fps=15, taskName=taskName)
        else:
            from environments.atari.atari_human import playAtari
            sampleTrajectories = playAtari(env, fps=15, zoom=2, taskName=taskName)

        pkl.dump(sampleTrajectories, open(taskName, "wb"))
        env.close()
        sys.exit()

    # =========================================== TRAIN EXPERT =========================================================
    # ==================================================================================================================

    if args.task == "train_RL_expert":
        logger.log("Training RL expert...")
        # ============================================= TRAIN TRPO =====================================================
        # ==============================================================================================================

        if args.alg == 'trpo':
            from gailtf.baselines.trpo_mpi import trpo_mpi
            taskName = args.alg + "." + args.env_id + "." + str(args.policy_hidden_size) + discrete + "." + str(
                args.maxSampleTrajectories)

            rank = MPI.COMM_WORLD.Get_rank()
            if rank != 0:
                logger.set_level(logger.DISABLED)
            workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
            set_global_seeds(workerseed)
            env = gym.make(args.env_id)

            env = bench.Monitor(env, logger.get_dir() and
                                osp.join(logger.get_dir(), "%i.monitor.json" % rank))
            env.seed(workerseed)
            gym.logger.setLevel(logging.WARN)

            args.checkpoint_dir = osp.join("data/training", taskName)
            trpo_mpi.learn(args,
                           env,
                           policy_fn,
                           timesteps_per_batch=1024,
                           max_iters=50_000,
                           vf_iters=5,
                           vf_stepsize=1e-3,
                           task_name=taskName
                           )

            env.close()
            sys.exit()

        else:
            return NotImplementedError

    # =================================================== GAIL =========================================================
    # ==================================================================================================================
    if args.task == 'train_gail' or args.task == 'evaluate':

        taskName = get_task_name(args)
        args.checkpoint_dir = osp.join(args.checkpoint_dir, taskName)
        args.log_dir = osp.join(args.log_dir, taskName)
        args.task_name = taskName

        dataset = Mujoco_Dset(expert_path=args.expert_path, ret_threshold=args.ret_threshold,
                              traj_limitation=args.traj_limitation)

        # discriminator
        if len(env.observation_space.shape) > 2:
            from gailtf.network.adversary_cnn import TransitionClassifier
        else:
            if args.wasserstein:
                from gailtf.network.w_adversary import TransitionClassifier
            else:
                from gailtf.network.adversary import TransitionClassifier

        discriminator = TransitionClassifier(env,
                                             args.adversary_hidden_size,
                                             entcoeff=args.adversary_entcoeff)

        pretrained_weight = None
        # pre-training with BC (optional):
        if (args.pretrained and args.task == 'train_gail') or args.alg == 'bc':
            # Pretrain with behavior cloning
            from gailtf.algo import behavior_clone
            if args.alg == 'bc' and args.task == 'evaluate':
                behavior_clone.evaluate(args, env, policy_fn, args.load_model_path,
                                        stochastic_policy=args.stochastic_policy)
                sys.exit()
            if args.load_model_path is None:
                pretrained_weight = behavior_clone.learn(args,
                                                         env,
                                                         policy_fn,
                                                         dataset)
            if args.alg == 'bc':
                sys.exit()

        if args.alg == 'trpo':
            # Set up for MPI seed
            rank = MPI.COMM_WORLD.Get_rank()
            if rank != 0:
                logger.set_level(logger.DISABLED)
            workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()
            set_global_seeds(workerseed)
            env.seed(workerseed)

            # if args.wasserstein:
            #     from gailtf.algo import w_trpo_mpi as trpo
            # else:
            from gailtf.algo import trpo_mpi as trpo

            if args.task == 'train_gail':
                trpo.learn(args,
                           env,
                           policy_fn,
                           discriminator,
                           dataset,
                           pretrained_weight=pretrained_weight,
                           cg_damping=0.1,
                           vf_iters=5,
                           vf_stepsize=1e-3
                           )
            elif args.task == 'evaluate':
                trpo.evaluate(env, policy_fn, args.load_model_path, timesteps_per_batch=1024,
                              number_trajs=10, stochastic_policy=args.stochastic_policy)
            else:
                raise NotImplementedError

        env.close()
        sys.exit()
