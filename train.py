from gailtf.agent.run_atari import *


def main():
    args = argsParser()

    args.env_id = 'Boxing-ram-v0'

    args.discrete = False  # switch action space between Discrete and MultiDiscrete
    args.visualize = True  # show training in pygame window
    args.policy_hidden_size = 100  # policy's size of hidden layers
    args.adversary_hidden_size = 100  # discriminator's size of hidden layers
    args.stochastic_policy = True  # type of environment
    args.pretrained = False  # use BC to pre-train weights

    if '-ram-' in args.env_id:
        args.networkName = 'MLP'
        args.trajectoriesPerBatch = 1024
    else:
        args.networkName = 'CNN'
        args.trajectoriesPerBatch = 512

    # ================================ TASK ==============================
    # 'train_RL_expert', 'train_gail', 'RL_expert', 'human_expert', 'play_agent'

    args.task = 'train_RL_expert'
    args.task = 'RL_expert'
    args.task = 'human_expert'
    args.task = 'train_gail'
    # args.task = 'play_agent'

    # ============================== ALGORITHM ============================
    # 'bc', 'trpo'
    args.alg = 'trpo'
    # args.alg = 'bc'

    # ================================ PATHS ==============================

    args.expert_path = 'data/expert/stochastic.trpo.Boxing-ram-v0.MD.1500.score_100-0.pkl'

    # args.load_model_path = 'data/training/trpo.Boxing-ram-v0.100.MD.1500/trpo.Boxing-ram-v0.100.MD.1500-0'

    # ============================ PATCHES =================================

    if args.task == 'RL_expert':
        args.maxSampleTrajectories = 1  # set number of samples for RL expert

    if args.alg == 'bc':
        args.task = 'train_gail'

    if args.alg == 'train_gail':
        assert osp.exists(args.expert_path)

    train(args)


if __name__ == '__main__':
    main()
