from gailtf.agent.run_atari import *


def main():
    args = argsParser()

    args.env_id = 'Boxing-ram-v0'

    args.discrete = False  # switch action space between Discrete and MultiDiscrete
    args.visualize = True  # show training in pygame window
    args.policy_hidden_size = 100  # policy's size of hidden layers
    args.adversary_hidden_size = 100  # discriminator's size of hidden layers
    args.stochastic_policy = True  # type of environment

    if '-ram-' in args.env_id:
        args.networkName = 'MLP'
        args.trajectoriesPerBatch = 1024
    else:
        args.networkName = 'CNN'
        args.trajectoriesPerBatch = 512

    # ================================ TASK ==============================
    # 'train_RL_expert', 'train_gail', 'RL_expert', 'human_expert', 'play_agent', 'evaluate'

    args.task = 'train_RL_expert'
    # args.task = 'train_gail'
    # args.task = 'RL_expert'
    # args.task = 'human_expert'
    args.task = 'play_agent'

    # ============================== ALGORITHM ============================
    # 'bc', 'trpo'
    args.alg = 'trpo'
    # args.alg = 'bc'

    # ================================ PATHS ==============================

    # args.expert_path = 'data/expert/human.MontezumaRevenge-ram-v0_MLP.50_traj_size_100.pkl'

    args.load_model_path = 'data/agent/Boxing/trpo_gail.stochastic.trpo.Boxing-ram-v0.MD.1500.score_100-0.pkl/trpo_gail.Boxing.g_step_3.d_step_1.policy_entcoeff_0.adversary_entcoeff_0.001-10200'

    # ============================ PATCHES =================================

    if args.task == 'RL_expert':
        args.maxSampleTrajectories = 1500  # set number of samples for RL expert

    if args.alg == 'bc':
        args.task = 'train_gail'

    if args.alg == 'train_gail':
        assert osp.exists(args.expert_path)

    printArgs(args)
    train(args)


if __name__ == '__main__':
    main()
