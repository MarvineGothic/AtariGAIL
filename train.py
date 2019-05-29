from gailtf.agent.run_atari import *


def main():
    args = argsParser()

    args.env_id = 'MontezumaRevenge-ram-v0'

    args.wasserstein = False
    args.clip_weights = False
    args.discrete = False
    args.deepmind = True
    args.downsample = False
    args.visualize = True
    args.policy_hidden_size = 100
    args.adversary_hidden_size = 100
    args.stochastic_policy = True

    # ===================== ALGORITHM: ==========================
    # 'bc', 'trpo', 'human'

    args.alg = 'trpo'
    # args.alg = 'human'
    # args.alg = 'bc'

    # ================================ TASK ==============================
    # 'train_expert', 'train_gail', 'sample_trajectory', 'play_agent', 'evaluate'

    args.task = 'train_expert'
    # args.task = 'train_gail'
    args.task = 'play_agent'
    # args.task = 'sample_trajectory'

    # args.expert_path = 'data/expert/stochastic.trpo.Boxing-ram-v0.MD.1500.score_100-0.pkl'

    args.load_model_path = 'data/agent/Montezuma/trpo_w_gail.MontezumaRevenge-ram-v0.100.MD.1500/humanLike/' \
                           'trpo_w_gail.MontezumaRevenge-ram-v0.100.MD.1500_score-10100'
    args.networkName = 'MLP'
    args.trajectoriesPerBatch = 1024

    if '-ram-' in args.env_id:
        args.networkName = 'MLP'
        args.trajectoriesPerBatch = 1024
    elif 'NoFrameskip' in args.env_id:
        args.networkName = 'CNN'
        args.trajectoriesPerBatch = 512

    # ============================ PATCHES =================================
    if args.alg == 'human':
        args.discrete = True
        args.task = 'train_expert'

    if args.task == 'sample_trajectory':
        args.maxSampleTrajectories = 50

    if args.alg == 'bc':
        args.task = 'train_gail'

    if args.alg == 'train_gail':
        assert osp.exists(args.expert_path)

    printArgs(args)
    train(args)


if __name__ == '__main__':
    main()
