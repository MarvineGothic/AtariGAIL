import argparse
import gym, numpy as np
import pickle as pkl
from mpi4py import MPI
import os
import seaborn as sns
import matplotlib.pyplot as plt

from gailtf.baselines.common import tf_util as U, colorize
from gailtf.baselines import logger
from gailtf.baselines.common.atari_wrappers import wrap_deepmind, make_atari
from environments.atari.wrappers.actionWrapper import ActionWrapper


def argsParser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='Boxing-ram-v0')
    parser.add_argument('--discrete', default=True)
    parser.add_argument('--deepmind', default=False)
    parser.add_argument('--downsample', default=False)
    parser.add_argument('--networkName', choices=['MLP', 'CNN'], default="MLP")
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=1)
    parser.add_argument('--trajectoriesPerBatch', type=int, default=1024)
    parser.add_argument('--maxSampleTrajectories', default=1500)
    # paths:
    parser.add_argument('--expert_path', type=str, default=None)
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='data/agent')
    parser.add_argument('--log_dir', help='the directory to save log file', default='data/log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    parser.add_argument('--max_to_keep', help='how many saves to keep in a directory', default=10)
    # Task
    parser.add_argument('--task', type=str,
                        choices=['train_RL_expert', 'train_gail', 'RL_expert', 'human_expert', 'play_agent',
                                 'evaluate'],
                        default='train_gail')
    parser.add_argument('--visualize', default=False)
    parser.add_argument('--task_name', default='taskName')
    parser.add_argument('--stochastic_policy', type=bool, default=True)
    #  Mujoco Dataset Configuration
    parser.add_argument('--ret_threshold', help='the return threshold for the expert trajectories', type=int,
                        default=0)
    parser.add_argument('--traj_limitation', type=int, default=np.inf)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--adversary_hidden_size', type=int, default=100)
    parser.add_argument('--clip_weights', type=bool, default=False)
    parser.add_argument('--wasserstein', type=bool, default=False)
    # Algorithms Configuration
    parser.add_argument('--alg', type=str, choices=['bc', 'trpo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--policy_entcoeff', help='entropy coefficient of policy', type=float, default=0)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficient of discriminator', type=float, default=1e-3)
    # Training Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=5e6)
    # Behavior Cloning
    parser.add_argument('--pretrained', help='Use BC to pretrain', type=bool, default=False)
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def printArgs(args):
    opt = vars(args)
    print("{:<25} {:<15}".format('Key', 'Label'))
    for k, v in opt.items():
        print("{:<25} {:<15}".format(str(k), str(v)))


def get_task_name(args):
    discrete = (".D." if args.discrete else ".MD")
    if args.task == 'train_gail':
        task_name = args.alg + "_gail."
        if args.pretrained:
            task_name += "with_pretrained."
        task_name += args.env_id
        task_name = task_name + "." + str(args.policy_hidden_size) + discrete

        if args.wasserstein:
            task_name = task_name + "_W"
            if args.clip_weights:
                task_name = task_name + "_CL"
        if args.expert_path is not None:
            suffix = args.expert_path.split('.')
            if len(suffix) > 4:
                task_name += "." + suffix[-2]
    else:
        return NotImplementedError
    return task_name


# ---------- Policy function -----------
def policy_fn(args, name, env, reuse=False):
    if args.networkName == "CNN":
        from gailtf.baselines.policies.nosharing_cnn_policy import CnnPolicy
        return CnnPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space)
    elif args.networkName == "MLP":
        from gailtf.baselines.policies.mlp_policy import MlpPolicy
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                         reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=2)
    else:
        return NotImplementedError


def save_expert_data_set_plot(data_set, D, t_name):
    sns.set(color_codes=True)
    exp_obs = data_set.dset.inputs
    exp_acs = data_set.dset.labels
    expert_data_logits = D.get_expert_logits(exp_obs, exp_acs)
    min = expert_data_logits.min()
    max = expert_data_logits.max()
    sns.distplot(expert_data_logits, color='blue', label="Expert dataset")
    plt.xlabel("Min: " + str(min) + "/Max: " + str(max))
    plt.ylabel("Out: " + str(len(expert_data_logits)))
    plt.legend()
    file_name = os.path.join('data/plot/', t_name + "/")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name + '/dl_expert_dataset.png')
    plt.clf()


def save_plots(exp_l, gen_l, i_so_far, index, t_name):
    sns.set(color_codes=True)
    min = exp_l.min() if exp_l.min() < gen_l.min() else gen_l.min()
    max = exp_l.max() if exp_l.max() > gen_l.max() else gen_l.max()
    print(str(exp_l.min()))
    sns.distplot(exp_l, color='blue', label="Expert")
    sns.distplot(gen_l, color='red', label="Generator")
    plt.title(
        'Expert batch: ' + str(index) + ' exp:' + str(len(exp_l)) + 'gen:' + str(len(gen_l)))
    plt.xlabel("Min: " + str(min) + "/Max: " + str(max))
    plt.ylabel("Out: " + str(len(exp_l)))
    plt.legend()
    file_name = os.path.join('data/plot/', t_name + "/")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name + '/dl_' + str(i_so_far) + '_' + str(index) + '.png')
    plt.clf()


def make_atari_vs_nes_env(args):
    # ============================
    isNes = False
    if "SuperMario" in args.env_id:
        isNes = True
        from environments import gym_super_mario_bros
        env = gym_super_mario_bros.make(args.env_id)
    else:
        env = gym.make(args.env_id)

    env = ActionWrapper(env, args.discrete)
    env.metadata = 0

    return env, isNes


def make_atari_env(args):
    env = make_atari(args.env_id)
    env = ActionWrapper(env, args.discrete)
    if args.deepmind:
        env = wrap_deepmind(env, False)
    return env


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0)  # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1 - new[t + 1]
        delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def sample_trajectory(load_model_path, traj_gen, task_name, sample_stochastic, max_sample_traj=1500):
    assert load_model_path is not None
    U.load_state(load_model_path)
    sample_trajs = []
    for iters_so_far in range(max_sample_traj):
        logger.log("********** Iteration %i ************" % iters_so_far)
        traj = traj_gen.__next__()
        ob, new, ep_ret, ac, rew, ep_len = traj['ob'], traj['new'], traj['ep_ret'], traj['ac'], traj['rew'], traj[
            'ep_len']
        logger.record_tabular("ep_ret", ep_ret)
        logger.record_tabular("ep_len", ep_len)
        logger.record_tabular("immediate reward", np.mean(rew))
        if MPI.COMM_WORLD.Get_rank() == 0:
            logger.dump_tabular()
        traj_data = {"ob": ob, "ac": ac, "rew": rew, "ep_ret": ep_ret}
        sample_trajs.append(traj_data)

    sample_ep_rets = [traj["ep_ret"] for traj in sample_trajs]
    logger.log("Average total return: %f" % (sum(sample_ep_rets) / len(sample_ep_rets)))

    pkl.dump(sample_trajs, open(task_name + ".pkl", "wb"))


def traj_segment_generator(pi, env, discriminator, horizon, stochastic, visualize=False):
    # Initialize state variables
    steps = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    true_rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    cur_ep_true_ret = 0
    ep_true_rets = []
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    true_rews = np.zeros(horizon, 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    save = False
    playerScore = opponentScore = 0
    wins = losses = ties = gamesTotal = totalPlayer = totalOpponent = 0

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if steps > 0 and steps % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets, "save": save}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_true_rets = []
            ep_lens = []
            save = False
        i = steps % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        rew = discriminator.get_reward(ob, ac)
        ob, true_rew, new, _ = env.step(ac)

        if true_rew > 0:
            playerScore += abs(true_rew)
        else:
            opponentScore += abs(true_rew)

        if 'Montezuma' in env.spec.id and true_rew >= 100:
            save = True

        rews[i] = rew
        true_rews[i] = true_rew
        if visualize:
            env.render()

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew
        cur_ep_len += 1
        if new:
            msg = format("End of game: score %d - %d" % (playerScore, opponentScore))
            print(colorize(msg, color='red'))
            gamesTotal += 1

            if 'Skiing' in env.spec.id and opponentScore < 5000:
                save = True

            if playerScore > opponentScore:
                wins += 1
            elif opponentScore > playerScore:
                losses += 1
            else:
                ties += 1

            totalPlayer += playerScore
            totalOpponent += opponentScore

            playerScore = opponentScore = 0

            msg = format("Status so far: \nGames played - %d wins - %d losses - %d ties - %d\n Total score: %d - %d" % (
                gamesTotal, wins, losses, ties, totalPlayer, totalOpponent))
            print(colorize(msg, color='red'))

            ep_rets.append(cur_ep_ret)
            ep_true_rets.append(cur_ep_true_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_true_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        steps += 1


def traj_episode_generator(pi, env, horizon, stochastic, render, downsample):
    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    playerScore = opponentScore = 0
    wins = losses = ties = gamesTotal = totalPlayer = totalOpponent = 0

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        if not downsample and hasattr(env.unwrapped, 'ale'):
            obs.append(env.unwrapped.ale.getScreenRGB2())
        else:
            obs.append(obs)

        obs.append(ob)
        news.append(new)
        acs.append(ac)
        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1

        if render:
            env.render()

        if t > 0 and (new or t % horizon == 0):
            msg = format("End of game: score %d - %d" % (playerScore, opponentScore))
            print(colorize(msg, color='red'))
            gamesTotal += 1
            if playerScore > opponentScore:
                wins += 1
            elif opponentScore > playerScore:
                losses += 1
            else:
                ties += 1

            totalPlayer += playerScore
            totalOpponent += opponentScore

            playerScore = opponentScore = 0

            msg = format("Status so far: \nGames played - %d wins - %d losses - %d ties - %d\n Total score: %d - %d" % (
                gamesTotal, wins, losses, ties, totalPlayer, totalOpponent))
            print(colorize(msg, color='red'))

            # convert list into numpy array
            obs = np.array(obs)
            rews = np.array(rews)
            news = np.array(news)
            acs = np.array(acs)
            yield {"ob": obs, "rew": rews, "new": news, "ac": acs,
                   "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
            ob = env.reset()
            cur_ep_ret = 0
            cur_ep_len = 0
            t = 0

            # Initialize history arrays
            obs = []
            rews = []
            news = []
            acs = []
        t += 1


def evaluate(env, policy_func, load_model_path, timesteps_per_batch, number_trajs=10,
             stochastic_policy=False):
    from tqdm import tqdm
    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=False)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    ep_gen = traj_episode_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
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


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
