import time

import gym
import pygame, pickle as pkl
from tqdm import tqdm
import numpy as np
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE
from gailtf.baselines.common import tf_util as U
from environments.atari.wrappers.actionWrapper import ActionWrapper
from gailtf.baselines.common import colorize


def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def replayACS(env, modelPath, transpose=True, fps=30, zoom=None):
    """
    Replays a game from recorded trajectories using actions
    This method is not precise though, because it indirectly recovers environment states from actions.
    Sometimes it gets asynchronous and distorts the real trajectory.
    :param env: Atari environment
    :param modelPath: path to trained model
    :param transpose:
    :param fps:
    :param zoom:
    :return:
    """
    global obs
    with open(modelPath, 'rb') as rfp:
        trajectories = pkl.load(rfp)

    U.make_session(num_cpu=1).__enter__()

    U.initialize()

    tempEnv = env
    while not isinstance(tempEnv, ActionWrapper):
        try:
            tempEnv = tempEnv.env
        except:
            break
    # using ActionWrapper:
    if isinstance(tempEnv, ActionWrapper):
        obs_s = tempEnv.screen_space
    else:
        obs_s = env.observation_space

    # assert type(obs_s) == Box
    assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3])

    if zoom is None:
        zoom = 1

    video_size = int(obs_s.shape[0] * zoom), int(obs_s.shape[1] * zoom)

    if transpose:
        video_size = tuple(reversed(video_size))

    # setup the screen using pygame
    flags = RESIZABLE | HWSURFACE | DOUBLEBUF
    screen = pygame.display.set_mode(video_size, flags)
    pygame.event.set_blocked(pygame.MOUSEMOTION)
    clock = pygame.time.Clock()

    # =================================================================================================================

    running = True
    envDone = False

    playerScore = opponentScore = 0
    wins = losses = ties = gamesTotal = totalPlayer = totalOpponent = 0

    while running:
        trl = len(trajectories)

        for i in range(trl):
            obs = env.reset()
            print("\nRunning trajectory {}".format(i))
            print("Length {}".format(len(trajectories[i]['ac'])))

            for ac in tqdm(trajectories[i]['ac']):
                if not isinstance(ac, list):
                    ac = np.atleast_1d(ac)

                obs, reward, envDone, info = env.step(ac)

                # track of player score:
                if reward > 0:
                    playerScore += abs(reward)
                else:
                    opponentScore += abs(reward)

                if hasattr(env, 'getImage'):
                    obs = env.getImage()

                if obs is not None:
                    if len(obs.shape) == 2:
                        obs = obs[:, :, None]
                    if obs.shape[2] == 1:
                        obs = obs.repeat(3, axis=2)
                    display_arr(screen, obs, video_size, transpose)

                    pygame.display.flip()
                    clock.tick(fps)

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
    pygame.quit()


def replayOBS(env, modelPath, transpose=True, fps=15, zoom=None):
    """
    Replays a game from recorded trajectories using observations as screen images
    :param modelPath:
    :param transpose:
    :param fps:
    :param zoom:
    :return:
    """
    global obs
    with open(modelPath, 'rb') as rfp:
        trajectories = pkl.load(rfp)

    obs_s = trajectories[0]['ob'][0]
    # assert type(obs_s) == Box
    assert len(obs_s.shape) == 2 or (len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3])

    if zoom is None:
        zoom = 1

    video_size = int(obs_s.shape[0] * zoom), int(obs_s.shape[1] * zoom)

    if transpose:
        video_size = tuple(reversed(video_size))

    # setup the screen using pygame
    flags = RESIZABLE | HWSURFACE | DOUBLEBUF
    screen = pygame.display.set_mode(video_size, flags)
    pygame.event.set_blocked(pygame.MOUSEMOTION)
    clock = pygame.time.Clock()

    # =================================================================================================================

    running = True
    envDone = False

    playerScore = opponentScore = 0
    wins = losses = ties = gamesTotal = totalPlayer = totalOpponent = 0

    while running:
        pygame.event.get()
        trl = len(trajectories)
        for i in range(trl):
            print("\nRunning trajectory {}".format(i))
            print("Length {}".format(len(trajectories[i]['ob'])))

            for obs in tqdm(trajectories[i]['ob']):
                if obs is not None:
                    if len(obs.shape) == 2:
                        obs = obs[:, :, None]
                    if obs.shape[2] == 1:
                        obs = obs.repeat(3, axis=2)
                    display_arr(screen, obs, video_size, transpose)

                    pygame.display.flip()
                    clock.tick(fps)
    pygame.quit()


if __name__ == '__main__':
    mP = '../../data/expert/human.MontezumaRevenge-ram-v0_MLP.50_traj_size_100.pkl'
    #replayOBS(mP)
    #env = make_atari('MontezumaRevengeNoFrameskip-v4')
    env = gym.make('MontezumaRevenge-ram-v0')
    env = ActionWrapper(env, True)

    replayOBS(env, mP)
