import gym
import pygame, time
import matplotlib
import numpy as np
from gym.spaces.box import Box

from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE

from environments.atari.wrappers.actionWrapper import ActionWrapper
from gailtf.baselines.common import colorize


def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def playAtari(env,
              agent,
              U,
              modelPath,
              transpose=True,
              stochastic=False,
              fps=30,
              zoom=None,
              delay=None):
    """
    Plays an Atari games' agent from trained model
    :param env: Atari environment
    :param agent: agent's policy model (neural network)
    :param U: baseline's tf_util
    :param modelPath: path to trained model
    :param transpose: transpose video
    :param stochastic: environment type
    :param fps:
    :param zoom:
    :param delay: start delay
    :return:
    """
    global obs

    U.initialize()
    U.load_state(modelPath)

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

    # obs_s = env.observation_space
    # if len(env.observation_space.shape) < 3:
    #     obs_s = env.env.screen_space

    assert type(obs_s) == Box
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

    observation = env.reset()

    running = True
    envDone = False

    playerScore = opponentScore = 0
    wins = losses = ties = gamesTotal = totalPlayer = totalOpponent = 0

    print("Get ready...")
    if delay is not None:
        time.sleep(delay)

    while running:
        pygame.event.get()
        if envDone:
            # results of game:
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

            if gamesTotal == 3:
                running = False

            envDone = False
            observation = env.reset()

        else:
            action, value_predicted = agent.act(stochastic, observation)

            if not isinstance(action, list):
                action = np.atleast_1d(action)

            observation, reward, envDone, info = env.step(action)

            # track of player score:
            if reward > 0:
                playerScore += abs(reward)
            else:
                opponentScore += abs(reward)

        # as we using RAM for observations so we need a screed data for playing a game with pygame:
        if hasattr(env, 'getImage'):
            obs = env.getImage()
        elif hasattr(env.unwrapped, 'ale'):
            obs = env.unwrapped.ale.getScreenRGB2()

        if obs is not None:
            if len(obs.shape) == 2:
                obs = obs[:, :, None]
            if obs.shape[2] == 1:
                obs = obs.repeat(3, axis=2)
            display_arr(screen, obs, video_size, transpose)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
