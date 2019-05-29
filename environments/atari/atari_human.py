import gym
import os.path as osp
import pygame
import matplotlib
import numpy as np
import pickle as pkl
from gym.spaces.box import Box
from gailtf.common.config import *
from environments.atari.wrappers.actionWrapper import ActionWrapper

try:
    matplotlib.use('GTK3Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass

import pyglet.window as pw

from collections import deque
from pygame.locals import HWSURFACE, DOUBLEBUF, RESIZABLE, VIDEORESIZE
from threading import Thread


def display_arr(screen, arr, video_size, transpose):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


def playAtari(env, transpose=True, fps=30, zoom=None, callback=None, keys_to_action=None, taskName=None,
              timeLimit=None):
    """
    Allows a human player to play Atari game and collect game data (trajectories) in form of OBSERVATIONS, ACTIONS,
    REWARDS and EPISODE_RETURNS.
    :param env: Atari environment
    :param transpose:
    :param fps:
    :param zoom:
    :param callback:
    :param keys_to_action:
    :param taskName:
    :param timeLimit:
    :return:
    """

    global obs

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

    if keys_to_action is None:
        if hasattr(env, 'get_keys_to_action'):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, 'get_keys_to_action'):
            keys_to_action = env.unwrapped.get_keys_to_action()
        else:
            assert False, env.spec.id + " does not have explicit key to action mapping, " + \
                          "please specify one manually"
    relevant_keys = set(sum(map(list, keys_to_action.keys()), []))

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
    pressed_keys = []
    running = True
    envDone = True

    time = 1
    trajNr = 0
    observation = env.reset()
    # Initialize history arrays
    observations = []
    rewards = []
    done = []
    actions = []

    currentEpisodeReward = 0  # return in current episode
    currentEpisodeLength = 0  # length of current episode

    sampleTrajectories = []

    if osp.exists(taskName):
        with open(taskName, 'rb') as rfp:
            sampleTrajectories = pkl.load(rfp)
            trajNr = len(sampleTrajectories)

    while running:
        if envDone:
            envDone = False
            observation = env.reset()

            currentEpisodeReward = 0
            currentEpisodeLength = 0
        else:
            action = keys_to_action.get(tuple(sorted(pressed_keys)), 0)

            if not isinstance(action, list):
                action = np.atleast_1d(action)

            previous_observation = observation

            observations.append(observation)
            done.append(envDone)
            actions.append(action)

            observation, reward, envDone, info = env.step(action)
            # print("reward: " + str(reward))
            rewards.append(reward)
            currentEpisodeReward += reward
            currentEpisodeLength += 1

            if callback is not None:
                callback(previous_observation, observation, action, reward, envDone, info)

        # as we using RAM for observations so we need a screen data for playing a game with pygame:
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
        # process pygame events
        running, screen, envDone = keyHandler(relevant_keys, pressed_keys, screen, envDone)

        if (envDone and timeLimit is None) or (timeLimit is not None and time % timeLimit == 0):
            # save trajectoryData:
            trajectoryData = {OBSERVATIONS: np.array(observations),
                              ACTIONS: np.array(actions),
                              REWARDS: np.array(rewards),
                              EPISODE_RETURNS: currentEpisodeReward
                              }
            # put it in samples list:
            sampleTrajectories.append(trajectoryData)
            print("Reward: " + str(currentEpisodeReward))

            trajNr += 1
            print("Trajectory nr %d" % trajNr)
            print("Episode length: %d" % len(actions))
            # envDone = True

            # init variables:
            observations = []
            rewards = []
            done = []
            actions = []
            currentEpisodeReward = 0
            currentEpisodeLength = 0

        pygame.display.flip()
        clock.tick(fps)
        time += 1
    pygame.quit()

    return sampleTrajectories


def keyHandler(relevant_keys, pressed_keys, screen, envDone):
    running = True
    for event in pygame.event.get():
        # test events, set key states
        if event.type == pygame.KEYDOWN:
            if event.key in relevant_keys:
                pressed_keys.append(event.key)
            elif event.key == 27:
                envDone = True
        elif event.type == pygame.KEYUP:
            if event.key in relevant_keys:
                pressed_keys.remove(event.key)
        elif event.type == pygame.QUIT:
            running = False
        elif event.type == VIDEORESIZE:
            video_size = event.size
            screen = pygame.display.set_mode(video_size)
            print(video_size)
    return running, screen, envDone


class PlayPlot(object):
    def __init__(self, callback, horizon_timesteps, plot_names):
        self.data_callback = callback
        self.horizon_timesteps = horizon_timesteps
        self.plot_names = plot_names

        num_plots = len(self.plot_names)
        self.fig, self.ax = plt.subplots(num_plots)
        if num_plots == 1:
            self.ax = [self.ax]
        for axis, name in zip(self.ax, plot_names):
            axis.set_title(name)
        self.t = 0
        self.cur_plot = [None for _ in range(num_plots)]
        self.data = [deque(maxlen=horizon_timesteps) for _ in range(num_plots)]

    def callback(self, obs_t, obs_tp1, action, rew, done, info):
        points = self.data_callback(obs_t, obs_tp1, action, rew, done, info)
        for point, data_series in zip(points, self.data):
            data_series.append(point)
        self.t += 1

        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for i, plot in enumerate(self.cur_plot):
            if plot is not None:
                plot.remove()
            self.cur_plot[i] = self.ax[i].scatter(range(xmin, xmax), list(self.data[i]))
            self.ax[i].set_xlim(xmin, xmax)
        plt.pause(0.000001)
