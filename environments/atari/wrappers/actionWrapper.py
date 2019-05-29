"""An environment wrapper to convert binary to discrete action space."""
from gym import spaces, Wrapper
import numpy as np
from gym.envs.atari.atari_env import AtariEnv


class ActionWrapper(Wrapper):

    def __init__(self, env, discrete):
        """
        Initialize a new binary to discrete action space wrapper.

        Args:
            env (gym.Env): the environment to wrap
            actions (list): an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """
        super(ActionWrapper, self).__init__(env)
        self.discrete = discrete
        # create the new action space
        # self.action_space = spaces.Box(low=0, high=17, dtype=np.int32, shape=(1,))
        if isinstance(env.unwrapped, AtariEnv):
            (screen_width, screen_height) = self.env.unwrapped.ale.getScreenDims()
            self.screen_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

        if not self.discrete:
            self.action_space = spaces.MultiDiscrete([env.action_space.n])

    def step(self, action):
        if self.discrete or isinstance(action, int):
            return self.env.step(action)

        if isinstance(self.env.unwrapped, AtariEnv):
            return self.env.step(action[0])
        return self.env.step(int(action[0]))

    def reset(self):
        """Reset the environment and return the initial observation."""
        return self.env.reset()

    def getImage(self):
        atari_env = self.env.unwrapped
        return atari_env.ale.getScreenRGB2()


# explicitly define the outward facing API of this module
__all__ = [ActionWrapper.__name__]
