import gym
import numpy as np


class ColorWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ColorWrapper, self).__init__(env)
        new_shape = (self.observation_space.shape[0], self.observation_space.shape[1])
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=new_shape, dtype=np.float32)

    def observation(self, obs):
        return ColorWrapper.process(obs)

    @staticmethod
    def process(obs):
        obs = obs[:, :, 2] / 255.
        return obs
