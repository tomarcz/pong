import gym
import cv2
import numpy as np


class FrameWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(100, 100)):
        super(FrameWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(1, *shape), dtype=np.float32)

    def observation(self, obs):
        return FrameWrapper.process(obs, (self.observation_space.shape[1], self.observation_space.shape[-1]))

    @staticmethod
    def process(obs, shape):
        obs = obs[34:-16, :]
        obs = cv2.resize(obs, shape)
        obs = np.reshape(obs, [1, shape[0], shape[1]])
        return obs
