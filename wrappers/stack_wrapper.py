import gym
import numpy as np
import collections


class StackWrapper(gym.ObservationWrapper):
    def __init__(self, env, n=4, dtype=np.float32):
        super(StackWrapper, self).__init__(env)
        self.dtype = dtype
        self.buffer = None
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n, axis=0),
                                                old_space.high.repeat(n, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return np.array(self.buffer).astype(np.float32)
