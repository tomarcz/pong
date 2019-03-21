import gym


class RandomBeginWrapper(gym.Wrapper):
    def __init__(self, env, n=55):
        super(RandomBeginWrapper, self).__init__(env)
        self.n  = n

    def reset(self):
        self.env.reset()
        total = 0
        for _ in range(self.n):
            obs, reward, done, info = self.env.step(self.env.action_space.sample())
            total += reward
        return obs

    def step(self, action):
        return self.env.step(action)
