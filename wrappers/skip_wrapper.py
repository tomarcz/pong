import gym


class SkipWrapper(gym.Wrapper):
    def __init__(self, env, skip_frames=4):
        super(SkipWrapper, self).__init__(env)
        self.skip_frames = skip_frames

    def reset(self):
        return self.env.reset()

    def step(self, action):
        total_reward = 0
        for _ in range(self.skip_frames):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
