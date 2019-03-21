import gym
import numpy as np
from wrappers.frame_wrapper import FrameWrapper
from wrappers.color_wrapper import ColorWrapper
from wrappers.stack_wrapper import StackWrapper
from wrappers.skip_wrapper import SkipWrapper
from wrappers.random_begin_wrapper import RandomBeginWrapper


def PongEnv(shape=(84, 84), stack_frames=4):
    env = gym.make('PongNoFrameskip-v4')
    env = RandomBeginWrapper(env, n=55)
    env = SkipWrapper(env, skip_frames=4)
    env = ColorWrapper(env)
    env = FrameWrapper(env, shape)
    env = StackWrapper(env, n=stack_frames)
    return env
