import gym
import numpy
import time
import torch
import matplotlib.pyplot as plt
from nets.net import Net
from wrappers.pong_env import PongEnv
from agents.dqn_agent import DqnAgent
from agents.linear_epsilon_tracker import LinearEpsilonTracker
from containers.memory import Memory

INPUT_SHAPE = (84, 84)
LEARNING_RATE = 0.0001
DECAY_RATE = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY_TO = 1000000
START_FROM = 80000
MEMORY_SIZE = 80000
BATCH_SIZE = 32
UPDATE_NET_EVERY = 1000
STACK_FRAMES = 4

CREATE_NEW_NET = False
LOAD_NET_PATH = 'data/net.torch'
SAVE_NET_PATH = 'data/net.torch'


env = PongEnv(shape=INPUT_SHAPE, stack_frames=STACK_FRAMES)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if CREATE_NEW_NET:
    net = Net(input_shape=INPUT_SHAPE, input_channels=STACK_FRAMES, output_size=env.action_space.n)
else:
    net = torch.load(LOAD_NET_PATH)
target_net = Net(input_shape=INPUT_SHAPE, input_channels=STACK_FRAMES, output_size=env.action_space.n)
epsilon_tracker = LinearEpsilonTracker(EPS_START, EPS_END, EPS_DECAY_TO)
memory = Memory(maxlen=MEMORY_SIZE)
agent = DqnAgent(env=env, device=device, lr=LEARNING_RATE, gamma=DECAY_RATE, batch_size=BATCH_SIZE,
                 net=net, target_net=target_net, memory=memory, epsilon_tracker=epsilon_tracker)
agent.update_net()


def train():
    total_reward = 0
    step = 1
    time_start = time.time()
    frame_start = 0

    while True:
        reward, done = agent.step()
        total_reward += reward
        if step >= START_FROM:
            agent.optimize()
        if step % UPDATE_NET_EVERY == 0:
            agent.update_net()
        if done:
            torch.save(net, "data/net.torch")
            fps = (step - frame_start) / (time.time() - time_start)
            print("%d | Reward: %.2f | Epsilon: %.2f | FPS: %.2f" % (step, total_reward, epsilon_tracker.get_value(), fps))
            frame_start = step
            time_start = time.time()
            total_reward = 0
        step += 1


def test(test_episodes=1, render=False, fps=40):
    test_env = PongEnv(shape=INPUT_SHAPE, stack_frames=STACK_FRAMES)
    total_rewards = 0
    for i in range(test_episodes):
        total_rewards += agent.test(test_env, render, fps)
        print('Episode %d done.' % i)

    print('Mean reward: %.3f' % (total_rewards/test_episodes))


if __name__ == "__main__":
    test(5)
