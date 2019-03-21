import random
import numpy as np
import collections
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'state_next', 'reward', 'done'))


class Memory:
    def __init__(self, maxlen):
        self.stack = collections.deque(maxlen=maxlen)

    def push(self, *args):
        transition = Transition(*args)
        self.stack.append(transition)

    def sample(self, batch):
        num = random.sample(range(len(self.stack)), batch)
        states = []
        actions = []
        states_next = []
        rewards = []
        dones = []
        for i in num:
            states.append(self.stack[i].state)
            actions.append(self.stack[i].action)
            states_next.append(self.stack[i].state_next)
            rewards.append(self.stack[i].reward)
            dones.append(self.stack[i].done)
        states = np.array(states, np.float32)
        actions = np.array(actions, np.int64)
        states_next = np.array(states_next, np.float32)
        rewards = np.array(rewards, np.float32)
        dones = np.array(dones, np.uint8)
        return states, actions, states_next, rewards, dones

    def __len__(self):
        return len(self.stack)
