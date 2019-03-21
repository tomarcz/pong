import torch
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim


class DqnAgent:
    def __init__(self, env, device, lr, gamma, batch_size, net, target_net, memory, epsilon_tracker):
        self.env = env
        self.device = device
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.net = net.to(device)
        self.target_net = target_net.to(device)
        self.update_net()
        self.memory = memory
        self.epsilon_tracker = epsilon_tracker

        self.state = self.env.reset()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_function = nn.MSELoss()

    def step(self):
        action = self.epsilon_greedy(self.state, self.epsilon_tracker.get())
        state_next, reward, done, _ = self.env.step(action)
        self.memory.push(self.state, action, state_next, reward, done)
        if done:
            self.state = self.env.reset()
        else:
            self.state = state_next
        return reward, done

    def optimize(self):
        states, actions, states_next, rewards, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        states_next = torch.tensor(states_next).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)

        outs = self.net(states).gather(1, actions.unsqueeze(-1))
        with torch.no_grad():
            targets = self.target_net(states_next).max(1)[0]
            targets[dones] = 0
            targets = rewards + self.gamma * targets
        targets = targets.unsqueeze(-1)
        loss = self.loss_function(outs, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def epsilon_greedy(self, state, epsilon=0.):
        if random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return self.optimal_action(state)

    def optimal_action(self, state):
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        x = self.net(x)
        return torch.argmax(x).item()

    def update_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def check(self):
        with torch.no_grad():
            x = torch.tensor([[self.state]], dtype=torch.float32).to(self.device)
        print(self.net(x))

    def test(self, env, render=False, fps=40):
        obs = env.reset()
        total_reward = 0
        while True:
            if render:
                env.render()
                time.sleep(1/fps)
            action = self.optimal_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        env.close()
        return total_reward
