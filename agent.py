# Copyright (c) 2025, Bongani. All rights reserved.
# This file is part of the Space War Shoot RL Version project.
# Author: Bongani Jele <jelebongani43@gmail.com>

########################################################################################
# This project implements a reinforcement learning agent using the Bellman Equation    #
# with PyTorch and NumPy in a Pygame-based space shooter game.                         #
#                                                                                      #
# Pygame doc: https://www.pygame.org/docs/ref/rect.html                                #
#                                                                                      #
# For just in case if you experience issues, email me or                               #
# contribute on GitHub — I'll appreciate your support!                                 #
########################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from collections import deque
from constants import WIDTH, HEIGHT, MAX_BULLETS

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self):
        self.model = DQN(6, 5)
        self.target_model = DQN(6, 5)
        self.target_model.load_state_dict(self.model.state_dict())
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_counter = 0
        
        self.training_rewards = []
        self.training_losses = []

        plt.ion()
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.show()
        self.fig.canvas.draw()

    def get_state(self, player, enemies):
        nearest_enemy = min(enemies.sprites(), key=lambda e: abs(e.rect.x - player.rect.x), default=None)
        if nearest_enemy:
            enemy_dx = nearest_enemy.rect.x - player.rect.x
            enemy_dy = nearest_enemy.rect.y - player.rect.y
        else:
            enemy_dx = WIDTH
            enemy_dy = 0
        return np.array([
            player.rect.y / HEIGHT,
            player.health / 3,
            enemy_dx / WIDTH,
            enemy_dy / HEIGHT,
            len(player.bullets) / MAX_BULLETS,
            len(enemies) / 10
        ], dtype=np.float32)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_losses.append(loss.item())
        self.update_counter += 1

        if self.update_counter % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            self.plot_training_progress()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def record_reward(self, reward):
        self.training_rewards.append(reward)

    def plot_training_progress(self):
        self.axs[0].clear()
        self.axs[1].clear()

        self.axs[0].set_title("Training Rewards Over Time")
        self.axs[0].set_xlabel("Episode")
        self.axs[0].set_ylabel("Reward")
        self.axs[0].step(range(len(self.training_rewards)),
                         self.training_rewards,
                         label="Reward per Episode",
                         where='mid',
                         color='blue',
                         linewidth=1.5)
        self.axs[0].legend()

        self.axs[1].set_title("Training Loss Over Time")
        self.axs[1].set_xlabel("Training Step")
        self.axs[1].set_ylabel("Loss")
        self.axs[1].step(range(len(self.training_losses)),
                         self.training_losses,
                         label="Loss",
                         where='mid',
                         color='red',
                         linewidth=1.5)
        self.axs[1].legend()

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
