import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

from ED_env_v0 import ER_easy


class ReplayBuffer:
    """
    A replay buffer for experience replay,
    as commonly used for off-policy Q-Learning methods.
    """

    def __init__(self, capacity):
        """

        :param capacity:
        """

        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def put(self, obs, action, reward, next_obs, terminated):
        """

        Put a tuple of (obs, action, rewards, next_obs, terminated) into the replay buffer.
        The max length specified by capacity should never be exceeded.
        The oldest elements inside the replay buffer should be overwritten first.
        :param obs: state
        :param action:
        :param reward:
        :param next_obs:
        :param terminated: flag for ending state
        :return:
        """

        self.buffer.append((obs, action, reward, next_obs, terminated))

    def get(self, batch_size):
        """
        Gives batch_size samples from the replay buffer.
        :param batch_size:
        :return:
        """
        transition = random.sample(population=self.buffer, k=batch_size)
        S, A, R, S_, Done = zip(*transition)
        return S, A, R, S_, Done

    def __len__(self):
        """
        Returns the number of tuples inside the replay buffer.
        :return:
        """
        return len(self.buffer)


class QNet(torch.nn.Module):
    """
    Q network with single hidden layer, Value Function Approximation by Q-Network
    """

    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim):
        """

        :param state_dim: number of states
        :param hidden_dim:  hidden units
        :param action_dim: number of actions
        """
        super(QNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim2, out_features=hidden_dim3),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim3, out_features=action_dim)
        )

    def forward(self, x):
        """

        :param x: input features
        :return: actions
        """
        return self.layers(x)


class AgentDDQN:
    """
    Double DQN method
    """

    def __init__(self, env, replay_size=10000, batch_size=64, gamma=0.99, sync_after=10, lr=0.001):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # initialize the DQN network
        self.dqn_net = QNet(state_dim=self.obs_dim, hidden_dim1=64, hidden_dim2=32, hidden_dim3=2,
                            action_dim=self.act_dim).to(self.device)
        self.dqn_target_net = QNet(self.obs_dim, 64, 32, 2, self.act_dim).to(self.device)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        # set up optimizer
        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)

        # list to storage the rewards during learning process, used for visualization
        self.all_rewards = []

    def learn(self, timesteps):
        episode_rewards = []
        obs = self.env.reset()
        for timestep in range(1, timesteps + 1):
            # sys.stdout.write('\rTimestep: {}/{}'.format(timestep, timesteps))
            # sys.stdout.flush()

            epsilon = self.epsilon_decay(timestep)
            action = self.predict(obs, epsilon)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_rewards.append(reward)
            # only start training when the buffer has stored enough samples
            if terminated or len(episode_rewards) >= 1000:
                obs = self.env.reset()
                episode_len = len(episode_rewards)
                reward = sum(episode_rewards)
                self.all_rewards.append(reward)
                if timestep % 1000 == 0:
                    print(f'Timestep:{timestep}, Reward: {reward}.')
                episode_rewards = []

            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_loss()
                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            #  Synchronize the target network
            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

    def predict(self, state, epsilon=0.0):
        e = np.random.random()
        if e >= epsilon:
            state = torch.FloatTensor(state).unsqueeze(
                0).to(self.device)
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = np.random.randint(self.act_dim)
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.dqn_net(state).max().item()

    def compute_loss(self):
        obs, actions, rewards, next_obs, terminated = self.replay_buffer.get(self.batch_size)
        obs = torch.stack([torch.Tensor(ob) for ob in obs]).to(self.device)
        # obs = torch.FloatTensor(obs).view(-1, 1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_obs = torch.stack([torch.Tensor(ob) for ob in next_obs]).to(self.device)
        terminated = torch.Tensor(terminated).to(self.device)

        q_values = self.dqn_net(obs).gather(1, actions)

        action_with_max_q = self.dqn_net(next_obs).max(1)[1].view(-1, 1)
        max_next_q_values = self.dqn_target_net(next_obs).gather(1, action_with_max_q)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - terminated)
        loss = F.mse_loss(q_values, q_targets)
        return loss

    def epsilon_decay(self, timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
        return max(epsilon_final,
                   epsilon_start - (float(timestep) / float(frames_decay)) * (epsilon_start - epsilon_final))

    def draw_learning_curve(self):
        plt.plot(self.all_rewards)
        plt.show()


if __name__ == '__main__':
    env = ER_easy()
    dqn = AgentDDQN(env)
    dqn.learn(50000)
    dqn.draw_learning_curve()
    print()