import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from helper import episode_reward_plot
from collections import deque

from GymED import ER_Gym


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


# class VAnet(torch.nn.Module):
#
#     ''' 只有一层隐藏层的A网络和V网络 '''
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(VAnet, self).__init__()
#         self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 共享网络部分
#         self.fc_A = torch.nn.Linear(hidden_dim, action_dim)
#         self.fc_V = torch.nn.Linear(hidden_dim, 1)
#
#     def forward(self, x):
#         A = self.fc_A(F.relu(self.fc1(x)))
#         V = self.fc_V(F.relu(self.fc1(x)))
#         Q = V + A - A.mean(1).view(-1, 1)  # Q值由V值和A值计算得到
#         return Q

class QNet(torch.nn.Module):

    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=hidden_dim1),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim2, out_features=hidden_dim3),
            nn.ReLU()
        )
        # self.fc1 = torch.nn.Linear(state_dim, hidden_dim1)
        self.fc_A = torch.nn.Linear(in_features=hidden_dim3, out_features=action_dim)
        self.fc_V = torch.nn.Linear(in_features=hidden_dim3, out_features=1)

    def forward(self, x):
        tmp = self.fc1(x)
        V = self.fc_V(tmp)
        A = self.fc_A(tmp)
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


class AgentDuelingDQN:

    def __init__(self, env, replay_size=10000, batch_size=64, gamma=0.99, sync_after=10, lr=0.001):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError('Continuous actions not implemented!')
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.n
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # initialize the DQN and the target network
        self.dqn_net = QNet(state_dim=self.obs_dim, hidden_dim1=64, hidden_dim2=32, hidden_dim3=2, action_dim=self.act_dim).to(self.device)
        self.dqn_target_net = QNet(state_dim=self.obs_dim, hidden_dim1=64, hidden_dim2=32, hidden_dim3=2, action_dim=self.act_dim).to(self.device)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        # set up optimizer
        self.optim_dqn = optim.Adam(self.dqn_net.parameters(), lr=lr)
        # list to storage the rewards during learning process, used for visualization
        self.all_rewards = []

    def learn(self, timesteps):
        """
        train the agent for multiple times
        :param timesteps:
        :return:
        """
        all_rewards = []
        episode_rewards = []
        obs = self.env.reset()
        for timestep in range(1, timesteps + 1):
            sys.stdout.write('\rTimestep: {}/{}'.format(timestep, timesteps))
            sys.stdout.flush()

            epsilon = self.epsilon_decay(timestep)
            action = self.predict(obs, epsilon)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated)

            obs = next_obs
            episode_rewards.append(reward)
            # only start training when the buffer has stored enough samples
            if terminated or len(episode_rewards) >= 500:
                obs = self.env.reset()
                episode_len = len(episode_rewards)
                all_rewards.append(sum(episode_rewards))
                # if timestep % 1000 == 0:
                #     print(f'Timestep:{timestep}, Reward: {reward}.')
                episode_rewards = []

            if len(self.replay_buffer) > self.batch_size:
                loss = self.compute_loss()
                self.optim_dqn.zero_grad()
                loss.backward()
                self.optim_dqn.step()

            #  Synchronize the target network
            if timestep % self.sync_after == 0:
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
            # visualization
            if timestep % 500 == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
            if timestep == 20000:
                plt.savefig('./Rewards/Dueling_DQN_rewards.png')

    def predict(self, state, epsilon=0.0):
        """
        predict the best action based on state, apply epsilon-greedy policy.
        :param state:
        :param epsilon:
        :return: the action to take --> int
        """
        e = np.random.random()
        if e > epsilon:
            state = torch.FloatTensor(state).unsqueeze(
                0).to(self.device)  # inserts empty first dimension for potential batch-processing
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = np.random.randint(self.act_dim)
        return action

    def compute_loss(self):
        """

        :return:
        """
        obs, actions, rewards, next_obs, terminated = self.replay_buffer.get(self.batch_size)
        obs = torch.stack([torch.Tensor(ob) for ob in obs]).to(self.device)
        # shape (batch_size, state_dimensionality)
        # obs = torch.FloatTensor(obs).view(-1, 1).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        next_obs = torch.stack([torch.Tensor(ob) for ob in next_obs]).to(self.device)
        terminated = torch.Tensor(terminated).to(self.device)

        # Compute q_values and next_q_values -> shape (batch_size, num_actions)
        q_values = self.dqn_net(obs)

        # Select Q-values of actions actually taken -> shape (batch_size)
        q_values = q_values.gather(1, actions).squeeze(1)

        # Choose between target and no target network

        # Calculate max over next Q-values

        # next_q_values = self.dqn_net(next_obs).max(1)[0]
        next_q_values = self.dqn_target_net(next_obs).max(1)[0]
        # The target we want to update our network towards
        expected_q_values = rewards + self.gamma * (1.0 - terminated) * next_q_values

        # Calculate loss
        loss = F.mse_loss(q_values, expected_q_values)
        return loss

    def epsilon_decay(self, timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000):
        """
         epsilon decays linearly after times,
         aims at reducing the exploration and boosting learning process
        :param timestep:
        :param epsilon_start:
        :param epsilon_final:
        :param frames_decay:
        :return:
        """
        return max(epsilon_final,
                   epsilon_start - (float(timestep) / float(frames_decay)) * (epsilon_start - epsilon_final))

    def draw_learning_curve(self):
        plt.plot(self.all_rewards)
        plt.show()


if __name__ == '__main__':
    env = ER_Gym(num_doctors=20, num_nurses=20, sim_time=12*60)
    dqn = AgentDuelingDQN(env, batch_size=100, gamma=0.9999)
    dqn.learn(20000)
    # TODO save reward
    dqn.draw_learning_curve()
    print()