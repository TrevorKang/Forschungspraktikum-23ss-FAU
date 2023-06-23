from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

from typing import Any, Tuple, Dict, Optional


class ShowerEnv(Env):
    def __init__(self):
        # actions to take, down(0), stay(1), up(2)
        self.action_space = Discrete(3)   # {0, 1, 2}
        # temperature as state
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # initial state: start temperature
        self.state = 38 + np.random.randint(-3, 3)
        # set up the shower length as 60 seconds
        self.shower_length = 60

    def step(self, action: int):
        # apply action
        self.state += (action - 1)
        # reduce shower length by 1 min
        self.shower_length -= 1
        # counting the rewards
        if 37 <= self.state <= 39:
            reward = 1
        else:
            reward = -1
        # check if the agent reaches the terminal state
        if self.shower_length <= 0:
            done = True
        else:
            done = False
        # Add some noise
        # self.state += random.randint(-1, 1)
        observation = self._observe()

        return observation, reward, done, False, {}

    def render(self):
        # visualize the implementation
        return

    def reset(self):
        # reset the water temperature and shower time
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return np.array([self.state, ])

    def _observe(self):
        return np.array([self.state, ])


if __name__ == '__main__':

    # test implementation

    env = ShowerEnv()
    print(env.observation_space.sample())

    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))