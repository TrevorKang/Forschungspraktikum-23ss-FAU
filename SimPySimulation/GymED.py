import copy

import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np

import random
from typing import Any, Tuple, Dict, Optional

from SimPy_ED import EmergencyDepartment

"""
    reward design: 
    waiting time, LOS, severe patients, au levels, processed patient
"""


class ER_Gym(Env):
    def __init__(self, num_doctors, num_nurses, sim_time, seed=258):
        # total resource to assign
        self.total_docs = num_doctors
        self.total_nurses = num_nurses

        # idle resource
        self.idle = np.array([self.total_docs, self.total_nurses])

        self.sim_time = sim_time
        self.random_seed = seed

        self.num_docs = 3
        self.num_nurse = 5
        # ED environment
        self.sim_ED_Env = None

        self.observation_space = spaces.Box(low=0, high=1000, shape=(2, ))
        self.action_space = spaces.Discrete(5)
        # 0: keep
        # 1: add 1 doctor
        # 2: minus 1 doctor
        # 3: add 1 nurse
        # 4: minus 1 nurse

        self.state = np.zeros(shape=(2, ))
        # [
        #  avg weighted waiting time,
        #  avg_weighted_los,
        #  waiting room % treating room,
        #  %processed patient
        # ]

        self.schedule_step = 365
        self.current_step = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        # TODO
        assert self.action_space.contains(action)
        terminated = False
        if self.idle[0] == 0 and self.idle[1] == 0:     # all resource is assigned
            terminated = True
        # assign doctor or nurse

        if action == 1 and self.idle[0] > 0:
            self.num_docs += 1
            self.idle[0] -= 1
        elif action == 2 and self.num_docs > 0:
            self.num_docs -= 1
            self.idle[0] += 1
        elif action == 3 and self.idle[1] > 0:
            self.num_nurse += 1
            self.idle[1] -= 1
        elif action == 4 and self.num_nurse > 0:
            self.num_nurse -= 1
            self.idle[1] += 1

        if self.num_docs > 0 and self.num_nurse > 0:
            sim_ED_Env = EmergencyDepartment(num_doctors=self.num_docs, num_nurses=self.num_nurse, sim_time=self.sim_time)
        else:
            sim_ED_Env = EmergencyDepartment(sim_time=self.sim_time)
        sim_ED_Env.start_simulation()

        self.state[0] = sim_ED_Env.calculate_avg_waiting_time()
        self.state[1] = sim_ED_Env.calculate_avg_los()
        # print(self.state)
        # reward design
        reward = - sim_ED_Env.calculate_avg_waiting_time() - sim_ED_Env.calculate_avg_weighted_los()
        reward += (0.5 * self.idle[0] + 0.1 * self.idle[1])
        if self.current_step >= self.schedule_step:
            terminated = True
        if self.num_docs == 0 or self.num_nurse == 0:
            terminated = True
        return self.state, reward, terminated, False, {}

    def seed(self):
        random.seed = self.random_seed

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.current_step = 0
        self.idle = np.array([self.total_docs, self.total_nurses])
        self.num_docs = 3
        self.num_nurse = 5
        self.state = np.zeros(shape=(2,))
        return self.state, {}

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = ER_Gym(20, 20, sim_time=60*8, seed=58)
    env.seed()

    episode = 10
    for eps in range(1, episode + 1):
        state, _ = env.reset()
        done = False
        eps_len = 0
        score = 0
        while not done:
            action = env.action_space.sample()
            n_state, r, done, truncated, info = env.step(action)
            eps_len += 1
            score += r
        print('Episode:{}, Length:{}, Score:{}'.format(eps, eps_len, score))