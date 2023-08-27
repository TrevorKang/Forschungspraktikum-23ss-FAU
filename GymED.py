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
        self.total_docs_to_assign = num_doctors
        self.total_nurses_to_assign = num_nurses

        # idle resource
        # self.idle = np.array([self.total_docs, self.total_nurses])
        self.idle_docs = self.total_docs_to_assign
        self.idle_nurses = self.total_nurses_to_assign

        self.num_working_docs = 1
        self.num_working_nurse = 1

        self.sim_time = sim_time
        self.random_seed = seed


        # ED environment
        # self.sim_ED_Env = None

        self.observation_space = spaces.Box(low=0, high=1000, shape=(2, ))
        self.action_space = spaces.Discrete(5)
        # 0: keep
        # 1: add 1 doctor
        # 2: minus 1 doctor
        # 3: add 1 nurse
        # 4: minus 1 nurse

        self.state = np.zeros(shape=(2, ))
        # [
        #  # working docs,
        #  # working nurses
        # ]

        self.schedule_step = 365
        self.current_step = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # TODO
        assert self.action_space.contains(action)
        terminated = False
        self.current_step += 1
        # TODO:
        # assign actions
        if action == 1:
            self.idle_docs -= 1
            self.num_working_docs += 1
        elif action == 2:
            self.idle_docs += 1
            self.num_working_docs += 1
        elif action == 3:
            self.idle_nurses -= 1
            self.num_working_nurse += 1
        elif action == 4:
            self.idle_nurses += 1
            self.num_working_nurse -= 1

        # assert if there's still working doctors/nurses
        # TODO
        if self.num_working_nurse <= 0 or self.num_working_docs <= 0:
            terminated = True
        # assert if there's doctor/nurse remains
        # TODO
        if self.idle_nurses <= 0 or self.idle_docs <= 0:
            terminated = True
        if self.current_step >= self.schedule_step:
            terminated = True
        if not terminated:
            sim_ED_Env = EmergencyDepartment(num_doctors=self.num_working_docs, num_nurses=self.num_working_nurse,
                                                  sim_time=self.sim_time)
            sim_ED_Env.start_simulation()
            # self.state = np.array([self.num_working_docs, self.num_working_docs])
            self.state = np.array([sim_ED_Env.calculate_avg_weighted_waiting_time(), sim_ED_Env.calculate_avg_weighted_los()])
            # reward design
            reward = -(sim_ED_Env.calculate_avg_weighted_waiting_time()
                       + sim_ED_Env.calculate_avg_weighted_los()) + \
                     0.1 * sim_ED_Env.patients_processed / sim_ED_Env.patient_count + \
                     0.1 * self.idle_nurses / self.total_nurses_to_assign + \
                     0.1 * self.idle_docs / self.total_docs_to_assign
        else:
            # self.sim_ED_Env = EmergencyDepartment(num_doctors=3, num_nurses=5,
            #                                       sim_time=self.sim_time)
            # self.sim_ED_Env.start_simulation()
            # self.state = np.array([self.sim_ED_Env.calculate_avg_waiting_time(),
            #                        self.sim_ED_Env.calculate_avg_los()])
            reward = -5
        # self.state = np.array([self.num_working_docs, self.num_working_nurse])
        return self.state, reward, terminated, False, {}

    def seed(self):
        random.seed = self.random_seed

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        # TODO
        self.current_step = 0
        self.num_working_docs = 1
        self.num_working_nurse = 1
        self.idle_nurses = self.total_docs_to_assign
        self.idle_docs = self.total_nurses_to_assign
        self.state = np.zeros(shape=(2, ))
        # self.sim_ED_Env = None
        return self.state

    def render(self, mode="human"):
        print(f'Current working doctors: {self.num_working_docs}')
        print(f'Current working nurses: {self.num_working_nurse}')
        print()


if __name__ == '__main__':
    env = ER_Gym(10, 10, sim_time=12*60, seed=58)
    env.seed()

    episode = 10
    for eps in range(1, episode + 1):
        state = env.reset()
        done = False
        eps_len = 0
        score = 0
        while not done:
            action = env.action_space.sample()
            n_state, r, done, truncated, info = env.step(action)
            eps_len += 1
            score += r
        print('Episode:{}, Length:{}, Score:{}'.format(eps, eps_len, score))