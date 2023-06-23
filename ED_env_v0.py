import copy

import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np

import random
from typing import Any, Tuple, Dict, Optional

# easy version of Emergency Department
# only consider 5 Doctors taking care of the patients
# 7 min treatment for each patient


class ER_easy(Env):
    def __init__(self, num_doctors=5):
        self.num_idle_doc = num_doctors
        self.total_docs = num_doctors
        self.acuity_level = 5
        # self.waiting_patients = {'Patient Number': [],
        #                          'Acuity Level': []
        #                          }
        self.doc_lst = np.zeros(shape=(self.total_docs, 2))      # [if idle, working time]
        self.doc_lst[:, 0] = True

        self.au_distribution = np.array([0.1, 0.3, 0.4, 0.1, 0.1])

        self.working_time = 7   # 7 min treatment
        self.weighted_waiting_time = np.array([3, 1.5, 0.1, 0.1, 0.1])
        self.weighted_acuity_level = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        # self.capacity = 50      # maximum number of patients that ED can take
        self.initial_state = 20 * self.au_distribution  # at first 20 patients

        self.state = (copy.deepcopy(self.initial_state)).astype(int)
        self.num_patients = np.sum(self.state)
        self.capcity = 50

        self.observation_space = spaces.Box(low=0, high=1, shape=(5, ))
        self.action_space = spaces.Discrete(5)

        self.schedule_time = 1440       # 24h
        self.current_time = 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        # TODO
        assert self.action_space.contains(action)
        terminated = False
        self.current_time += 1

        # new patient comes with random acuity level
        au = np.random.choice(np.arange(0, 5), p=self.au_distribution)
        self.state[au] += 1

        # apply action if there were still doctor free
        if self.num_idle_doc > 0:
            # assign a doctor to the patient
            idle_index = np.random.choice(np.where(self.doc_lst[:, 0] == True)[0])
            # start to work

            # set the idle state to false
            self.doc_lst[idle_index, 0] = False
            # update the amount of free doctors
            self.num_idle_doc -= 1

            # apply actions
            if self.state[action] >= 1:
                self.state[action] -= 1

        # check if we can turn some doctors' states to idle
        self.doc_lst[self.doc_lst[:, 1] == self.working_time, 0] = True
        self.doc_lst[self.doc_lst[:, 1] == self.working_time, 1] = 0

        # update the working state of doctors
        self.doc_lst[(self.doc_lst[:, 0] == False) & (self.doc_lst[:, 1] < self.working_time), 1] += 1

        # calculate reward by weighting the waiting time
        self.num_patients = np.sum(self.state)
        reward = -np.sum(self.weighted_waiting_time *
                         self.weighted_acuity_level * self.state/self.num_patients)

        # see if the terminal
        if self.current_time >= self.schedule_time or self.num_patients > self.capcity:
            terminated = True

        # see how many doctors are still free
        self.num_idle_doc = np.count_nonzero(self.doc_lst[:, 0])

        return self._observe(), reward, terminated, False, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        # TODO
        self.current_time = 0
        self.num_idle_doc = self.total_docs
        self.doc_lst = np.zeros(shape=(self.total_docs, 2))  # [if idle, working time]
        self.doc_lst[:, 0] = True
        self.state = copy.deepcopy(self.initial_state)
        return self._observe()

    def _observe(self):
        # TODO
        # normalize the state vector
        return np.array(self.state / np.sum(self.state))

    def render(self, mode="human"):
        pass


if __name__ == '__main__':
    env = ER_easy()
    print()
    episode = 10
    for eps in range(1, episode+1):
        state = env.reset()
        done = False
        eps_len = 0
        score = 0
        while not done:
            action = env.action_space.sample()
            n_state, r, done, truncated, info = env.step(action)
            eps_len += 1
            score += r
        print('Episode:{}, Length:{}, Score:{}'.format(eps,eps_len, score))
