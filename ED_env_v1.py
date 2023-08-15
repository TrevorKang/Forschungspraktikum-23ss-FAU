import copy

import gymnasium as gym
from gymnasium import Env
from gymnasium import spaces, logger
from gymnasium.utils import seeding
import numpy as np

import random
from typing import Any, Tuple, Dict, Optional

"""
    Modified version of Emergency Department
    Consider the following simplified workflow:
        Registration(Nurse) -> Triage & Evaluation(Doctor) -> CT Scan(Technician) -> Laboratory(Lab Doctors)
"""


class EmergencyDepartmentEnv(Env):
    def __init__(self, num_nurses=10, num_docs=5, num_ct=3, num_lab=3):
        self.num_nurses = num_nurses
        self.num_docs = num_docs
        self.num_ct = num_ct
        self.num_lab = num_lab

        self.idle_resource_total = {"Nurses": num_nurses, "Doctors": num_docs, "CT": num_ct, "Laboratory": num_lab}
        self.resource_lst = {"Nurses": np.zeros(shape=(self.idle_resource_total["Nurses"], 2)),
                             "Doctors": np.zeros(shape=(self.idle_resource_total["Doctors"], 2)),
                             "CT": np.zeros(shape=(self.idle_resource_total["CT"], 2)),
                             "Laboratory": np.zeros(shape=(self.idle_resource_total["Laboratory"], 2))}
        # [if idle, working time]
        self.workflow = list(self.idle_resource_total.keys())
        for k in self.workflow:
            self.resource_lst[k][:, 0] = True
        # 7 min for registration,
        # 10 min for triage and evaluation, 20 min for CT scan and for lab
        self.working_time = [7, 10, 20, 20]

        self.acuity_level = 5
        self.au_distribution = np.array([0.1, 0.3, 0.4, 0.1, 0.1])

        # design rewards
        self.weighted_waiting_time = np.array([3, 1.5, 0.1, 0.1, 0.1])
        self.weighted_acuity_level = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        self.capacity = 200  # maximum number of patients that ED can take

        # define state and action space
        self.initial_state = np.tile(10 * self.au_distribution, 4).reshape(4, 5)
        # self.initial_state = np.zeros(shape=(4, 5))
        self.observation_space = spaces.Box(low=0, high=1, shape=(4, 5))
        self.state = (copy.deepcopy(self.initial_state)).astype(int)
        # self.action_space = spaces.Box(low=0, high=4, shape=(4,), dtype=np.uint8)
        self.action_space = spaces.MultiDiscrete([5, 5, 5, 5])

        self.num_patients = np.sum(self.state)
        self.schedule_time = 1440  # 24h
        self.current_time = 0

    def step(
        self, action: np.ndarray(shape=(4, ))
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # TODO
        terminated = False
        self.current_time += 1

        # new patient comes with random acuity level
        # TODO: add exponential distribution of patient incoming
        prob = random.expovariate(lambd=7)
        if random.random() > prob:
            au = np.random.choice(5, p=self.au_distribution)
            room = np.random.randint(low=0, high=4)
            self.state[room, au] += 1
            # for i in range(len(au)):
            #     self.state[i, au[i]] += 1

        # apply action if there is still idle resource
        # e.g. action = [1, 3, 2, 4]
        for i in range(len(action)):
            key = self.workflow[i]  # 'Nurse' or 'Doctors' or ...
            lst_single = copy.deepcopy(self.resource_lst[key])

            if self.idle_resource_total[key] > 0:     # check if still remain idle person
                # assign a person to the patient
                idle_index = np.random.choice(np.where(lst_single[:, 0] == True)[0])

                # start to work
                # set the idle state to false
                lst_single[idle_index, 0] = False
                # update the amount of free person
                self.idle_resource_total[key] -= 1

                # apply actions
                if self.state[i, action[i]] >= 1:
                    self.state[i, action[i]] -= 1

            # check if we can turn some personals' states to idle
            lst_single[lst_single[:, 1] == self.working_time[i], 0] = True
            lst_single[lst_single[:, 1] == self.working_time[i], 1] = 0

            # update the working state of staff
            lst_single[(lst_single[:, 0] == False) & (lst_single[:, 1] < self.working_time[i]), 1] += 1

            # see how many doctors/nurses are still free
            self.idle_resource_total[key] = np.count_nonzero(lst_single[:, 0])
            self.resource_lst[key] = lst_single

        # calculate reward by weighting the waiting time
        self.num_patients = np.sum(self.state)
        num_severe_patient = np.sum(self.state[:, 0]) + np.sum(self.state[:, 1])
        reward = - np.sum(self.weighted_acuity_level * self.weighted_waiting_time * self.state / self.num_patients)
        # see if the terminal
        if self.current_time >= self.schedule_time or self.num_patients > self.capacity:
            terminated = True

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
        self.current_time = 0
        self.idle_resource_total = {"Nurses": self.num_nurses, "Doctors": self.num_docs,
                                    "CT": self.num_ct, "Laboratory": self.num_lab}
        self.resource_lst = {"Nurses": np.zeros(shape=(self.idle_resource_total["Nurses"], 2)),
                             "Doctors": np.zeros(shape=(self.idle_resource_total["Doctors"], 2)),
                             "CT": np.zeros(shape=(self.idle_resource_total["CT"], 2)),
                             "Laboratory": np.zeros(shape=(self.idle_resource_total["Laboratory"], 2))}
        for k in self.workflow:
            self.resource_lst[k][:, 0] = True
        self.state = copy.deepcopy(self.initial_state)
        return self._observe(), {}

    def _observe(self):
        return self.state / (np.expand_dims(np.sum(self.state, axis=1), axis=1) + np.finfo(np.float32).eps)


if __name__ == '__main__':
    env = EmergencyDepartmentEnv()
    env.seed(seed=42)

    print()
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
