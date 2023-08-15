import random

import numpy as np
import simpy


class Patient:
    """
        Defines a patient by his/her:
        acuity level, arrival time and leaving_time
    """

    patient_num = 0

    def __init__(self):
        self.env = simpy.Environment()
        Patient.patient_num += 1
        self.id = Patient.patient_num
        self.acuity_level = random.choices([1, 2, 3, 4, 5], [0.1, 0.3, 0.4, 0.1, 0.1])[0]
        self.arrival_time = self.env.now
        self.leave_time = 0
        self.ed_waiting_time = 0
        self.los = 0

    def get_acuity_level(self):
        assert self.acuity_level > 0
        return self.acuity_level

    def get_waiting_time(self):
        assert self.ed_waiting_time >= 0
        return self.ed_waiting_time

    def get_length_of_stay(self):
        self.los = self.leave_time - self.arrival_time
        return self.los