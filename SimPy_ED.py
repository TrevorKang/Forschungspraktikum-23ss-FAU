import simpy
import random
import numpy as np
import statistics
from patient import Patient


class EmergencyDepartment:

    patient_count = 0

    def __init__(self, num_doctors=3, num_nurses=5, num_CTs=3, num_lab=4, sim_time=12*60, seed=None):
        # TODO
        if seed is not None:
            random.seed(seed)
        self.env = simpy.Environment()

        self.num_doctors = num_doctors
        self.doctor = simpy.Resource(self.env, capacity=self.num_doctors)
        self.num_nurses = num_nurses
        self.nurse = simpy.Resource(self.env, capacity=self.num_nurses)
        self.num_CTs = num_CTs
        self.CT = simpy.Resource(self.env, capacity=self.num_CTs)
        self.num_lab = num_lab
        self.lab = simpy.Resource(self.env, capacity=self.num_lab)

        self.simulation_time = sim_time

        self.patient_lst = []
        self.inter_arrival_time = 0
        self.ed_waiting_room = []

        self.ed_waiting_room_len = 0
        self.patients_processed = 0

        self.waiting_time_total = []
        self.waiting_time_au = {"1": [], "2": [], "3": [], "4": [], "5": []}

        self.los_total = []
        self.los_au = {"1": [], "2": [], "3": [], "4": [], "5": []}

    def reset_env(self):
        EmergencyDepartment.patient_count = 0

        self.doctor = simpy.Resource(self.env, capacity=self.num_doctors)
        self.nurse = simpy.Resource(self.env, capacity=self.num_nurses)
        self.CT = simpy.Resource(self.env, capacity=self.num_CTs)
        self.lab = simpy.Resource(self.env, capacity=self.num_lab)

        self.patient_lst = []
        self.inter_arrival_time = 0
        self.ed_waiting_room = []

        self.ed_waiting_room_len = 0
        self.patients_processed = 0

        self.waiting_time_total = []
        self.waiting_time_au = {"1": [], "2": [], "3": [], "4": [], "5": []}

        self.los_total = []
        self.los_au = {"1": [], "2": [], "3": [], "4": [], "5": []}

    def warm_up(self):
        while len(self.ed_waiting_room) <= 20:
            self.inter_arrival_time = random.expovariate(7)
            self.patient_count += 1
            patient = Patient()
            self.patient_lst.append(patient)
            self.ed_waiting_room.append(patient.id)
            yield self.env.timeout(self.inter_arrival_time)

    def patients_generation(self):
        # TODO
        while True:
            self.inter_arrival_time = random.expovariate(1/7)
            self.patient_count += 1
            patient = Patient()
            self.patient_lst.append(patient)
            # print(f"Patient {patient.id} produced")
            self.ed_waiting_room.append(patient.id)
            self.env.process(self.patient_flow(patient))
            yield self.env.timeout(self.inter_arrival_time)

    def patient_flow(self, patient: Patient):
        # TODO
        triage_queue_start = self.env.now
        # print(f"Patient {patient.id} enters ED waiting room.")
        # self.ed_waiting_room.append(patient.id)
        # Triage
        with self.doctor.request() as doctor_request:
            yield doctor_request
            # print(f"Patient {patient.id} leaves ED waiting room")
            self.ed_waiting_room.remove(patient.id)
            triage_queue_end = self.env.now
            patient.ed_waiting_time = triage_queue_end - triage_queue_start
            self.waiting_time_au.get(str(patient.acuity_level)).append(patient.ed_waiting_time)
            self.waiting_time_total.append(patient.ed_waiting_time)
            delay_triage = random.expovariate(1/7)
            yield self.env.timeout(delay_triage)
            self.doctor.release(doctor_request)

        # Registration
        with self.nurse.request() as nurse_request:
            yield nurse_request
            delay_registration = random.expovariate(1/5.5)
            yield self.env.timeout(delay_registration)
            self.nurse.release(nurse_request)

        # further treatments
        if patient.get_acuity_level() == 1:
            yield self.env.process(self.au_level_1_process(patient))

        else:
            # evaluation:
            with self.doctor.request() as doctor_request:
                yield doctor_request
                eva_delay = random.normalvariate(mu=14, sigma=6)
                yield self.env.timeout(abs(eva_delay))
                self.doctor.release(doctor_request)

            # CT scan or Lab Test, then discharge
            yield self.env.process(self.none_au_1_process(patient))

            patient.leave_time = self.env.now
            # print(f"Patient{patient.id} leaves ED and is discharged")
            self.los_total.append(patient.get_length_of_stay())
            self.los_au.get(str(patient.acuity_level)).append(patient.get_length_of_stay())
            self.patients_processed += 1

    def au_level_1_process(self, patient):
        # TODO
        # send to resuscitation room -> discharge
        with self.doctor.request() as doctor_request:
            yield doctor_request
            with self.nurse.request() as nurse_request:
                yield nurse_request
                delay_transfer = random.expovariate(1/5)
                yield self.env.timeout(delay_transfer)
                patient.leave_time = self.env.now
                self.doctor.release(doctor_request)
                self.nurse.release(nurse_request)
                self.los_total.append(patient.get_length_of_stay())
                self.los_au.get(str(patient.acuity_level)).append(patient.get_length_of_stay())
                self.patients_processed += 1
                # print(f"Patient {patient.id} has send to resuscitation room.")

    def none_au_1_process(self, patient):
        # TODO
        # CT scan or Lab Test
        if random.random() > 0.7:
            if random.random() > 0.5:
                with self.CT.request() as CT_request:
                    yield CT_request
                    CT_delay = np.random.lognormal(mean=29, sigma=14)
                    yield self.env.timeout(CT_delay)
                    self.CT.release(CT_request)
            else:
                with self.lab.request() as lab_request:
                    yield lab_request
                    with self.nurse.request() as nurse_lab_request:
                        yield nurse_lab_request
                        lab_decay = np.random.lognormal(mean=35, sigma=15)
                        yield self.env.timeout(lab_decay)
                        self.lab.release(lab_request)
                        self.nurse.release(nurse_lab_request)
        # consultant
        if random.random() <= 0.1:
            with self.nurse.request() as nurse_con_request:
                yield nurse_con_request
                consultant_delay = np.random.lognormal(mean=15, sigma=8)
                yield self.env.timeout(consultant_delay)
                self.nurse.release(nurse_con_request)

        # discharge
        with self.nurse.request() as nurse_disc_request:
            yield nurse_disc_request
            discharge_delay = 10
            yield self.env.timeout(discharge_delay)
            self.nurse.release(nurse_disc_request)

    def start_simulation(self):

        self.env.process(self.patients_generation())
        self.env.run(until=self.simulation_time)
        self.ed_waiting_room_len = len(self.ed_waiting_room)

    def calculate_avg_waiting_time(self):
        if len(self.waiting_time_total) != 0:
            return np.average(self.waiting_time_total) / 60
        else:
            return 0

    def calculate_avg_los(self):
        if len(self.los_total) != 0:
            return np.average(self.los_total)
        else:
            return 0

    def calculate_avg_weighted_waiting_time(self):
        # TODO: avoid NAN
        # wwt1 = 5 * np.average(self.waiting_time_au['1']) / 60
        # wwt2 = 4 * np.average(self.waiting_time_au['2']) / 60
        # wwt3 = 3 * np.average(self.waiting_time_au['3']) / 60
        # wwt4 = 2 * np.average(self.waiting_time_au['4']) / 60
        # wwt5 = 1 * np.average(self.waiting_time_au['5']) / 60
        sum = 0
        for k in range(len(self.waiting_time_au)):
            key = str(k + 1)
            if len(self.waiting_time_au[key]) != 0:
                sum += (0.1 * (5 - k) * np.sum(self.waiting_time_au[key]) / len(self.waiting_time_au[key]))

        return sum / 60

    def calculate_avg_weighted_los(self):
        # TODO: avoid NAN
        # wwt1 = 0.5 * np.average(self.los_au['1']) / 60
        # wwt2 = 0.4 * np.average(self.los_au['2']) / 60
        # wwt3 = 0.3 * np.average(self.los_au['3']) / 60
        # wwt4 = 0.2 * np.average(self.los_au['4']) / 60
        # wwt5 = 0.1 * np.average(self.los_au['5']) / 60
        sum = 0
        for k in range(len(self.los_au)):
                key = str(k+1)
                if len(self.los_au[key]) != 0:
                    sum += (0.1 * (5 - k) * np.sum(self.los_au[key]) / len(self.los_au[key]))
        return sum / 60

    def output(self):
        print(f'Simulation Time: {self.simulation_time / 60} Hour(s)\n')
        print(f'Total Patients Generation: {self.patient_count}'
              f'\n')
        print(f'Total Patients Processed: {self.patients_processed}'
              f'\n')
        print(f'Number of Patients in ED Waiting Room: {self.ed_waiting_room_len}'
              f'\n')
        print(f'Resource remaining: \n'
              f'Idle doctor(s): {self.num_doctors - self.doctor.count}\n'
              f'Idle nurse(s): {self.num_nurses - self.nurse.count}\n'
              f'Idle CT-Machine(s): {self.num_CTs - self.CT.count}\n'
              f'Idle laboratory(s): {self.num_lab - self.lab.count}\n'
              f'')

        print(f'Avg waiting time: {self.calculate_avg_waiting_time()} Hour(s)')
        print(f'Avg length of stay: {self.calculate_avg_los()} Hour(s)')
        # print(f'Avg weighted waiting time:{self.calculate_avg_weighted_waiting_time()} Hour(s)')
        # print(f'Avg weighted LOS:{self.calculate_avg_weighted_los()} Hour(s)')


if __name__ == '__main__':
    for i in range(10):
        ED_env = EmergencyDepartment(num_doctors=3, num_nurses=5, num_CTs=2, num_lab=3, sim_time=12*60, seed=258)

        ED_env.start_simulation()
        ED_env.output()

