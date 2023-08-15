import simpy
import random
import statistics

waiting_time = []


class Theater:
    def __init__(self, env: simpy.Environment, num_cashiers, num_ushers, num_servers):
        self.env = env
        self.cashier = simpy.Resource(env, capacity=num_cashiers)
        self.usher = simpy.Resource(env, capacity=num_ushers)
        self.server = simpy.Resource(env, capacity=num_servers)

    def purchase_ticket(self, customer):
        yield self.env.timeout(random.randint(1, 3))

    def check_ticket(self, customer):
        yield self.env.timeout(3 / 60)

    def sell_food(self, customer):
        yield self.env.timeout(random.randint(1, 5))


def go_to_movies(env: simpy.Environment, customer, theater: Theater):
    arrival_time = env.now

    with theater.cashier.request() as request:
        yield request
        yield env.process(theater.purchase_ticket(customer))

    with theater.usher.request() as request:
        yield request
        yield env.process(theater.check_ticket(customer))

    if random.choice([True, False]):
        with theater.server.request() as request:
            yield request
            yield env.process(theater.sell_food(customer))

    waiting_time.append(env.now - arrival_time)


def run_theater(env: simpy.Environment, num_cashiers, num_servers, num_ushers):
    theater = Theater(env, num_cashiers, num_ushers, num_servers)

    for customer in range(3):

        env.process(go_to_movies(env, customer, theater))

    while True:
        yield env.timeout(0.2)  # every 0.2 min generate a new customer
        customer += 1
        env.process(go_to_movies(env, customer, theater))


def calculate_waiting_time(wait_time):
    avg_time = statistics.mean(wait_time)
    minutes, frac_minutes = divmod(avg_time, 1)
    seconds = frac_minutes * 60
    return round(minutes), round(seconds)


def get_input():
    num_cashiers = input('cashier number: ')
    num_ushers = input('ushers number: ')
    num_servers = input('servers number: ')
    params = [num_cashiers, num_ushers, num_servers]
    if all(str(i).isdigit() for i in params):
        params = [int(x) for x in params]
    else:
        print('input wrong, simulation start with default value')
        params = [1, 1, 1]
    return params


def main():
    # setup
    random.seed(42)
    num_cashiers, num_ushers, num_servers = get_input()

    # run
    env = simpy.Environment()
    env.process(run_theater(env, num_cashiers, num_ushers, num_servers))
    env.run(until=60)

    # output
    mins, secs = calculate_waiting_time(wait_time=waiting_time)
    print("Running simulation...",
          f"\nThe average wait time is {mins} minutes and {secs} seconds.")


if __name__ == '__main__':
    main()
