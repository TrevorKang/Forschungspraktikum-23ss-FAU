from stable_baselines3 import DQN
from GymED import ER_Gym

env = ER_Gym(num_doctors=20, num_nurses=20, sim_time=12*60)
model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, batch_size=100, exploration_fraction=0.999, buffer_size=10000)
model.learn(total_timesteps=10000)
