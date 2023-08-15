from ED_env_v0 import ER_easy
from ED_env_v1 import EmergencyDepartmentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

env = EmergencyDepartmentEnv()
# It will check your custom environment and output additional warnings if needed
# check_env(env)


# env = ER_easy()

tmp_path = "./tmp/sb3_log_PPO/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/tmp/sb3_log/")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="/tmp/sb3_log_PPO/", learning_rate=1e-3)
model.set_logger(new_logger)
model.learn(total_timesteps=50000, progress_bar=True)

model.save("ED_PPO_trial")
# del model
# model = PPO.load("ED_PPO")



