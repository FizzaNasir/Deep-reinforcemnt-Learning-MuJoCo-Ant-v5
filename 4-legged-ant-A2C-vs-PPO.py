#%%
# ============================================================
# PPO vs A2C on MuJoCo Ant-v5
# ============================================================
# Goal:
# Compare PPO and A2C under identical conditions
# using:
#   - Same environment
#   - Same timesteps
#   - Same network architecture
#   - Same learning rate
#
# Outputs:
#   1. Reward curves
#   2. Average rewards
#   3. Stability comparison
#
# ============================================================

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# ============================================================
# CONFIGURATION
# ============================================================

ENV_NAME = "Ant-v5"

TOTAL_TIMESTEPS = 1_000_000

LEARNING_RATE = 3e-4
GAMMA = 0.99

POLICY_KWARGS = dict(
    net_arch=[64, 64]
)

# ============================================================
# CUSTOM CALLBACK TO STORE REWARDS
# ============================================================

class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]

        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(
                    info["episode"]["r"]
                )

        return True

# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def make_env():
    env = gym.make(ENV_NAME)
    env = Monitor(env)
    return env

ppo_env = DummyVecEnv([make_env])
a2c_env = DummyVecEnv([make_env])

# ============================================================
# PPO MODEL
# ============================================================

ppo_model = PPO(
    policy="MlpPolicy",
    env=ppo_env,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    policy_kwargs=POLICY_KWARGS,
    verbose=1
)

# ============================================================
# A2C MODEL
# ============================================================

a2c_model = A2C(
    policy="MlpPolicy",
    env=a2c_env,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    policy_kwargs=POLICY_KWARGS,
    verbose=1
)

# ============================================================
# TRAIN PPO
# ============================================================

print("\nTraining PPO...\n")

ppo_callback = RewardCallback()

ppo_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=ppo_callback
)

# ============================================================
# TRAIN A2C
# ============================================================

print("\nTraining A2C...\n")

a2c_callback = RewardCallback()

a2c_model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=a2c_callback
)

# ============================================================
# PLOT REWARD CURVES
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    ppo_callback.episode_rewards,
    label="PPO"
)

plt.plot(
    a2c_callback.episode_rewards,
    label="A2C"
)

plt.xlabel("Episodes")
plt.ylabel("Episode Reward")
plt.title("PPO vs A2C on Ant-v5")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# STABILITY ANALYSIS
# ============================================================

ppo_mean = np.mean(ppo_callback.episode_rewards[-100:])
ppo_std = np.std(ppo_callback.episode_rewards[-100:])

a2c_mean = np.mean(a2c_callback.episode_rewards[-100:])
a2c_std = np.std(a2c_callback.episode_rewards[-100:])

print("\n================ RESULTS ================\n")

print("PPO")
print(f"Average Reward : {ppo_mean:.2f}")
print(f"Reward Std Dev : {ppo_std:.2f}")

print()

print("A2C")
print(f"Average Reward : {a2c_mean:.2f}")
print(f"Reward Std Dev : {a2c_std:.2f}")

# ============================================================
# CONVERGENCE COMPARISON
# ============================================================

def moving_average(data, window=20):
    return np.convolve(
        data,
        np.ones(window)/window,
        mode='valid'
    )

ppo_smooth = moving_average(
    ppo_callback.episode_rewards
)

a2c_smooth = moving_average(
    a2c_callback.episode_rewards
)

plt.figure(figsize=(12, 6))

plt.plot(
    ppo_smooth,
    label="PPO Smoothed"
)

plt.plot(
    a2c_smooth,
    label="A2C Smoothed"
)

plt.xlabel("Episodes")
plt.ylabel("Smoothed Reward")

plt.title("Convergence Comparison")

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# SAVE MODELS
# ============================================================

ppo_model.save("ppo_ant_model")
a2c_model.save("a2c_ant_model")

print("\nModels saved successfully.")
# %%
