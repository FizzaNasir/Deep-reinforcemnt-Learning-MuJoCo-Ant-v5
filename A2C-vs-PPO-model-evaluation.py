# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("a2c_ant_model")

env = RecordVideo(
    env,
    video_folder="./ppo_videos",
    episode_trigger=lambda x: True
)
rewards = []
obs, info = env.reset()
# pip install --force-reinstall gymnasium
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        break   # ❗ stop instead of resetting

print("Mean reward:", np.mean(rewards))
print("Std reward:", np.std(rewards))
env.close()


#%%
# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("ppo_ant_model")

env = RecordVideo(
    env,
    video_folder="./ppo_videos",
    episode_trigger=lambda x: True
)
rewards = []
obs, info = env.reset()
# pip install --force-reinstall gymnasium
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        break   # ❗ stop instead of resetting

print("Mean reward:", np.mean(rewards))
print("Std reward:", np.std(rewards))
env.close()
# %%
