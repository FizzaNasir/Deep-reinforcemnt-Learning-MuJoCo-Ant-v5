# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO

# %%
# =================== DISPLAY ORIGINAL ANT ENV ==================
# from gymnasium.wrappers import RecordVideo
# import gymnasium as gym
# from stable_baselines3 import PPO
# import os

xml_path = r"C:\Users\HP\Documents\MPhill\Deep Reinforcement Learning\Antv5\ant.xml"
print("Exists:", os.path.exists(xml_path))

env = gym.make("Ant-v5", 
               render_mode="rgb_array",
            #    xml_file=xml_path
            )

env = RecordVideo(
    env,
    video_folder="./test_env",
    episode_trigger=lambda x: True
)

obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()

# %%


# from gymnasium.wrappers import RecordVideo
# import gymnasium as gym
# from stable_baselines3 import PPO
# import os

# %%
# ============ TRAIN THE MODEL ON SIMPLE 4 LEGGED ANT ==========
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
env = gym.make("Ant-v5", render_mode="rgb_array")
model_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    # tensorboard_log="./ppo_logs/"
)

model_ppo.learn(total_timesteps=2000000)
model_ppo.save("ppo_ant")

env.close()

#%%
# ============ TRAIN THE MODEL ON SIMPLE 4 LEGGED ANT (50000 steps)==========
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
env = gym.make("Ant-v5", render_mode="rgb_array")
model_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    # tensorboard_log="./ppo_logs/"
)

model_ppo.learn(total_timesteps=50000)
model_ppo.save("ppo_ant_50000")

env.close()


#%%
# ============ TRAIN THE MODEL ON SIMPLE 4 LEGGED ANT (100000 steps)==========
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
env = gym.make("Ant-v5", render_mode="rgb_array")
model_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    # tensorboard_log="./ppo_logs/"
)

model_ppo.learn(total_timesteps=100000)
model_ppo.save("ppo_ant_100000")

env.close()


#%%
# ============ TRAIN THE MODEL ON SIMPLE 4 LEGGED ANT (1M steps)==========
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
env = gym.make("Ant-v5", render_mode="rgb_array")
model_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    # tensorboard_log="./ppo_logs/"
)

model_ppo.learn(total_timesteps=1000000)
model_ppo.save("ppo_ant_1M")

env.close()


#%%
# ============ TRAIN THE MODEL ON SIMPLE 4 LEGGED ANT (10k steps)==========
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
env = gym.make("Ant-v5", render_mode="rgb_array")
model_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    # tensorboard_log="./ppo_logs/"
)

model_ppo.learn(total_timesteps=10000)
model_ppo.save("ppo_ant_10k")

env.close()


#=================== EVALUATE MODEL ON 4 LEGGED ================4
# %%
# from gymnasium.wrappers import RecordVideo
# import gymnasium as gym
# import numpy as np

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("ppo_ant")

env = RecordVideo(
    env,
    video_folder="./ppo_videos",
    episode_trigger=lambda x: True
)
rewards = []
obs, info = env.reset()

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
import gymnasium
# print(gymnasium.__file__)

#=================== EVALUATE MODEL ON 4 LEGGED ant (50000 steps)================4
# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("ppo_ant_50000")

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
#=================== EVALUATE MODEL ON 4 LEGGED ant (10k steps)================4
# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("ppo_ant_10k")

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

#=================== EVALUATE MODEL ON 4 LEGGED ant (1M steps)================4


# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("ppo_ant_1M")

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

#=================== EVALUATE MODEL ON 4 LEGGED ant (100000 steps)================4

# %%
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make("Ant-v5", render_mode="rgb_array")
model = PPO.load("ppo_ant_1M")

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
