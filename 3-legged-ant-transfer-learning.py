#%%
# ======================== DISPLAY INITIAL 3 LEGGED ENV ================

from gymnasium.wrappers import RecordVideo
import gymnasium as gym
from stable_baselines3 import PPO
import os

xml_path = r"C:\Users\HP\Documents\MPhill\Deep Reinforcement Learning\Antv5\ant.xml"
print("Exists:", os.path.exists(xml_path))

env = gym.make("Ant-v5", 
               render_mode="rgb_array",
               xml_file=xml_path
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


#=================== EVALUATE MODEL TRAINED WITH 4 LEGGED ON 3 Legged ================4

# %%
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import gymnasium as gym
import numpy as np
import os

xml_path = r"C:\Users\HP\Documents\MPhill\Deep Reinforcement Learning\Antv5\ant.xml"
print("Exists:", os.path.exists(xml_path))

def adapt_action(action, env):
    return action[:env.action_space.shape[0]]

class ObsAdapter(gym.ObservationWrapper):
    def __init__(self, env, target_dim):
        super().__init__(env)
        self.target_dim = target_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(target_dim,),
            dtype=np.float32
        )

    def observation(self, obs):
        obs = np.array(obs, dtype=np.float32)

        if obs.shape[0] < self.target_dim:
            # pad with zeros
            padded = np.zeros(self.target_dim, dtype=np.float32)
            padded[:obs.shape[0]] = obs
            return padded

        # truncate if needed
        return obs[:self.target_dim]
    
TARGET_DIM = 105  # Ant-v5 original observation size

model = PPO.load("ppo_ant")

env = gym.make(
    "Ant-v5",
    render_mode="rgb_array",
    xml_file=xml_path  # your modified 3-leg version
)

# 🔥 wrap observation so PPO accepts it
env = ObsAdapter(env, TARGET_DIM)

env = RecordVideo(
    env,
    video_folder="./ant_eval",
    episode_trigger=lambda x: True
)

obs, info = env.reset()

rewards = []

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    action = adapt_action(action, env)
    obs, reward, terminated, truncated, info = env.step(action)

    rewards.append(reward)

    if terminated or truncated:
        break;

env.close()

print("Mean reward:", np.mean(rewards))
print("Std reward:", np.std(rewards))
# %%
