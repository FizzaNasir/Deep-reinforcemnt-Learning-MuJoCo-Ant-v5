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

model_4 = PPO.load("ppo_ant")

model_3 = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=64,
    clip_range=0.1,
    ent_coef=0.01,
    verbose=1
)

# copy ONLY policy weights (not spaces)
model_3.policy.load_state_dict(model_4.policy.state_dict(), strict=False)
model_3.learn(total_timesteps=3000000)

# Save trained 3-leg model
model_3.save("ppo_ant_3leg")


#=========================== MODEL EVALUATION =====================================
#%%
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

model = PPO.load("ppo_ant_3leg")

env = gym.make(
    "Ant-v5",
    render_mode="rgb_array",
    xml_file=xml_path  # your modified 3-leg version
)

# 🔥 wrap observation so PPO accepts it
env = ObsAdapter(env, TARGET_DIM)

env = RecordVideo(
    env,
    video_folder="./3_legged_trained",
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
