import gymnasium as gym
import numpy as np

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
