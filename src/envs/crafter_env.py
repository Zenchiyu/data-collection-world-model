import crafter
import gymnasium
import numpy as np


class CrafterEnv(gymnasium.Env):
    def __init__(self, size: int = 64):
        self.env = crafter.Env(size=(size, size))
        self.observation_space = gymnasium.spaces.Box(0, 255, (size, size, 3), np.uint8)
        self.action_space = gymnasium.spaces.Discrete(17)

    def reset(self, *args, **kwargs):
        obs = self.env.reset()
        return obs, {}

    def step(self, act):
        obs, rew, end, info = self.env.step(act)
        trunc = False
        return obs, rew, end, trunc, info
    
    def get_map(self, episode=0):
        # I (Stephane) added this
        env = crafter.Env(area=(64, 64), view=(64, 64), size=1024, seed=self.env._seed)
        env._episode = episode
        return env.reset()
