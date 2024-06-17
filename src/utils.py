import cv2
import logging
from functools import wraps
from pathlib import Path
import random

import numpy as np
import torch


log = logging.getLogger(__name__)

def skip_if_run_is_over(func):
    # Avoids running func (stops early) if there's a .run_over file
    # It avoids going through the long wandb.init before stopping
    def inner(*args, **kwargs):
        path_run_is_over = Path(".run_over")
        if not path_run_is_over.is_file():
            func(*args, **kwargs)
            path_run_is_over.touch()
        else:
            log.info(f"Run is marked as finished. To unmark, remove '{str(path_run_is_over)}'.")

    return inner

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()


def try_until_no_except(fn):
    while True:
        try:
            fn()
        except:
            continue
        else:
            break

class RandomHeuristic:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))

class RandomHeuristicCrafter:
    """Crafter specific random policy"""
    def __init__(self, num_envs):
        self.n = num_envs
        self.sticky = np.zeros(num_envs)
        self.previous_action = np.zeros(num_envs)
    
    def act(self, obs):
        assert obs.ndim == 4 # (N, H, W, C)
        actions = torch.empty(self.n, dtype=torch.long)
        for i in range(self.n):
            actions[i] = self._act(i)
        return actions

    def _act(self, i):
        if self.sticky[i] > 0:
            self.sticky[i] -= 1
            return self.previous_action[i]
        
        while True:
            action = random.randint(0, 16)
            if action == 6 and random.random() > 0.1: # downsample sleep
                continue
            break

        if 1 <= action <= 4: # repeat navigation actions
            self.sticky[i] = random.randint(0, 6)
        elif action == 5: # repeat do
            self.sticky[i] = random.randint(1, 3)
        else:
            self.sticky[i] = 0

        self.previous_action[i] = action
        return action


def coroutine(func):
    @wraps(func)
    def primer(*args,**kwargs):
        gen = func(*args,**kwargs)
        next(gen)
        return gen
    return primer