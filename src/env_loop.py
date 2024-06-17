import torch
import random

from utils import coroutine, RandomHeuristicCrafter


@coroutine
def make_env_loop(env):
    num_steps = yield
    heuristic = RandomHeuristicCrafter(env.num_envs)

    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])

    while True:
        all_ = []
        n = 0
        while n < num_steps:
            act = heuristic.act(obs).to(obs.device)
            next_obs, rew, end, trunc, info = env.step(act)
            
            all_.append([obs, act, rew, end, trunc])
            obs = next_obs
            n += 1
        
        all_obs, act, rew, end, trunc = (torch.stack(x, dim=1) for x in zip(*all_))
        num_steps = yield all_obs, act, rew, end, trunc
