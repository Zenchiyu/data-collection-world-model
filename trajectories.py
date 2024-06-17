import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from src.envs.env import make_crafter_env
from src.utils import RandomHeuristicCrafter
from tqdm import tqdm


def play(env, max_num_steps, num_envs):
    heuristic = RandomHeuristicCrafter(num_envs)

    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(num_envs)])
    # initial_map = env.unwrapped.call("get_map", episode=0)
    
    trajectories = [[torch.zeros((0,2), dtype=torch.int)] for _ in range(num_envs)]
    for num_steps in tqdm(range(0, max_num_steps, num_envs)):
        act = heuristic.act(obs).to(obs.device)
        # print(obs.shape, num_steps, act)
        next_obs, rew, end, trunc, info = env.step(act)

        # Due to https://gymnasium.farama.org/api/vector/
        # There's an implicit reset -> new map every time agent dies
        for n in range(num_envs):
            if end[n] == 0:
                trajectories[n][-1] = torch.cat((trajectories[n][-1], torch.tensor(info["player_pos"][n], dtype=torch.int)[None, :]))
            else:
                # TODO: check when it resets. Can I get the correct last player pos too?
                trajectories[n].append(torch.zeros((0,2), dtype=torch.int))
        obs = next_obs
    
    min_num_episodes = (min([len(episodes) for episodes in trajectories])//2)*2
    maps = [env.unwrapped.call("get_map", episode=ep) for ep in range(min_num_episodes)]
    # Sanity check when getting the map
    # assert torch.all(torch.cat([torch.isclose(torch.tensor(x), torch.tensor(y)) for x,y in zip(maps[0], initial_map)]))
    return trajectories, maps

def plot_trajectories_maps(trajectories, maps, num_envs, filename="trajectories.pdf"):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set3.colors)
    # https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
    min_num_episodes = len(maps)
    fig, axs = plt.subplots(2, min_num_episodes//2, figsize=(5*min_num_episodes/2, 10), sharey=True)
    major_ticks = torch.arange(0, 64+1, 8)
    minor_ticks = torch.arange(0, 64+1, 1)

    for ep in range(min_num_episodes):  # Nb of worlds we'll see
        ax = axs[ep // (min_num_episodes//2), ep % (min_num_episodes//2)]
        ax.imshow(maps[ep][0], extent=[0, 64, 0, 64], origin="lower")
        for n in range(num_envs):
            trajs = trajectories[n][ep] + torch.tensor([0.5, -0.5])[None, :]   # Just to make it centered
            ax.plot(*trajs.T, '.--', linewidth=0.5, markersize=1, zorder=5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=300)

def plot_exploration_area(trajectory, map, nav_only=True, filename="overlap_one_episode"):
    num_visits_matrix = torch.zeros(64, 64, dtype=torch.int)
    for i, position in enumerate(trajectory):
        if nav_only and torch.all(position == trajectory[i-1]): # If do not move, do not count (it could also be due to obstacles!)
            continue
        pos_x = position[0]     # column
        pos_y = position[1]-1   # row. XXX: -1 due to an offset https://github.com/danijar/crafter/blob/e04542a2159f1aad3d4c5ad52e8185717380ee3a/crafter/engine.py#L161
        pos_rows = torch.arange(64)[:, None]
        pos_cols = torch.arange(64)[None, :]
        view_mask = ((pos_y - 3 <= pos_rows) & (pos_rows <= pos_y + 3)) 
        view_mask = view_mask & ((pos_x -4 <= pos_cols) & (pos_cols <= pos_x + 4))
        num_visits_matrix[view_mask] += 1
    mymap = torch.tensor(map)/255
    alpha = num_visits_matrix.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)[..., None].to(torch.float)
    x = alpha[alpha != 0]/alpha[alpha != 0].max()
    alpha[alpha != 0] = 0.4*(1-x) + x   # 0.25
    alpha[alpha == 0] = 0.1
    mymap = torch.cat((mymap, torch.clip(alpha, max=1)), dim=-1)

    plt.figure()
    ax = plt.gca()
    major_ticks = torch.arange(0, 64+1, 8)
    minor_ticks = torch.arange(0, 64+1, 1)
    ax.imshow(mymap, extent=[0, 64, 0, 64], origin="lower")# alpha=0.4

    traj = trajectory + torch.tensor([0.5, -0.5])[None, :]   # Just to make it centered
    ax.plot(*traj.T, 'r.--', linewidth=0.5, markersize=1, zorder=5)
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    ax.invert_yaxis()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=600)
    return num_visits_matrix

def plot_exploration_areas(trajectories, maps, num_envs, nav_only=True, heatmap=False, filename="overlap.pdf"):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Set3.colors)
    # https://stackoverflow.com/questions/24943991/change-grid-interval-and-specify-tick-labels
    min_num_episodes = len(maps)
    fig, axs = plt.subplots(2, min_num_episodes//2, figsize=(5*min_num_episodes/2, 10), sharey=True)
    major_ticks = torch.arange(0, 64+1, 8)
    minor_ticks = torch.arange(0, 64+1, 1)
    num_visits_matrices = []
    for ep in range(min_num_episodes):  # Nb of worlds we'll see
        ax = axs[ep // (min_num_episodes//2), ep % (min_num_episodes//2)]
        # ax.imshow(maps[ep][0], extent=[0, 64, 0, 64], origin="lower")
        num_visits_matrix = torch.zeros(64, 64, dtype=torch.int)

        for n in range(num_envs):
            trajectory = trajectories[n][ep]
            for i, position in enumerate(trajectory):
                if nav_only and torch.all(position == trajectory[i-1]): # If do not move, do not count (it could also be due to obstacles!)
                    continue
                pos_x = position[0]     # column
                pos_y = position[1]-1   # row. XXX: -1 due to an offset https://github.com/danijar/crafter/blob/e04542a2159f1aad3d4c5ad52e8185717380ee3a/crafter/engine.py#L161
                pos_rows = torch.arange(64)[:, None]
                pos_cols = torch.arange(64)[None, :]
                view_mask = ((pos_y - 3 <= pos_rows) & (pos_rows <= pos_y + 3)) 
                view_mask = view_mask & ((pos_x -4 <= pos_cols) & (pos_cols <= pos_x + 4))
                # XXX: cumulative, not doing an average
                num_visits_matrix[view_mask] += 1

            traj = trajectory + torch.tensor([0.5, -0.5])[None, :]   # Just to make it centered
            ax.plot(*traj.T, '.--', linewidth=0.5, markersize=1, zorder=5)
        
        num_visits_matrices.append(num_visits_matrix)
        mymap = torch.tensor(maps[ep][0])/255
        alpha = num_visits_matrix.repeat_interleave(16, dim=0).repeat_interleave(16, dim=1)[..., None].to(torch.float)
        x = alpha[alpha != 0]/alpha[alpha != 0].max()
        alpha[alpha != 0] = 0.7*(1-x) + x if heatmap else 1 # Linear interp.
        alpha[alpha == 0] = 0.4
        mymap = torch.cat((mymap, torch.clip(alpha, max=1)), dim=-1)
        ax.imshow(mymap, extent=[0, 64, 0, 64], origin="lower")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        ax.set_yticks(minor_ticks, minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, dpi=300)
    return num_visits_matrices

if __name__ == "__main__":
    MAX_NUM_STEPS = 10_000 # 5_000, 10_000
    NUM_ENVS = 10   # 5, 10
    assert MAX_NUM_STEPS % NUM_ENVS == 0
    DEVICE, WORLD_SIZE = "cuda" if torch.cuda.is_available() else "cpu", 64
    
    env = make_crafter_env(num_envs=NUM_ENVS, device=DEVICE, size=WORLD_SIZE)
    trajectories, maps = play(env, num_envs=NUM_ENVS, max_num_steps=MAX_NUM_STEPS)
    
    # plot_trajectories_maps(trajectories, maps, num_envs=NUM_ENVS, filename=f"trajectories_{NUM_ENVS}_copies_{MAX_NUM_STEPS}_max_steps.pdf")
    # plot_exploration_areas(trajectories, maps, NUM_ENVS)
    
    # XXX: for a single map
    num_visits_matrices = []
    for n in range(NUM_ENVS):
        num_visits_matrix = plot_exploration_area(trajectories[n][0], maps[0][0], nav_only=True)
        num_visits_matrices.append(num_visits_matrix)
    
    fig, axs = plt.subplots(2, NUM_ENVS//2, figsize=(30, 15))
    frequencies = {"uniques": [], "counts": []}
    max_num_visits = 1
    for n in range(NUM_ENVS):
        num_visits_matrix = num_visits_matrices[n]
        uniques, counts = torch.unique(num_visits_matrix, return_counts=True)
        max_num_visits = max(max_num_visits, uniques.max())
        frequencies["uniques"].append(uniques)
        frequencies["counts"].append(counts)
    # This does not tell if we're going in circle within a large or small area

    total_counts = torch.zeros((max_num_visits+1,))
    for n in range(NUM_ENVS):
        total_counts[frequencies["uniques"][n]] += frequencies["counts"][n]
    uniques, avg_counts = torch.arange(1, max_num_visits+1), total_counts[1:]/10    # ignore 0
    probs = avg_counts/avg_counts.sum()
    num_observed_cells = torch.tensor([m.count_nonzero().item() for m in num_visits_matrices], dtype=torch.float)
    
    plt.figure()
    plt.bar(uniques, probs)
    plt.xlabel("# of visits of a cell")
    plt.ylabel("Frequency of # of visits, averaged over $10$ trajectories")
    # Frequency of cells with a certain number of visits
    plt.title("Frequency of # of visits, averaged over $10$ trajectories\n"+\
              f"The agents observes {num_observed_cells.mean():.2f}$\pm${num_observed_cells.std():.2f} cells in its FOV")
    plt.savefig("freq_visits.pdf")
    # Conditional probabilities, given we're using navigation actions
    # Maximum proba to leave an area completely: probs[uniques >= 7].sum(), 7 is from the shortest way (vertical direction) to leave an area completely
    # Maximum proba to leave an area completely and come back completely: probs[uniques >= 14].sum()