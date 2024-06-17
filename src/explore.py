import argparse
import torch

from data import EpisodeDataset
from visualizer import Visualizer
from visualizer.keymap import get_keymap_and_action_names
from visualizer.dataset_env import DatasetEnv
from pathlib import Path
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--no-header", action="store_true")    
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    cfg = OmegaConf.load("config/collector.yaml")
    
    h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]
    
    _, env_action_names = get_keymap_and_action_names(cfg.env.keymap)

    datasets = []
    for p in Path("dataset").iterdir():
        if p.is_dir():
            datasets.append(EpisodeDataset(p, p.stem, False, False, False))

    env = DatasetEnv(datasets, env_action_names)

    visualizer = Visualizer(env, size=size, fps=args.fps, verbose=not args.no_header)
    visualizer.run()


if __name__ == "__main__":
    main()
