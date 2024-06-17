import hydra
import shutil
import torch
import wandb

from collect import make_collector, NumToCollect
from data import EpisodeDataset
from datetime import datetime
from envs import make_atari_env, make_crafter_env
from functools import partial
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from utils import set_seed, try_until_no_except


class Collector:
    def __init__(self, cfg: DictConfig) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self.cfg = cfg
        if cfg.common.seed is None:
            cfg.common.seed = int(datetime.now().timestamp()) % 10**5
        set_seed(cfg.common.seed)
        try_until_no_except(
            partial(
                wandb.init,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
                resume=True,
                **cfg.wandb,
            )
        )

        self.device = torch.device(cfg.common.device)
        if "cuda" in cfg.common.device:
            torch.cuda.set_device(self.device)  # (quick) fix compilation error on multi-gpu nodes

        # Directories
        self.dataset_dir = Path("dataset")

        config_dir = Path("config")
        config_path = config_dir / "collector.yaml"
        config_dir.mkdir(exist_ok=False, parents=False)
        shutil.copy(".hydra/config.yaml", config_path)
        wandb.save(str(config_path))
        shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
        
        num_workers = cfg.num_workers_data_loaders
        cache_in_ram = cfg.cache_in_ram

        self.dataset = EpisodeDataset(
            directory=self.dataset_dir / "recent",
            name="dataset",
            save_on_disk=True,
            cache_in_ram=cache_in_ram,
            use_manager_list=cache_in_ram and (num_workers > 0),
        )
        
        env_fn = partial(
            make_crafter_env,
            num_envs=cfg.collection.num_envs,
            device=self.device,
            **cfg.env.train,
        )
        # num_actions = int(env_fn().single_action_space.n)

        self.collector = make_collector(env_fn, dataset=self.dataset)   # XXX: creates once num_envs different worlds


    def run(self) -> None:
        to_log = []
        
        c = self.cfg.collection
        steps = c.num_steps_total
        
        print("\nCollecting\n")
        
        to_log.extend(self.collector.send(NumToCollect(steps=steps)))

        print("\nSummary of collect:")
        print(f"Num steps: {self.dataset.num_steps} / {c.num_steps_total}")
        print(f"Reward counts: {dict(self.dataset.counter_rew)}")

        remaining_steps = c.num_steps_total - self.dataset.num_steps
        assert remaining_steps == 0

        # TODO wandb.log the "to_log"
        self.dataset.save_info()

        self.finish()
        return 0

    def finish(self) -> None:
        wandb.finish()
