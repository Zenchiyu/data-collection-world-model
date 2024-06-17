"""
Modified from https://github.com/eloialonso/diffusion/src/data/dataset.py
"""
import logging
import multiprocessing as mp
import numpy as np
import shutil
import torch

from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment


log = logging.getLogger(__name__)

class EpisodeDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        name: str,
        save_on_disk: bool,
        cache_in_ram: bool,
        use_manager_list: bool
    ) -> None:

        super().__init__()
        self.directory = Path(directory).expanduser()
        self.name = name
        self.save_on_disk = save_on_disk
        self.cache_in_ram = cache_in_ram
        self.use_manager_list = use_manager_list

        # Number of episodes in the dataset
        self.num_episodes = None
        # Sum of number of (time) steps from all episodes
        self.num_steps = None

        self.start_idx = None   # (global) start indices of each Episode
        self.lengths = None     # lengths of each Episode
        # Used to check rew sign and end class proportions
        self.counter_rew = None
        self.counter_end = None
        
        # Cache in memory episodes from the dataset
        self.cache = mp.Manager().dict() if self.use_manager_list else {}

        if not self.directory.is_dir():
            self._init_empty()
        else:
            self._load_info()
            log.info(f"({name}) {self.num_episodes} episodes, {self.num_steps} steps.")

    def __len__(self) -> int:
        # Sum of time steps from all episodes
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        # Segment of an episode
        return self._load_segment(segment_id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.directory.absolute()}, {self.name})"
    
    def __str__(self) -> str:
        return f'{self.__class__.__name__}\nName: {self.name}\nDirectory: {self.directory.absolute()}\nNum steps: {self.info["num_steps"]}\nNum episode: {self.info["num_episodes"]}'
    
    def _init_empty(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=False)
        self.num_episodes, self.num_steps = 0, 0
        self.start_idx = np.array([], dtype=np.int64)
        self.lengths = np.array([], dtype=np.int64)
        self.counter_rew = Counter()  # only tracks the signs
        self.counter_end = Counter()
        self.save_info()
    
    def _load_info(self) -> None:
        info = torch.load(self.info_path)
        self.num_episodes = info["num_episodes"]
        self.num_steps = info["num_steps"]
        self.start_idx = info["start_idx"]
        self.lengths = info["lengths"]
        self.counter_rew = info["counter_rew"]
        self.counter_end = info["counter_end"]

    def save_info(self) -> None:
        torch.save(self.info, self.info_path)

    @property
    def info(self) -> dict[str, int | np.ndarray]:
        # Since dataset may change in general (experience collection),
        # The following may change
        return {
            "num_episodes": self.num_episodes,
            "num_steps": self.num_steps,
            "start_idx": self.start_idx,
            "lengths": self.lengths,
            "counter_rew": self.counter_rew,
            "counter_end": self.counter_end
        }
    
    @property
    def info_path(self) -> Path:
        return self.directory / "info.pt"
        
    # Tracking reward signs (negative, zero, positive).
    @property
    def counts_rew(self) -> list[int]:
        return [self.counter_rew[r] for r in [-1, 0, 1]]

    @property
    def counts_end(self) -> list[int]:
        return [self.counter_end[e] for e in [0, 1]]

    # def clear(self) -> None:
    #     # Should not be used on static dataset
    #     shutil.rmtree(self.directory)
    #     self._init_empty()

    def _get_episode_path(self, episode_id: int) -> Path:
        # Path to episode .pt file (obs, rew etc. of an episode)
        n = 3  # number of hierarchies
        powers = np.arange(n)[::-1]
        # episode_id = sum of powers of 10 (base 10)
        subfolders = (np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers).astype(int)
        subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
        return self.directory / subfolders / f"{episode_id}.pt"
    
    def _load_segment(self, segment_id: SegmentId, should_pad: bool=True) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        return make_segment(episode, segment_id, should_pad)
    
    def load_episode(self, episode_id: int) -> Episode:
        if self.cache_in_ram and (episode_id in self.cache):
            episode = self.cache[episode_id]
        else:
            episode = Episode.load(self._get_episode_path(episode_id), map_location="cpu")
            if self.cache_in_ram:
                self.cache[episode_id] = episode
        return episode

    def add_episode(self, episode: Episode, *, episode_id: Optional[int]=None) -> int:
        # https://stackoverflow.com/questions/2965271/how-can-we-force-naming-of-parameters-when-calling-a-function/14298976#14298976
        if episode_id is None:
            episode_id = self.num_episodes
            self.start_idx = np.concatenate((self.start_idx, np.array([self.num_steps])))  # new episode is adjacent to all prev.
            self.lengths = np.concatenate((self.lengths, np.array([len(episode)])))
            self.num_steps += len(episode)
            self.num_episodes += 1
        else:
            assert episode_id < self.num_episodes
            # Complete an "unfinished episode"
            old_episode = self.load_episode(episode_id)
            incr_num_steps = len(episode) - len(old_episode)
            self.lengths[episode_id] = len(episode)
            self.start_idx[episode_id+1:] += incr_num_steps  # shift all the next start_idx's
            self.num_steps += incr_num_steps
            self.counter_rew.subtract(old_episode.rew.sign().tolist())
            self.counter_end.subtract(old_episode.end.tolist())

        self.counter_rew.update(episode.rew.sign().tolist())
        self.counter_end.update(episode.end.tolist())

        # Save new episode on disk
        if self.save_on_disk:
            episode_path = self._get_episode_path(episode_id)
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            episode.save(episode_path.with_suffix(".tmp"))
            episode_path.with_suffix(".tmp").rename(episode_path)

        return episode_id