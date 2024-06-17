"""
Modified from https://github.com/eloialonso/diffusion/src/data/episode.py
"""
from __future__ import annotations

import torch

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class Episode:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor # due to horizon
    # There's no mask_padding

    def __len__(self) -> int:
        return self.obs.shape[0]
    
    def __add__(self, other: Episode) -> Episode:
        # concat each tensor by just using "+" operator between Episodes
        assert self.dead.sum() == 0
        return Episode(
            **{
                k: torch.cat((v, other.__dict__[k]), dim=0)
                for k, v in self.__dict__.items()
            }
        )
    
    def to(self, device: torch.device) -> Episode:
        return Episode(**{k: v.to(device) for k, v in self.__dict__.items()})
    
    @property
    def dead(self) -> torch.ByteTensor:
        return (self.end + self.trunc).clip(max=1)
    
    def compute_metrics(self) -> dict[str, Any]:
        return {"length": len(self), "return": self.rew.sum().item()}  # no discounting
    
    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device]=None) -> Episode:
        return cls(
            **{
                k: v.div(255).mul(2).sub(1) if k == "obs" else v  # [-1, 1] for obs
                for k, v in torch.load(Path(path), map_location=map_location).items()
            }
        )

    def save(self, path: Path) -> None:
        d = {
            k: v.add(1).div(2).mul(255).byte() if k == "obs" else v  # {0, 1, ..., 255} for saved obs
            for k, v in self.__dict__.items()
        }
        torch.save(d, Path(path))