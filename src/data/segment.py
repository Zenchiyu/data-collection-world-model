"""
Modified from https://github.com/eloialonso/diffusion/src/data/segment.py
"""
import torch

from dataclasses import dataclass


@dataclass
class SegmentId:
    episode_id: int
    start: int  # if negative => left padding
    stop: int   # if more than len(episode) => right padding 


@dataclass
class Segment:
    obs: torch.FloatTensor
    act: torch.LongTensor
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor  # due to horizon
    mask_padding: torch.BoolTensor  # True for non-padded, False for padded time steps.
    id: SegmentId
    
    @property
    def effective_size(self):
        return self.mask_padding.sum().item()