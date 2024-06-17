"""
Modified from https://github.com/eloialonso/diffusion/src/data/batch.py
"""
from __future__ import annotations

import torch

from dataclasses import dataclass
from .segment import SegmentId


@dataclass
class Batch:
    obs: torch.FloatTensor   # x_t
    act: torch.LongTensor   # a_t
    rew: torch.FloatTensor  # r_t after performing action a_t from x_t
    end: torch.LongTensor   # d_t after performing action a_t from x_t
    trunc: torch.LongTensor # due to horizon
    mask_padding: torch.BoolTensor
    segment_ids: list[SegmentId]

    def pin_memory(self) -> Batch:
        # Pin_memory all attributes except segment_ids (not a tensor)
        return Batch(**{k: v if k == 'segment_ids' else v.pin_memory() for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        # Move to device all attributes except segment_ids (not a tensor)
        return Batch(**{k: v if k == 'segment_ids' else v.to(device) for k, v in self.__dict__.items()})