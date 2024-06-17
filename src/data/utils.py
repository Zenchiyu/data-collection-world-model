"""
Modified from https://github.com/eloialonso/diffusion/src/data/utils.py
"""
import math
import torch
import torch.nn.functional as F

from typing import Generator

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId


def collate_segments_to_batch(segments: list[Segment]) -> Batch:
    # Make a Batch out of Segments (Not Episodes!)
    stack = (torch.stack([getattr(s, x) for s in segments]) for x in ('obs', 'act', 'rew', 'end', 'trunc', 'mask_padding'))
    return Batch(*stack, [s.id for s in segments])

def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool=True) -> Segment:
    # Make a Segment out of an Episode
    assert segment_id.start < len(episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    padding_length_left = max(0, -segment_id.start)  # segment_id.start < 0 => left pad
    padding_length_right = max(0, segment_id.stop - len(episode))  # segment_id.stop > len(episode) => right pad
    
    assert padding_length_left == padding_length_right == 0 or should_pad

    def pad(x):
        # Pad only for the time dimension (first dimension)
        pad_left = F.pad(x, [0 for _ in range(2*(x.ndim - 1))] + [padding_length_left, 0]) if padding_length_left > 0 else x
        return F.pad(pad_left, [0 for _ in range(2*(x.ndim - 1))] + [0, padding_length_right]) if padding_length_right > 0 else pad_left
    
    # Start and stop w/o padding (it's within a single Episode)
    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)
    mask_padding = torch.cat((
        torch.zeros(padding_length_left), torch.ones(stop - start), torch.zeros(padding_length_right)
    )).bool()

    return Segment(
        pad(episode.obs[start:stop]),
        pad(episode.act[start:stop]),
        pad(episode.rew[start:stop]),
        pad(episode.end[start:stop]),
        pad(episode.trunc[start:stop]),
        mask_padding,
        id=SegmentId(segment_id.episode_id, start, stop)
    )


class DatasetTraverser:
    def __init__(self, dataset, batch_size: int, chunk_size: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size  # sequence length of each Segment
        self._num_batches = math.ceil(
            sum([math.ceil(dataset.lengths[episode_id] / chunk_size) - int(dataset.lengths[episode_id] % chunk_size == 1)
                 for episode_id in range(dataset.num_episodes)]
            ) / batch_size
        )
        # int(dataset.lengths[episode_id] % chunk_size == 1) is because we throw away
        # any Segment with just 1 frame as effective size

    def __len__(self):
        return self._num_batches
    
    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            # Chunk episode into segments of chunk_size frames (stop can go beyond episode, it's padding)
            # And extend the "chunks" list
            chunks.extend(make_segment(episode, SegmentId(episode_id, start=i * self.chunk_size, stop=(i + 1) * self.chunk_size), should_pad=True) for i in range(math.ceil(len(episode) / self.chunk_size)))
            # Note: starts and stops of segments differ from BatchSampler. No randomness
            if chunks[-1].effective_size < 2:
                # if just 1 frame in last chunk, throw it away
                chunks.pop()
            
            # Yield batch of Segments (of length chunk_size)
            while len(chunks) >= self.batch_size:
                yield collate_segments_to_batch(chunks[:self.batch_size])
                # "pop" the batch
                chunks = chunks[self.batch_size:]

        # Yield the last batch (may have less Segments)
        if len(chunks) > 0:
            yield collate_segments_to_batch(chunks)