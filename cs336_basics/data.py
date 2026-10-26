import torch
import numpy as np

import numpy.typing as npt


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(low=0, high=len(dataset) - context_length, size=batch_size)
    batch_seg = []
    target_seg = []
    for start in starts:
        batch_seg.append(dataset[start : start + context_length])
        target_seg.append(dataset[start + 1 : start + context_length + 1])
    batch = torch.tensor(np.stack(batch_seg), dtype=torch.long, device=device)
    target = torch.tensor(np.stack(target_seg), dtype=torch.long, device=device)
    return (batch, target)
