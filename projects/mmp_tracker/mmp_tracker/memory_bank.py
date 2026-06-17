from dataclasses import dataclass

import torch


@dataclass
class VisibleMemoryBank:
    descriptors: torch.Tensor
    positions: torch.Tensor
    visibility: torch.Tensor
    valid: torch.Tensor


def initialize_memory_bank(query_desc: torch.Tensor, query_points: torch.Tensor, capacity: int) -> VisibleMemoryBank:
    batch, num_points, channels = query_desc.shape
    capacity = max(int(capacity), 1)
    descriptors = torch.zeros(batch, num_points, capacity, channels, device=query_desc.device, dtype=query_desc.dtype)
    positions = torch.zeros(batch, num_points, capacity, 2, device=query_desc.device, dtype=query_desc.dtype)
    visibility = torch.zeros(batch, num_points, capacity, device=query_desc.device, dtype=query_desc.dtype)
    valid = torch.zeros(batch, num_points, capacity, device=query_desc.device, dtype=torch.bool)
    descriptors[:, :, 0] = query_desc
    positions[:, :, 0] = query_points
    visibility[:, :, 0] = 1.0
    valid[:, :, 0] = True
    return VisibleMemoryBank(descriptors=descriptors, positions=positions, visibility=visibility, valid=valid)


def append_memory(
    memory: VisibleMemoryBank,
    descriptors: torch.Tensor,
    positions: torch.Tensor,
    visibility: torch.Tensor,
    capacity: int,
) -> VisibleMemoryBank:
    capacity = max(int(capacity), 1)
    if memory.descriptors.shape[2] != capacity:
        raise ValueError(f"Memory capacity mismatch: bank={memory.descriptors.shape[2]} requested={capacity}")

    write_mask = visibility > 0
    shifted_desc = torch.cat([memory.descriptors[:, :, 1:, :], descriptors.unsqueeze(2)], dim=2)
    shifted_pos = torch.cat([memory.positions[:, :, 1:, :], positions.unsqueeze(2)], dim=2)
    shifted_vis = torch.cat([memory.visibility[:, :, 1:], visibility.unsqueeze(2)], dim=2)
    shifted_valid = torch.cat(
        [memory.valid[:, :, 1:], torch.ones_like(write_mask, dtype=torch.bool).unsqueeze(2)],
        dim=2,
    )

    desc_mask = write_mask.unsqueeze(-1).unsqueeze(-1)
    scalar_mask = write_mask.unsqueeze(-1)
    desc = torch.where(desc_mask, shifted_desc, memory.descriptors)
    pos = torch.where(desc_mask, shifted_pos, memory.positions)
    vis = torch.where(scalar_mask, shifted_vis, memory.visibility)
    valid = torch.where(scalar_mask, shifted_valid, memory.valid)
    return VisibleMemoryBank(descriptors=desc, positions=pos, visibility=vis, valid=valid)
