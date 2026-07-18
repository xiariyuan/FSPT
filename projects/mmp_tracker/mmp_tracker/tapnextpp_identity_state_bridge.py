"""Query-token state persistence for recurrent TAP models.

TAPNext/TAPNext++ flatten image-patch and point-query tokens into the batch
axis of each recurrent cache.  This module isolates only the trailing point
query tokens, allowing an experiment to persist or replace point identity
state without modifying the current image-token state, other queries, or the
tracker's causal step.

The functions are intentionally model-parameter free.  They establish a
strict interface for later oracle and learned state-bridge experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


IDENTITY_STATE_BRIDGE_SCHEMA_VERSION = "tapnextpp_identity_state_bridge_v0"


@dataclass(frozen=True)
class QueryLayerState:
    """One recurrent layer's cache restricted to point-query tokens."""

    rg_lru_state: torch.Tensor  # B,Q,E
    conv1d_state: torch.Tensor  # B,Q,W,E


@dataclass(frozen=True)
class PersistentQueryState:
    """Persistent identity state extracted at a specific causal step."""

    source_step: int
    query_points: torch.Tensor  # B,Q,3
    image_tokens_per_batch: int
    layers: tuple[QueryLayerState, ...]

    @property
    def batch_size(self) -> int:
        return int(self.query_points.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.query_points.shape[1])


def _state_metadata(state: Any) -> tuple[int, int, int]:
    query_points = getattr(state, "query_points", None)
    hidden_state = getattr(state, "hidden_state", None)
    if not torch.is_tensor(query_points) or query_points.ndim != 3:
        raise ValueError("state.query_points must be a BxQx3 tensor")
    if query_points.shape[-1] != 3:
        raise ValueError("state.query_points must encode [time,y,x]")
    if not isinstance(hidden_state, Sequence) or not hidden_state:
        raise ValueError("state.hidden_state must be a non-empty layer sequence")
    batch, queries = int(query_points.shape[0]), int(query_points.shape[1])
    first = hidden_state[0]
    rg = getattr(first, "rg_lru_state", None)
    conv = getattr(first, "conv1d_state", None)
    if not torch.is_tensor(rg) or rg.ndim < 2:
        raise ValueError("each cache needs rg_lru_state with flattened token axis")
    if not torch.is_tensor(conv) or conv.ndim < 3:
        raise ValueError("each cache needs conv1d_state with flattened token axis")
    if rg.shape[0] != conv.shape[0]:
        raise ValueError("RG-LRU and conv caches have different token counts")
    if rg.shape[0] % batch:
        raise ValueError("flattened token count is not divisible by batch size")
    tokens_per_batch = int(rg.shape[0] // batch)
    image_tokens = tokens_per_batch - queries
    if image_tokens <= 0:
        raise ValueError("state does not contain image tokens before query tokens")
    return batch, queries, image_tokens


def validate_tracking_state(state: Any) -> tuple[int, int, int]:
    """Validate all layers and return (batch, queries, image_tokens)."""
    batch, queries, image_tokens = _state_metadata(state)
    total = batch * (image_tokens + queries)
    for layer_id, cache in enumerate(state.hidden_state):
        rg = getattr(cache, "rg_lru_state", None)
        conv = getattr(cache, "conv1d_state", None)
        if not torch.is_tensor(rg) or not torch.is_tensor(conv):
            raise ValueError(f"layer {layer_id} is not a recurrent cache")
        if int(rg.shape[0]) != total or int(conv.shape[0]) != total:
            raise ValueError(f"layer {layer_id} token count differs from layer zero")
        if rg.device != conv.device:
            raise ValueError(f"layer {layer_id} cache tensors use different devices")
        if rg.dtype != conv.dtype:
            raise ValueError(f"layer {layer_id} cache tensors use different dtypes")
    return batch, queries, image_tokens


def _reshape_tokens(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] % batch:
        raise ValueError("tensor token axis is not divisible by batch size")
    return tensor.reshape(batch, tensor.shape[0] // batch, *tensor.shape[1:])


def extract_persistent_query_state(
    state: Any,
    *,
    clone: bool = True,
    detach: bool = True,
) -> PersistentQueryState:
    """Extract all query-token caches while excluding image-token state."""
    batch, queries, image_tokens = validate_tracking_state(state)
    layers: list[QueryLayerState] = []
    for cache in state.hidden_state:
        rg = _reshape_tokens(cache.rg_lru_state, batch)[:, image_tokens:]
        conv = _reshape_tokens(cache.conv1d_state, batch)[:, image_tokens:]
        if detach:
            rg, conv = rg.detach(), conv.detach()
        if clone:
            rg, conv = rg.clone(), conv.clone()
        layers.append(QueryLayerState(rg_lru_state=rg, conv1d_state=conv))
    query_points = state.query_points.detach() if detach else state.query_points
    if clone:
        query_points = query_points.clone()
    return PersistentQueryState(
        source_step=int(state.step),
        query_points=query_points,
        image_tokens_per_batch=image_tokens,
        layers=tuple(layers),
    )


def _normalize_query_mask(
    query_mask: torch.Tensor | None,
    *,
    batch: int,
    queries: int,
    device: torch.device,
) -> torch.Tensor:
    if query_mask is None:
        return torch.ones(batch, queries, dtype=torch.bool, device=device)
    mask = torch.as_tensor(query_mask, device=device)
    if mask.shape == (queries,):
        mask = mask[None].expand(batch, -1)
    if mask.shape != (batch, queries):
        raise ValueError("query_mask must have shape Q or BxQ")
    return mask.bool()


def _normalize_blend(
    blend: float | torch.Tensor,
    *,
    batch: int,
    queries: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = torch.as_tensor(blend, device=device, dtype=dtype)
    if value.ndim == 0:
        value = value.expand(batch, queries)
    elif value.shape == (queries,):
        value = value[None].expand(batch, -1)
    if value.shape != (batch, queries):
        raise ValueError("blend must be scalar, Q, or BxQ")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError("blend must lie in [0,1]")
    return value


def _replace_cache(cache: Any, rg: torch.Tensor, conv: torch.Tensor) -> Any:
    if hasattr(cache, "_replace"):
        return cache._replace(rg_lru_state=rg, conv1d_state=conv)
    try:
        return type(cache)(rg_lru_state=rg, conv1d_state=conv)
    except TypeError:
        return type(cache)(rg, conv)


def _replace_tracking_state(state: Any, hidden_state: list[Any]) -> Any:
    if hasattr(state, "_replace"):
        return state._replace(hidden_state=hidden_state)
    try:
        return type(state)(
            step=state.step,
            query_points=state.query_points,
            hidden_state=hidden_state,
        )
    except TypeError:
        return type(state)(state.step, state.query_points, hidden_state)


def inject_persistent_query_state(
    current_state: Any,
    persistent_state: PersistentQueryState,
    *,
    query_mask: torch.Tensor | None = None,
    blend: float | torch.Tensor = 1.0,
) -> Any:
    """Blend persistent query caches into current state.

    Current image-token caches, unselected queries, `step`, and `query_points`
    are preserved exactly.  A blend of zero is byte-identical to the current
    recurrent cache values.
    """
    batch, queries, image_tokens = validate_tracking_state(current_state)
    if persistent_state.batch_size != batch or persistent_state.query_count != queries:
        raise ValueError("persistent and current query dimensions differ")
    if persistent_state.image_tokens_per_batch != image_tokens:
        raise ValueError("persistent and current image-token layouts differ")
    if len(persistent_state.layers) != len(current_state.hidden_state):
        raise ValueError("persistent and current layer counts differ")
    if persistent_state.query_points.shape != current_state.query_points.shape:
        raise ValueError("persistent and current query-point shapes differ")

    output: list[Any] = []
    for layer_id, (cache, saved) in enumerate(
        zip(current_state.hidden_state, persistent_state.layers)
    ):
        current_rg = _reshape_tokens(cache.rg_lru_state, batch)
        current_conv = _reshape_tokens(cache.conv1d_state, batch)
        if saved.rg_lru_state.shape != current_rg[:, image_tokens:].shape:
            raise ValueError(f"layer {layer_id} RG-LRU shape mismatch")
        if saved.conv1d_state.shape != current_conv[:, image_tokens:].shape:
            raise ValueError(f"layer {layer_id} conv-cache shape mismatch")
        if saved.rg_lru_state.device != current_rg.device:
            raise ValueError(f"layer {layer_id} persistent state is on another device")
        mask = _normalize_query_mask(
            query_mask,
            batch=batch,
            queries=queries,
            device=current_rg.device,
        )
        alpha = _normalize_blend(
            blend,
            batch=batch,
            queries=queries,
            device=current_rg.device,
            dtype=current_rg.dtype,
        )
        alpha = alpha * mask.to(alpha.dtype)
        rg_alpha = alpha.view(batch, queries, *([1] * (current_rg.ndim - 2)))
        conv_alpha = alpha.view(batch, queries, *([1] * (current_conv.ndim - 2)))

        rg = current_rg.clone()
        conv = current_conv.clone()
        rg_query = rg[:, image_tokens:]
        conv_query = conv[:, image_tokens:]
        rg[:, image_tokens:] = torch.lerp(
            rg_query, saved.rg_lru_state.to(rg_query.dtype), rg_alpha
        )
        conv[:, image_tokens:] = torch.lerp(
            conv_query, saved.conv1d_state.to(conv_query.dtype), conv_alpha
        )
        output.append(
            _replace_cache(
                cache,
                rg.reshape_as(cache.rg_lru_state),
                conv.reshape_as(cache.conv1d_state),
            )
        )
    return _replace_tracking_state(current_state, output)


def query_state_distance(
    left: PersistentQueryState,
    right: PersistentQueryState,
) -> dict[str, torch.Tensor]:
    """Per-query normalized state distance for diagnostics, never a GT label."""
    if (
        left.query_points.shape != right.query_points.shape
        or left.image_tokens_per_batch != right.image_tokens_per_batch
        or len(left.layers) != len(right.layers)
    ):
        raise ValueError("persistent states are structurally incompatible")
    rg_rows = []
    conv_rows = []
    for left_layer, right_layer in zip(left.layers, right.layers):
        rg_delta = (left_layer.rg_lru_state - right_layer.rg_lru_state).float()
        conv_delta = (left_layer.conv1d_state - right_layer.conv1d_state).float()
        rg_rows.append(rg_delta.flatten(2).square().mean(dim=-1))
        conv_rows.append(conv_delta.flatten(2).square().mean(dim=-1))
    rg_mse = torch.stack(rg_rows, dim=-1).mean(dim=-1)
    conv_mse = torch.stack(conv_rows, dim=-1).mean(dim=-1)
    return {
        "rg_lru_mse": rg_mse,
        "conv1d_mse": conv_mse,
        "combined_mse": rg_mse + conv_mse,
    }
