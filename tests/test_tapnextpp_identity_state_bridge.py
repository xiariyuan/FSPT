from collections import namedtuple
from dataclasses import dataclass

import pytest
import torch

from mmp_tracker.tapnextpp_identity_state_bridge import (
    extract_persistent_query_state,
    inject_persistent_query_state,
    query_state_distance,
    validate_tracking_state,
)

Cache = namedtuple("Cache", "rg_lru_state conv1d_state")


@dataclass
class State:
    step: int
    query_points: torch.Tensor
    hidden_state: list[Cache]


def make_state(offset: float = 0.0) -> State:
    # B=2, image tokens=4, queries=3, flattened token count=14.
    generator = torch.Generator().manual_seed(13)
    layers = []
    for _ in range(3):
        rg = torch.randn(14, 5, generator=generator) + offset
        conv = torch.randn(14, 4, 5, generator=generator) + offset
        layers.append(Cache(rg, conv))
    query = torch.tensor(
        [
            [[0.0, 1.0, 2.0], [0.0, 3.0, 4.0], [0.0, 5.0, 6.0]],
            [[0.0, 7.0, 8.0], [0.0, 9.0, 10.0], [0.0, 11.0, 12.0]],
        ]
    )
    return State(step=17, query_points=query, hidden_state=layers)


def reshaped(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(2, 7, *tensor.shape[1:])


def test_extracts_only_trailing_query_tokens() -> None:
    state = make_state()
    assert validate_tracking_state(state) == (2, 3, 4)
    saved = extract_persistent_query_state(state)
    assert saved.source_step == 17
    assert len(saved.layers) == 3
    assert saved.layers[0].rg_lru_state.shape == (2, 3, 5)
    assert saved.layers[0].conv1d_state.shape == (2, 3, 4, 5)
    assert torch.equal(saved.layers[0].rg_lru_state, reshaped(state.hidden_state[0].rg_lru_state)[:, 4:])


def test_zero_blend_is_exact_current_state() -> None:
    current = make_state(offset=0.0)
    persistent = extract_persistent_query_state(make_state(offset=9.0))
    output = inject_persistent_query_state(current, persistent, blend=0.0)
    assert output.step == current.step
    assert output.query_points is current.query_points
    for left, right in zip(output.hidden_state, current.hidden_state):
        assert torch.equal(left.rg_lru_state, right.rg_lru_state)
        assert torch.equal(left.conv1d_state, right.conv1d_state)


def test_selected_query_replacement_preserves_image_and_other_queries() -> None:
    current = make_state(offset=0.0)
    source = make_state(offset=9.0)
    persistent = extract_persistent_query_state(source)
    mask = torch.tensor([False, True, False])
    output = inject_persistent_query_state(current, persistent, query_mask=mask, blend=1.0)
    for out_cache, current_cache, source_cache in zip(
        output.hidden_state, current.hidden_state, source.hidden_state
    ):
        out_rg = reshaped(out_cache.rg_lru_state)
        cur_rg = reshaped(current_cache.rg_lru_state)
        src_rg = reshaped(source_cache.rg_lru_state)
        assert torch.equal(out_rg[:, :4], cur_rg[:, :4])
        assert torch.equal(out_rg[:, 4], cur_rg[:, 4])
        assert torch.equal(out_rg[:, 5], src_rg[:, 5])
        assert torch.equal(out_rg[:, 6], cur_rg[:, 6])

        out_conv = reshaped(out_cache.conv1d_state)
        cur_conv = reshaped(current_cache.conv1d_state)
        src_conv = reshaped(source_cache.conv1d_state)
        assert torch.equal(out_conv[:, :4], cur_conv[:, :4])
        assert torch.equal(out_conv[:, 4], cur_conv[:, 4])
        assert torch.equal(out_conv[:, 5], src_conv[:, 5])
        assert torch.equal(out_conv[:, 6], cur_conv[:, 6])


def test_half_blend_and_state_distance() -> None:
    current = make_state(offset=0.0)
    source = make_state(offset=4.0)
    saved_current = extract_persistent_query_state(current)
    saved_source = extract_persistent_query_state(source)
    output = inject_persistent_query_state(current, saved_source, blend=0.5)
    saved_output = extract_persistent_query_state(output)
    for out, left, right in zip(saved_output.layers, saved_current.layers, saved_source.layers):
        assert torch.allclose(out.rg_lru_state, (left.rg_lru_state + right.rg_lru_state) / 2)
        assert torch.allclose(out.conv1d_state, (left.conv1d_state + right.conv1d_state) / 2)
    distance = query_state_distance(saved_current, saved_source)
    assert distance["combined_mse"].shape == (2, 3)
    assert bool((distance["combined_mse"] > 0).all())


def test_rejects_incompatible_query_layout() -> None:
    current = make_state()
    source = make_state()
    source.query_points = source.query_points[:, :2]
    persistent = extract_persistent_query_state(source)
    try:
        inject_persistent_query_state(current, persistent)
    except ValueError as error:
        assert "dimensions differ" in str(error)
    else:
        raise AssertionError("expected incompatible query dimensions to fail")


def test_compose_partial_state_selects_only_requested_component_and_layer():
    from mmp_tracker.tapnextpp_identity_state_bridge import (
        compose_persistent_query_state,
        extract_persistent_query_state,
        persistent_state_replaced_fraction,
    )

    current = make_state(offset=0.0)
    donor = make_state(offset=10.0)
    current_q = extract_persistent_query_state(current)
    donor_q = extract_persistent_query_state(donor)
    mixed = compose_persistent_query_state(
        current_q,
        donor_q,
        layer_mask=[False, True, False],
        use_rg_lru=True,
        use_conv1d=False,
    )
    assert torch.equal(mixed.layers[0].rg_lru_state, current_q.layers[0].rg_lru_state)
    assert torch.equal(mixed.layers[1].rg_lru_state, donor_q.layers[1].rg_lru_state)
    assert torch.equal(mixed.layers[1].conv1d_state, current_q.layers[1].conv1d_state)
    assert torch.equal(mixed.layers[2].rg_lru_state, current_q.layers[2].rg_lru_state)
    assert 0.0 < persistent_state_replaced_fraction(current_q, mixed) < 0.5


def test_compose_rejects_wrong_layer_mask_length():
    from mmp_tracker.tapnextpp_identity_state_bridge import (
        compose_persistent_query_state,
        extract_persistent_query_state,
    )

    state = make_state()
    query = extract_persistent_query_state(state)
    with pytest.raises(ValueError, match="layer_mask"):
        compose_persistent_query_state(query, query, layer_mask=[True])


def test_query_state_vector_roundtrip_is_exact():
    from mmp_tracker.tapnextpp_identity_state_bridge import (
        flatten_persistent_query_state,
        unflatten_persistent_query_state,
    )

    state = extract_persistent_query_state(make_state())
    vector, layout = flatten_persistent_query_state(state)
    rebuilt = unflatten_persistent_query_state(state, vector)
    assert vector.shape == (2, 3, layout.total_dim)
    assert layout.block_names == (
        "layer00.rg_lru", "layer00.conv1d",
        "layer01.rg_lru", "layer01.conv1d",
        "layer02.rg_lru", "layer02.conv1d",
    )
    for left, right in zip(state.layers, rebuilt.layers):
        assert torch.equal(left.rg_lru_state, right.rg_lru_state)
        assert torch.equal(left.conv1d_state, right.conv1d_state)


def test_add_flattened_delta_reconstructs_donor_exactly():
    from mmp_tracker.tapnextpp_identity_state_bridge import (
        add_persistent_query_state_delta,
        flatten_persistent_query_state,
    )

    current = extract_persistent_query_state(make_state(offset=0.0))
    donor = extract_persistent_query_state(make_state(offset=3.0))
    current_vector, current_layout = flatten_persistent_query_state(current)
    donor_vector, donor_layout = flatten_persistent_query_state(donor)
    assert current_layout == donor_layout
    repaired = add_persistent_query_state_delta(
        current,
        donor_vector - current_vector,
        source_step=donor.source_step,
    )
    assert repaired.source_step == donor.source_step
    for left, right in zip(repaired.layers, donor.layers):
        assert torch.equal(left.rg_lru_state, right.rg_lru_state)
        assert torch.equal(left.conv1d_state, right.conv1d_state)


def test_unflatten_rejects_wrong_dimension():
    from mmp_tracker.tapnextpp_identity_state_bridge import (
        flatten_persistent_query_state,
        unflatten_persistent_query_state,
    )

    state = extract_persistent_query_state(make_state())
    vector, _ = flatten_persistent_query_state(state)
    with pytest.raises(ValueError, match="dimension"):
        unflatten_persistent_query_state(state, vector[..., :-1])
