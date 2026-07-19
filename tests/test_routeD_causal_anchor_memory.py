import pytest
import torch

from projects.mmp_tracker.mmp_tracker.routeD_causal_anchor_memory import (
    extract_immutable_query_anchor_memory,
    select_anchor_rows,
)


class _FakeCoTrackerModel:
    corr_levels = 2
    model_resolution = (16, 20)
    stride = 4

    def __init__(self):
        self.prefix_lengths = []
        self.coordinates = []

    def get_track_feat(self, fmap, queried_frames, coordinates, support_radius):
        self.prefix_lengths.append(int(fmap.shape[1]))
        self.coordinates.append(coordinates.detach().clone())
        points = int(queried_frames.shape[1])
        channels = int(fmap.shape[2])
        selected = fmap[0, queried_frames[0], :, 0, 0]
        feature = selected[:, None, :].permute(1, 0, 2)[None]
        support_tokens = (2 * int(support_radius) + 1) ** 2
        support = feature.expand(1, support_tokens, points, channels).clone()
        return feature, support


def test_query_anchor_extraction_structurally_truncates_future_frames():
    model = _FakeCoTrackerModel()
    pyramid = [
        torch.arange(1 * 8 * 3 * 4 * 5).reshape(1, 8, 3, 4, 5).float(),
        torch.arange(1 * 8 * 3 * 2 * 3).reshape(1, 8, 3, 2, 3).float(),
    ]
    frames = torch.tensor([1, 3])
    coordinates = torch.tensor([[0.0, 0.0], [255.0, 255.0]])
    feature, support = extract_immutable_query_anchor_memory(
        model,
        pyramid,
        frames,
        coordinates,
        input_height=256,
        input_width=256,
        support_radius=1,
    )
    assert model.prefix_lengths == [4, 4]
    assert feature[0].shape == (1, 1, 2, 3)
    assert support[0].shape == (1, 9, 2, 3)
    assert model.coordinates[0][0, 1].tolist() == pytest.approx([19.0 / 4.0, 15.0 / 4.0])
    assert model.coordinates[1][0, 1].tolist() == pytest.approx([19.0 / 8.0, 15.0 / 8.0])


def test_query_anchor_ignores_changes_after_latest_query_frame():
    frames = torch.tensor([0, 2])
    coordinates = torch.tensor([[20.0, 30.0], [40.0, 50.0]])
    original = [torch.randn(1, 6, 4, 5, 5), torch.randn(1, 6, 4, 3, 3)]
    changed = [value.clone() for value in original]
    for value in changed:
        value[:, 3:] = 100000.0
    first = extract_immutable_query_anchor_memory(
        _FakeCoTrackerModel(), original, frames, coordinates, support_radius=1
    )
    second = extract_immutable_query_anchor_memory(
        _FakeCoTrackerModel(), changed, frames, coordinates, support_radius=1
    )
    for left_values, right_values in zip(first, second):
        for left, right in zip(left_values, right_values):
            assert torch.equal(left, right)


def test_query_anchor_rejects_noninteger_or_unobserved_query_frames():
    pyramid = [torch.zeros(1, 3, 2, 2, 2), torch.zeros(1, 3, 2, 1, 1)]
    coordinates = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="integer-valued"):
        extract_immutable_query_anchor_memory(
            _FakeCoTrackerModel(), pyramid, torch.tensor([1.5]), coordinates
        )
    with pytest.raises(ValueError, match="ends before"):
        extract_immutable_query_anchor_memory(
            _FakeCoTrackerModel(), pyramid, torch.tensor([3]), coordinates
        )


def test_select_anchor_rows_preserves_native_token_order():
    values = (torch.arange(1 * 3 * 4 * 2).reshape(1, 3, 4, 2),)
    selected = select_anchor_rows(values, torch.tensor([3, 1]))
    assert torch.equal(selected[0], values[0][:, :, [3, 1]])
