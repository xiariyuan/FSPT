import torch

from projects.mmp_tracker.mmp_tracker.routeD_temporal_identity_features import (
    STATIC_FEATURE_CHANNELS,
    TEMPORAL_FEATURE_CHANNELS,
    assemble_candidate_temporal_identity_features,
    sample_spatiotemporal_descriptors,
)


def test_spatiotemporal_sampling_shape_and_normalization():
    feature = torch.ones(3, 4, 2, 2)
    tracklets = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            [[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
        ]
    )
    value = sample_spatiotemporal_descriptors(
        feature, tracklets, input_height=4, input_width=4
    )
    assert value.shape == (2, 3, 4)
    assert torch.allclose(
        torch.linalg.vector_norm(value, dim=-1), torch.ones(2, 3)
    )


def test_temporal_identity_feature_shapes_and_native_reference():
    descriptor = torch.nn.functional.normalize(torch.randn(3, 4, 6), dim=-1)
    tracklets = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]],
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0], [4.0, 2.0]],
            [[1.0, 3.0], [2.0, 3.0], [3.0, 3.0], [4.0, 3.0]],
        ]
    )
    output = assemble_candidate_temporal_identity_features(
        descriptor_sequence=descriptor,
        query_descriptor=descriptor[0, 0],
        tracklets_xy=tracklets,
        visibility_probability=torch.full((3, 4), 0.8),
        confidence_probability=torch.full((3, 4), 0.7),
        candidate_scores=torch.tensor([0.0, 2.0, 1.0]),
        valid_mask=torch.tensor([True, True, False]),
        query_frame=0,
        query_coordinate_xy=torch.tensor([1.0, 1.0]),
        input_height=8,
        input_width=8,
    )
    temporal = output["temporal_features"]
    static = output["static_features"]
    assert temporal.shape == (3, 4, len(TEMPORAL_FEATURE_CHANNELS))
    assert static.shape == (3, len(STATIC_FEATURE_CHANNELS))
    assert torch.allclose(temporal[0, :, -1], torch.zeros(4))
    assert static[0, STATIC_FEATURE_CHANNELS.index("is_native")] == 1.0
    assert static[2, STATIC_FEATURE_CHANNELS.index("is_valid")] == 0.0
    assert torch.count_nonzero(temporal[2]) == 0
