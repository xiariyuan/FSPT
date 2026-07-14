import torch

from projects.mmp_tracker.mmp_tracker import MMPTracker, MMPTrackerConfig


def test_mmp_tracker_forward_shapes():
    config = MMPTrackerConfig()
    model = MMPTracker(config)
    video = torch.randn(2, 4, 3, 64, 64)
    query_points = torch.tensor(
        [
            [[0.0, 0.25, 0.25], [0.0, 0.75, 0.75]],
            [[0.0, 0.30, 0.40], [0.0, 0.60, 0.20]],
        ],
        dtype=torch.float32,
    )
    tracks, visibility, info = model(video, query_points, return_info=True)
    assert tracks.shape == (2, 2, 4, 2)
    assert visibility.shape == (2, 2, 4)
    assert info["confidence"].shape == (2, 2, 4)
    k = 1 + config.global_relocator.topk
    assert info["hypothesis_candidate_points"].shape == (2, 2, 4, k, 2)
    assert info["hypothesis_candidate_quality"].shape == (2, 2, 4, k)
    assert info["hypothesis_candidate_entropy"].shape == (2, 2, 4, k)
    assert info["hypothesis_previous_points"].shape == (2, 2, 4, 2)
    assert info["hypothesis_previous_confidence"].shape == (2, 2, 4)

