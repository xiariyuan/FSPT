import pytest

torch = pytest.importorskip("torch")


def test_sample_affine_identity():
    from utils.affine_augmentation import sample_affine_matrices

    B = 4
    cfg = {
        "flip": False,
        "rotation_deg": 0.0,
        "scale": (1.0, 1.0),
        "translate": (0.0, 0.0),
    }
    theta_o2a, theta_a2o = sample_affine_matrices(
        batch_size=B, device=torch.device("cpu"), dtype=torch.float32, cfg=cfg
    )

    eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(theta_o2a, eye.expand(B, 2, 3), atol=0.0, rtol=0.0)
    assert torch.allclose(theta_a2o, eye.expand(B, 2, 3), atol=0.0, rtol=0.0)


def test_transform_points_invertible_on_in_bounds_subset():
    from utils.affine_augmentation import sample_affine_matrices, transform_points_yx

    torch.manual_seed(0)
    B, N = 2, 128
    cfg = {
        "flip": True,
        "flip_prob": 0.5,
        "rotation_deg": 5.0,
        "scale": (0.95, 1.0),
        "translate": (0.02, 0.02),
    }
    theta_o2a, theta_a2o = sample_affine_matrices(
        batch_size=B, device=torch.device("cpu"), dtype=torch.float32, cfg=cfg
    )

    points = torch.rand(B, N, 2) * 0.6 + 0.2  # keep away from boundaries
    p1, in1 = transform_points_yx(points, theta_o2a)
    p2, in2 = transform_points_yx(p1, theta_a2o)

    mask = in1 & in2
    assert mask.any().item()
    max_err = (p2[mask] - points[mask]).abs().max().item()
    assert max_err < 5e-3


def test_warp_video_affine_identity():
    from utils.affine_augmentation import sample_affine_matrices, warp_video_affine

    torch.manual_seed(0)
    B, T, C, H, W = 2, 3, 3, 16, 16
    video = torch.randn(B, T, C, H, W)

    theta_o2a, theta_a2o = sample_affine_matrices(
        batch_size=B,
        device=video.device,
        dtype=torch.float32,
        cfg={"flip": False, "rotation_deg": 0.0, "scale": (1.0, 1.0), "translate": (0.0, 0.0)},
    )
    warped = warp_video_affine(video, theta_a2o, mode="bilinear", padding_mode="zeros", align_corners=False)
    assert warped.shape == video.shape
    assert torch.allclose(warped, video, atol=1e-5, rtol=1e-5)
