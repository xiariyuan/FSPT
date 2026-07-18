import torch

from mmp_tracker.routeD_point_constellation import (
    PointConstellationConfig,
    combined_constellation_score,
    constellation_coordinates,
    direct_correspondence_similarity,
    sample_point_constellations,
    structural_self_similarity,
)


def test_constellation_coordinates_are_bounded() -> None:
    config = PointConstellationConfig(input_height=32, input_width=40)
    coords = torch.tensor([[[0.0, 0.0], [39.0, 31.0]]])
    result = constellation_coordinates(coords, config)
    assert result.shape == (1, 2, len(config.offsets_xy_px), 2)
    assert float(result[..., 0].min()) == 0.0
    assert float(result[..., 0].max()) == 39.0
    assert float(result[..., 1].min()) == 0.0
    assert float(result[..., 1].max()) == 31.0


def test_identical_constellations_have_unit_similarity() -> None:
    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(2, 13, 8, generator=generator)
    reference = torch.nn.functional.normalize(reference, dim=-1)
    candidates = reference[:, None].repeat(1, 3, 1, 1)
    direct = direct_correspondence_similarity(reference, candidates)
    structure = structural_self_similarity(reference, candidates)
    assert torch.allclose(direct, torch.ones_like(direct), atol=1e-6)
    assert torch.allclose(structure, torch.ones_like(structure), atol=1e-5)


def test_local_structure_breaks_equal_center_tie() -> None:
    generator = torch.Generator().manual_seed(11)
    reference = torch.randn(1, 13, 16, generator=generator)
    reference = torch.nn.functional.normalize(reference, dim=-1)
    correct = reference.clone()
    decoy = torch.randn(1, 13, 16, generator=generator)
    decoy = torch.nn.functional.normalize(decoy, dim=-1)
    # The centre appearance is deliberately identical for both candidates.
    decoy[:, 0] = reference[:, 0]
    candidates = torch.stack((decoy[0], correct[0]), dim=0)[None]
    center = torch.ones(1, 2)
    direct = direct_correspondence_similarity(reference, candidates)
    structure = structural_self_similarity(reference, candidates)
    combined = combined_constellation_score(
        center, direct, structure, direct_weight=0.5, structure_weight=0.5
    )
    assert int(combined.argmax(dim=-1).item()) == 1
    assert float(direct[0, 1]) > float(direct[0, 0])
    assert float(structure[0, 1]) > float(structure[0, 0])


def test_sampling_shape_and_normalization() -> None:
    config = PointConstellationConfig(input_height=32, input_width=32)
    fmap = torch.randn(2, 6, 8, 8)
    coords = torch.tensor(
        [
            [[8.0, 8.0], [24.0, 24.0]],
            [[10.0, 12.0], [20.0, 18.0]],
        ]
    )
    sampled = sample_point_constellations(fmap, coords, config)
    assert sampled.shape == (2, 2, len(config.offsets_xy_px), 6)
    norms = torch.linalg.norm(sampled, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
