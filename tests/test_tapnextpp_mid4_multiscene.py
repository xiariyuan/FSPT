from scripts.audit_tapnextpp_mid4_multiscene import (
    bootstrap_mean_ci,
    evaluate_multiscene_gate,
)


def row(scene, gain, retained=0.9, frame_fraction=0.8):
    return {
        "scene": scene,
        "status": "complete",
        "delta": {
            "mid4_gain_vs_native": gain,
            "mid4_retained_full_query_gain": retained,
            "mid4_improved_frame_fraction": frame_fraction,
        },
    }


def test_bootstrap_is_deterministic():
    left = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=17, samples=1000)
    right = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=17, samples=1000)
    assert left == right
    assert left["lower"] > 0.0


def test_gate_passes_strong_six_scene_signal():
    rows = [row(str(i), gain) for i, gain in enumerate([2.1, 2.2, 2.3, 2.4, 2.5, 2.6])]
    gate = evaluate_multiscene_gate(rows, expected_scenes=6)
    assert gate["pass"]
    assert gate["decision"] == "ALLOW_FIT_ONLY_MID4_RECONSTRUCTOR_DESIGN"
    assert not gate["training_allowed"]


def test_gate_fails_closed_on_missing_scene():
    rows = [row(str(i), 3.0) for i in range(5)]
    gate = evaluate_multiscene_gate(rows, expected_scenes=6)
    assert not gate["pass"]
    assert not gate["checks"]["all_six_scenes_complete"]


def test_gate_fails_on_one_large_regression():
    rows = [row(str(i), gain) for i, gain in enumerate([3.0, 3.0, 3.0, 3.0, 3.0, -1.1])]
    gate = evaluate_multiscene_gate(rows, expected_scenes=6)
    assert not gate["pass"]
    assert not gate["checks"]["no_scene_regression_worse_than_1px"]
