import copy
import unittest

from scripts.build_routeD_paper_tables import (
    AuditMismatch,
    EXPECTED_HASHES,
    REQUIRED_SCOPE_SENTENCE,
    build_paper_payload,
    render_latex,
    render_markdown,
)


def _stats(mean, low, high, videos, positive, negative, tie):
    return {
        "videos": videos,
        "mean": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_videos": positive,
        "negative_videos": negative,
        "tie_videos": tie,
        "bootstrap_resamples": 20000,
    }


def make_final_audit():
    metric_validity = {}
    for system in ("baseline", "routeD_open", "routeD_closed"):
        metric_validity[system] = {
            "AJ": {"finite_videos": 1138, "nonfinite_videos": 6},
            "OA": {"finite_videos": 1138, "nonfinite_videos": 6},
            "delta_avg": {"finite_videos": 1137, "nonfinite_videos": 7},
        }
    return {
        "official_protocol_pass": True,
        "primary_pass": True,
        "paper_claim_eligible": True,
        "protocol_sha256": EXPECTED_HASHES["protocol_sha256"],
        "merged_sha256": EXPECTED_HASHES["merged_sha256"],
        "paired_sha256": EXPECTED_HASHES["paired_sha256"],
        "official_protocol_summary": {
            "annotation_groups": 1147,
            "materialized_groups": 1144,
            "missing_groups": 3,
            "metric_coordinate_contract": "x * width, y * height",
        },
        "aggregate": {
            "baseline": {
                "AJ": 0.3249447807008109,
                "OA": 0.9398428444023879,
                "delta_avg": 0.42046051513808747,
            },
            "routeD_open": {
                "AJ": 0.3373900717715783,
                "OA": 0.9398428444023879,
                "delta_avg": 0.4365033011038564,
            },
            "routeD_closed": {
                "AJ": 0.34798777423256194,
                "OA": 0.9398428444023879,
                "delta_avg": 0.4479908188009542,
            },
        },
        "delta_routeD_vs_baseline": {
            "open_vs_baseline": {
                "AJ": 0.012445291070767417,
                "delta_avg": 0.016042785965768913,
            },
            "closed_vs_baseline": {
                "AJ": 0.023042993531751044,
                "delta_avg": 0.02753030366286674,
            },
            "closed_vs_open": {
                "AJ": 0.010597702460983627,
                "delta_avg": 0.011487517697097827,
            },
        },
        "invalid_video_count": 7,
        "invalid_videos": [
            {"sample": i, "video_name": f"undefined_{i}"} for i in range(7)
        ],
        "metric_validity": metric_validity,
    }


def make_paired():
    return {
        "paper_claim_eligible": True,
        "paired_bootstrap": {
            "open_vs_baseline": {
                "AJ": _stats(
                    0.0124452910707674,
                    0.011200122495572201,
                    0.01371487770862955,
                    1138,
                    770,
                    188,
                    180,
                ),
                "delta_avg": _stats(
                    0.016042785965768906,
                    0.014671329920135522,
                    0.017425029714860183,
                    1137,
                    784,
                    169,
                    184,
                ),
            },
            "closed_vs_baseline": {
                "AJ": _stats(
                    0.023042993531751075,
                    0.02056717201686339,
                    0.02556867030613993,
                    1138,
                    775,
                    199,
                    164,
                ),
                "delta_avg": _stats(
                    0.02753030366286673,
                    0.024800232453642643,
                    0.03026046249105195,
                    1137,
                    783,
                    188,
                    166,
                ),
            },
            "closed_vs_open": {
                "AJ": _stats(
                    0.010597702460983675,
                    0.008804620374081439,
                    0.012395053200525674,
                    1138,
                    673,
                    274,
                    191,
                ),
                "delta_avg": _stats(
                    0.011487517697097823,
                    0.009494177881478579,
                    0.013464582784450176,
                    1137,
                    654,
                    289,
                    194,
                ),
            },
        },
        "failure_audit": {
            "worst_closed_AJ_videos": [
                {
                    "sample": 113,
                    "video_name": "kinetics_source_s000_p000113_kinetics_s000_000113",
                    "closed_vs_baseline_AJ": -0.2976264864079905,
                    "closed_vs_baseline_delta_avg": -0.27485307763686967,
                    "mean_trajectory_diff_px": 6.26536750793457,
                },
                {
                    "sample": 1115,
                    "video_name": "kinetics_source_s009_p000080_kinetics_s000_000080",
                    "closed_vs_baseline_AJ": -0.25226620093881064,
                    "closed_vs_baseline_delta_avg": -0.2648870636550308,
                    "mean_trajectory_diff_px": 9.612665176391602,
                },
                {
                    "sample": 954,
                    "video_name": "kinetics_source_s008_p000034_kinetics_s000_000034",
                    "closed_vs_baseline_AJ": -0.15953708873379868,
                    "closed_vs_baseline_delta_avg": -0.09607731665719144,
                    "mean_trajectory_diff_px": 5.850607395172119,
                },
            ]
        },
    }


class TestBuildRouteDPaperTables(unittest.TestCase):
    def build(self, final_audit=None, paired=None):
        return build_paper_payload(
            final_audit or make_final_audit(),
            paired or make_paired(),
            final_audit_sha256=EXPECTED_HASHES["final_audit_sha256"],
            paired_sha256=EXPECTED_HASHES["paired_sha256"],
        )

    def test_valid_payload_uses_canonical_scope_and_numbers(self):
        payload = self.build()
        self.assertEqual(payload["scope"]["required_wording"], REQUIRED_SCOPE_SENTENCE)
        self.assertEqual(payload["scope"]["official_annotation_groups"], 1147)
        self.assertEqual(payload["scope"]["materialized_groups"], 1144)
        self.assertAlmostEqual(payload["systems"]["baseline"]["AJ"], 0.3249447807008109)
        self.assertAlmostEqual(payload["systems"]["routeD_closed"]["AJ"], 0.34798777423256194)
        self.assertAlmostEqual(payload["paper_scale"]["absolute_AJ_gain_points"], 2.3042993531751044)
        self.assertEqual(len(payload["severe_failures"]), 3)

    def test_rendered_tables_exclude_superseded_scope_language(self):
        payload = self.build()
        markdown = render_markdown(payload)
        latex = render_latex(payload)
        for text in (markdown, latex):
            self.assertNotIn("1,189", text)
            self.assertNotIn("1,000", text)
            self.assertNotIn("255-scale", text)
            self.assertNotIn("official train-only split", text)
        self.assertIn("NaN, never replaced by zero", markdown)
        self.assertIn("8cf2af93c6629542", markdown)
        self.assertIn("-0.297626", markdown)
        self.assertIn(r"\begin{table}[t]", latex)
        self.assertNotIn(r"\\begin{table}[t]", latex)
        self.assertIn(r"System & AJ $\uparrow$", latex)
        self.assertTrue(any(line.endswith(r"\\") for line in latex.splitlines()))

    def test_rejects_legacy_255_scale_contract(self):
        final_audit = make_final_audit()
        final_audit["official_protocol_summary"]["metric_coordinate_contract"] = (
            "x * (width - 1), y * (height - 1)"
        )
        with self.assertRaisesRegex(AuditMismatch, "coordinate contract"):
            self.build(final_audit=final_audit)

    def test_rejects_1189_as_release_csv_count(self):
        final_audit = make_final_audit()
        final_audit["official_protocol_summary"]["annotation_groups"] = 1189
        final_audit["official_protocol_summary"]["missing_groups"] = 45
        with self.assertRaisesRegex(AuditMismatch, "annotation-group count"):
            self.build(final_audit=final_audit)

    def test_rejects_fixed_1000_scope(self):
        final_audit = make_final_audit()
        final_audit["official_protocol_summary"]["materialized_groups"] = 1000
        final_audit["official_protocol_summary"]["missing_groups"] = 147
        with self.assertRaisesRegex(AuditMismatch, "materialized-group count"):
            self.build(final_audit=final_audit)

    def test_rejects_nan_zero_imputation_accounting(self):
        final_audit = make_final_audit()
        final_audit["invalid_video_count"] = 0
        final_audit["invalid_videos"] = []
        for system in final_audit["metric_validity"].values():
            system["AJ"] = {"finite_videos": 1144, "nonfinite_videos": 0}
            system["delta_avg"] = {"finite_videos": 1144, "nonfinite_videos": 0}
        with self.assertRaisesRegex(AuditMismatch, "undefined-metric video count"):
            self.build(final_audit=final_audit)

    def test_rejects_baseline_open_closed_metric_drift(self):
        for system in ("baseline", "routeD_open", "routeD_closed"):
            final_audit = make_final_audit()
            final_audit["aggregate"][system]["AJ"] += 0.001
            with self.subTest(system=system):
                with self.assertRaisesRegex(AuditMismatch, rf"{system}\.AJ drifted"):
                    self.build(final_audit=final_audit)

    def test_rejects_artifact_hash_drift(self):
        with self.assertRaisesRegex(AuditMismatch, "final audit SHA-256"):
            build_paper_payload(
                make_final_audit(),
                make_paired(),
                final_audit_sha256="0" * 64,
                paired_sha256=EXPECTED_HASHES["paired_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
