#!/usr/bin/env python3
"""Render audited Route-D paper tables from the corrected final artifacts.

This script is intentionally strict. It refuses inputs whose hashes, official
scope, coordinate contract, finite-pair accounting, or headline metrics drift
from the frozen corrected official-scale evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

SYSTEMS = (
    ("baseline", "Independent baseline/local"),
    ("routeD_open", "Route-D open loop"),
    ("routeD_closed", "Route-D closed loop"),
)
COMPARISONS = (
    ("open_vs_baseline", "Open loop vs baseline"),
    ("closed_vs_baseline", "Closed loop vs baseline"),
    ("closed_vs_open", "Closed loop vs open loop"),
)
METRICS = (
    ("AJ", "Average Jaccard"),
    ("delta_avg", "Average point-threshold accuracy"),
)

EXPECTED_HASHES = {
    "protocol_sha256": "61f1458ee0e62b865cde85d9f77e765b79da567add8fc277fe10599b0db13155",
    "merged_sha256": "4b6d796ed017a86773c67c6447fd12827e9dfc185b0198430eef5f9813bf7b7e",
    "paired_sha256": "f97da99d8e905d765c1f267521e4995b1c940ec2ae59bdcd751159a2b6e38c9a",
    "final_audit_sha256": "8cf2af93c6629542b02062a625fc8c54ac3a14c6c65a6a3fb4111228c450408c",
}
EXPECTED_SCOPE = {
    "official_annotation_groups": 1147,
    "materialized_groups": 1144,
    "missing_groups": 3,
    "invalid_video_count": 7,
}
EXPECTED_SYSTEM_METRICS = {
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
}
EXPECTED_GAINS = {
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
}
EXPECTED_FINITE_VIDEOS = {"AJ": 1138, "delta_avg": 1137}
EXPECTED_NONFINITE_VIDEOS = {"AJ": 6, "OA": 6, "delta_avg": 7}
EXPECTED_BOOTSTRAP_RESAMPLES = 20_000
EXPECTED_COORDINATE_CONTRACT = "x * width, y * height"
FLOAT_ABS_TOL = 5e-12

REQUIRED_SCOPE_SENTENCE = (
    "Exact order-preserving local materialization of 1,144 of the 1,147 uniquely "
    "annotated video segments in the byte-verified official TAP-Vid-Kinetics "
    "release CSV, evaluated with pinned official metric formulas and a frozen "
    "controller. Three release-CSV segments were not materialized."
)
UNDEFINED_METRIC_POLICY = (
    "Undefined official metrics are preserved as NaN, never replaced by zero, "
    "and excluded only from the affected finite-pair bootstrap."
)


class AuditMismatch(ValueError):
    """Raised when a paper-table input differs from the frozen authority."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Required key is missing: {key}")
    return mapping[key]


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AuditMismatch(f"{label} drifted: expected {expected!r}, got {actual!r}")


def require_close(actual: Any, expected: float, label: str) -> float:
    value = float(actual)
    if not math.isfinite(value):
        raise AuditMismatch(f"{label} must be finite, got {value!r}")
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=FLOAT_ABS_TOL):
        raise AuditMismatch(
            f"{label} drifted: expected {expected:.17g}, got {value:.17g}"
        )
    return value


def _validate_hashes(
    final_audit: Mapping[str, Any],
    *,
    final_audit_sha256: str,
    paired_sha256: str,
) -> None:
    require_equal(
        final_audit_sha256,
        EXPECTED_HASHES["final_audit_sha256"],
        "final audit SHA-256",
    )
    require_equal(
        paired_sha256,
        EXPECTED_HASHES["paired_sha256"],
        "paired artifact SHA-256",
    )
    require_equal(
        str(require(final_audit, "protocol_sha256")),
        EXPECTED_HASHES["protocol_sha256"],
        "protocol SHA-256",
    )
    require_equal(
        str(require(final_audit, "merged_sha256")),
        EXPECTED_HASHES["merged_sha256"],
        "merged artifact SHA-256",
    )
    require_equal(
        str(require(final_audit, "paired_sha256")),
        EXPECTED_HASHES["paired_sha256"],
        "paired SHA-256 recorded by final audit",
    )


def _validate_metric_validity(final_audit: Mapping[str, Any]) -> None:
    validity = require(final_audit, "metric_validity")
    for system_key, _ in SYSTEMS:
        system_validity = require(validity, system_key)
        for metric_key, expected_nonfinite in EXPECTED_NONFINITE_VIDEOS.items():
            metric_validity = require(system_validity, metric_key)
            require_equal(
                int(require(metric_validity, "nonfinite_videos")),
                expected_nonfinite,
                f"{system_key}.{metric_key} nonfinite-video count",
            )
            require_equal(
                int(require(metric_validity, "finite_videos")) + expected_nonfinite,
                EXPECTED_SCOPE["materialized_groups"],
                f"{system_key}.{metric_key} finite/nonfinite accounting",
            )


def _extract_severe_failures(paired: Mapping[str, Any]) -> list[dict[str, Any]]:
    failure_audit = require(paired, "failure_audit")
    rows = list(require(failure_audit, "worst_closed_AJ_videos"))
    if len(rows) < 3:
        raise AuditMismatch("Paired artifact must retain at least three severe failures")
    failures: list[dict[str, Any]] = []
    for row in rows[:3]:
        failures.append(
            {
                "sample": int(require(row, "sample")),
                "video_name": str(require(row, "video_name")),
                "closed_vs_baseline_AJ": float(require(row, "closed_vs_baseline_AJ")),
                "closed_vs_baseline_delta_avg": float(
                    require(row, "closed_vs_baseline_delta_avg")
                ),
                "mean_trajectory_diff_px": float(
                    require(row, "mean_trajectory_diff_px")
                ),
            }
        )
    if not all(row["closed_vs_baseline_AJ"] < 0.0 for row in failures):
        raise AuditMismatch("Severe-failure rows must retain negative closed-loop AJ deltas")
    return failures


def build_paper_payload(
    final_audit: Mapping[str, Any],
    paired: Mapping[str, Any],
    *,
    final_audit_sha256: str,
    paired_sha256: str,
) -> dict[str, Any]:
    """Build and validate the single source-of-truth paper payload."""
    for key in ("official_protocol_pass", "primary_pass", "paper_claim_eligible"):
        if not bool(require(final_audit, key)):
            raise AuditMismatch(f"Final audit decision failed: {key}")
    if not bool(require(paired, "paper_claim_eligible")):
        raise AuditMismatch("Paired artifact is not paper-claim eligible")

    _validate_hashes(
        final_audit,
        final_audit_sha256=final_audit_sha256,
        paired_sha256=paired_sha256,
    )

    aggregate = require(final_audit, "aggregate")
    deltas = require(final_audit, "delta_routeD_vs_baseline")
    paired_bootstrap = require(paired, "paired_bootstrap")
    official = require(final_audit, "official_protocol_summary")

    annotation_groups = int(require(official, "annotation_groups"))
    materialized_groups = int(require(official, "materialized_groups"))
    missing_groups = int(require(official, "missing_groups"))
    invalid_video_count = int(require(final_audit, "invalid_video_count"))
    require_equal(
        annotation_groups,
        EXPECTED_SCOPE["official_annotation_groups"],
        "release-CSV annotation-group count",
    )
    require_equal(
        materialized_groups,
        EXPECTED_SCOPE["materialized_groups"],
        "materialized-group count",
    )
    require_equal(
        missing_groups,
        EXPECTED_SCOPE["missing_groups"],
        "missing-group count",
    )
    require_equal(
        invalid_video_count,
        EXPECTED_SCOPE["invalid_video_count"],
        "undefined-metric video count",
    )
    require_equal(
        annotation_groups,
        materialized_groups + missing_groups,
        "official package scope accounting",
    )
    require_equal(
        str(require(official, "metric_coordinate_contract")),
        EXPECTED_COORDINATE_CONTRACT,
        "metric coordinate contract",
    )
    _validate_metric_validity(final_audit)

    systems: dict[str, Any] = {}
    for key, label in SYSTEMS:
        row = require(aggregate, key)
        expected = EXPECTED_SYSTEM_METRICS[key]
        systems[key] = {
            "label": label,
            "AJ": require_close(require(row, "AJ"), expected["AJ"], f"{key}.AJ"),
            "OA": require_close(require(row, "OA"), expected["OA"], f"{key}.OA"),
            "delta_avg": require_close(
                require(row, "delta_avg"), expected["delta_avg"], f"{key}.delta_avg"
            ),
        }

    comparisons: dict[str, Any] = {}
    for key, label in COMPARISONS:
        delta_row = require(deltas, key)
        stats_row = require(paired_bootstrap, key)
        metrics: dict[str, Any] = {}
        for metric_key, metric_label in METRICS:
            expected_gain = EXPECTED_GAINS[key][metric_key]
            aggregate_gain = require_close(
                require(delta_row, metric_key),
                expected_gain,
                f"{key}.{metric_key} aggregate gain",
            )
            stats = require(stats_row, metric_key)
            paired_mean = require_close(
                require(stats, "mean"),
                expected_gain,
                f"{key}.{metric_key} paired mean",
            )
            finite_videos = int(require(stats, "videos"))
            require_equal(
                finite_videos,
                EXPECTED_FINITE_VIDEOS[metric_key],
                f"{key}.{metric_key} finite-video count",
            )
            positive = int(require(stats, "positive_videos"))
            negative = int(require(stats, "negative_videos"))
            tie = int(require(stats, "tie_videos"))
            require_equal(
                positive + negative + tie,
                finite_videos,
                f"{key}.{metric_key} support accounting",
            )
            bootstrap_resamples = int(require(stats, "bootstrap_resamples"))
            require_equal(
                bootstrap_resamples,
                EXPECTED_BOOTSTRAP_RESAMPLES,
                f"{key}.{metric_key} bootstrap resamples",
            )
            ci95_low = float(require(stats, "ci95_low"))
            ci95_high = float(require(stats, "ci95_high"))
            if not all(math.isfinite(v) for v in (ci95_low, ci95_high)):
                raise AuditMismatch(f"{key}.{metric_key} CI must be finite")
            if not ci95_low <= paired_mean <= ci95_high:
                raise AuditMismatch(f"{key}.{metric_key} mean lies outside its CI")
            metrics[metric_key] = {
                "label": metric_label,
                "aggregate_gain": aggregate_gain,
                "finite_videos": finite_videos,
                "paired_mean": paired_mean,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "positive_videos": positive,
                "negative_videos": negative,
                "tie_videos": tie,
                "bootstrap_resamples": bootstrap_resamples,
            }
        comparisons[key] = {"label": label, "metrics": metrics}

    # Independently recompute the three headline deltas from aggregate rows.
    aggregate_pairs = {
        "open_vs_baseline": ("routeD_open", "baseline"),
        "closed_vs_baseline": ("routeD_closed", "baseline"),
        "closed_vs_open": ("routeD_closed", "routeD_open"),
    }
    for comparison_key, (lhs, rhs) in aggregate_pairs.items():
        for metric_key, _ in METRICS:
            computed = systems[lhs][metric_key] - systems[rhs][metric_key]
            require_close(
                computed,
                EXPECTED_GAINS[comparison_key][metric_key],
                f"recomputed {comparison_key}.{metric_key}",
            )

    severe_failures = _extract_severe_failures(paired)
    invalid_videos = list(require(final_audit, "invalid_videos"))
    require_equal(len(invalid_videos), invalid_video_count, "invalid-video list length")

    baseline_aj = systems["baseline"]["AJ"]
    closed_aj = systems["routeD_closed"]["AJ"]
    absolute_gain = closed_aj - baseline_aj

    return {
        "kind": "routeD_officialscale_paper_result_source",
        "source_artifacts": {
            "protocol_sha256": EXPECTED_HASHES["protocol_sha256"],
            "merged_sha256": EXPECTED_HASHES["merged_sha256"],
            "paired_sha256": paired_sha256,
            "final_audit_sha256": final_audit_sha256,
        },
        "decisions": {
            "primary_pass": True,
            "official_protocol_pass": True,
            "paper_claim_eligible": True,
        },
        "scope": {
            "official_annotation_groups": annotation_groups,
            "materialized_groups": materialized_groups,
            "missing_groups": missing_groups,
            "metric_coordinate_contract": EXPECTED_COORDINATE_CONTRACT,
            "required_wording": REQUIRED_SCOPE_SENTENCE,
            "invalid_video_count": invalid_video_count,
            "invalid_videos": invalid_videos,
            "undefined_metric_policy": UNDEFINED_METRIC_POLICY,
        },
        "systems": systems,
        "comparisons": comparisons,
        "paper_scale": {
            "baseline_AJ_points": baseline_aj * 100.0,
            "closed_loop_AJ_points": closed_aj * 100.0,
            "absolute_AJ_gain_points": absolute_gain * 100.0,
            "relative_AJ_gain_percent": absolute_gain / baseline_aj * 100.0,
        },
        "severe_failures": severe_failures,
        "claim_boundary": (
            "Component-level frozen-controller transfer evidence; not a leaderboard, "
            "SOTA, or competitive full-system claim."
        ),
    }


def fmt(value: float, signed: bool = False) -> str:
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _assert_render_safety(text: str) -> None:
    for forbidden in ("1,189", "fixed 1,000", "255-scale", "official train-only split"):
        if forbidden.lower() in text.lower():
            raise AuditMismatch(f"Rendered paper artifact contains forbidden wording: {forbidden}")


def render_markdown(payload: Mapping[str, Any]) -> str:
    scope = payload["scope"]
    paper_scale = payload["paper_scale"]
    source = payload["source_artifacts"]
    lines = [
        "# Route-D Corrected Official-Scale Paper Tables",
        "",
        "> Generated by `scripts/build_routeD_paper_tables.py`; do not edit numerical cells manually.",
        "",
        "## Audited scope",
        "",
        str(scope["required_wording"]),
        "",
        f"Coordinate contract: `{scope['metric_coordinate_contract']}`.",
        "",
        str(scope["undefined_metric_policy"]),
        "",
        "## Aggregate metrics",
        "",
        "| System | Average Jaccard | Occlusion accuracy | Average point-threshold accuracy |",
        "|---|---:|---:|---:|",
    ]
    for key, _ in SYSTEMS:
        row = payload["systems"][key]
        lines.append(
            f"| {row['label']} | {fmt(row['AJ'])} | {fmt(row['OA'])} | "
            f"{fmt(row['delta_avg'])} |"
        )
    lines.extend(
        [
            "",
            "### AJ-point view",
            "",
            (
                f"Baseline AJ: {paper_scale['baseline_AJ_points']:.2f}; closed-loop AJ: "
                f"{paper_scale['closed_loop_AJ_points']:.2f}; absolute gain: "
                f"{paper_scale['absolute_AJ_gain_points']:+.2f} AJ points; relative gain: "
                f"{paper_scale['relative_AJ_gain_percent']:+.2f}%."
            ),
            "",
            "## Paired-video comparisons",
            "",
            "| Comparison | Metric | Finite videos | Mean gain | 95% bootstrap CI | Positive / negative / tie |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for comparison_key, _ in COMPARISONS:
        comparison = payload["comparisons"][comparison_key]
        for metric_key, _ in METRICS:
            row = comparison["metrics"][metric_key]
            lines.append(
                f"| {comparison['label']} | {row['label']} | {row['finite_videos']} | "
                f"{fmt(row['paired_mean'], True)} | "
                f"[{fmt(row['ci95_low'], True)}, {fmt(row['ci95_high'], True)}] | "
                f"{row['positive_videos']} / {row['negative_videos']} / {row['tie_videos']} |"
            )
    lines.extend(
        [
            "",
            "## Severe closed-loop failures retained for limitations",
            "",
            "| Video | Delta AJ | Delta point-threshold average | Mean trajectory difference |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in payload["severe_failures"]:
        lines.append(
            f"| `{row['video_name']}` | {fmt(row['closed_vs_baseline_AJ'], True)} | "
            f"{fmt(row['closed_vs_baseline_delta_avg'], True)} | "
            f"{row['mean_trajectory_diff_px']:.3f} px |"
        )
    lines.extend(
        [
            "",
            "## Artifact hashes",
            "",
            "| Artifact | SHA-256 |",
            "|---|---|",
            f"| Protocol | `{source['protocol_sha256']}` |",
            f"| Merged | `{source['merged_sha256']}` |",
            f"| Paired | `{source['paired_sha256']}` |",
            f"| Final audit | `{source['final_audit_sha256']}` |",
            "",
            "## Claim boundary",
            "",
            str(payload["claim_boundary"]),
            "",
        ]
    )
    rendered = "\n".join(lines)
    _assert_render_safety(rendered)
    return rendered


def tex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def render_latex(payload: Mapping[str, Any]) -> str:
    row_end = "\\\\"
    lines = [
        "% Generated by scripts/build_routeD_paper_tables.py. Do not edit manually.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Corrected official-scale TAP-Vid-Kinetics results on the exact local materialization of 1,144 of 1,147 release-CSV segments.}",
        "\\label{tab:routed-main}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        f"System & AJ $\\uparrow$ & OA $\\uparrow$ & $\\delta_{{\\mathrm{{avg}}}}$ $\\uparrow$ {row_end}",
        "\\midrule",
    ]
    for key, _ in SYSTEMS:
        row = payload["systems"][key]
        lines.append(
            f"{tex_escape(row['label'])} & {fmt(row['AJ'])} & {fmt(row['OA'])} & "
            f"{fmt(row['delta_avg'])} {row_end}"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    lines.extend(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Paired-video bootstrap comparisons with 20,000 video-level resamples and seed 17.}",
            "\\label{tab:routed-paired}",
            "\\begin{tabular}{llrrrr}",
            "\\toprule",
            f"Comparison & Metric & $N$ & Mean & 95\\% CI & $+/-/=$ {row_end}",
            "\\midrule",
        ]
    )
    for comparison_key, _ in COMPARISONS:
        comparison = payload["comparisons"][comparison_key]
        for metric_key, _ in METRICS:
            row = comparison["metrics"][metric_key]
            ci = f"[{fmt(row['ci95_low'], True)}, {fmt(row['ci95_high'], True)}]"
            support = f"{row['positive_videos']}/{row['negative_videos']}/{row['tie_videos']}"
            lines.append(
                f"{tex_escape(comparison['label'])} & {tex_escape(row['label'])} & "
                f"{row['finite_videos']} & {fmt(row['paired_mean'], True)} & {ci} & "
                f"{support} {row_end}"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    rendered = "\n".join(lines)
    _assert_render_safety(rendered)
    return rendered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-audit", required=True, type=Path)
    parser.add_argument("--paired", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--output-tex", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    final_audit = json.loads(args.final_audit.read_text(encoding="utf-8"))
    paired = json.loads(args.paired.read_text(encoding="utf-8"))
    payload = build_paper_payload(
        final_audit,
        paired,
        final_audit_sha256=sha256_file(args.final_audit),
        paired_sha256=sha256_file(args.paired),
    )
    for path in (args.output_json, args.output_markdown, args.output_tex):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    args.output_tex.write_text(render_latex(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
