#!/usr/bin/env python3
"""
Generate YAML config variants by sweeping one or more keys.

Why this exists
---------------
When iterating on a research idea, manually editing dozens of YAML files is
error-prone and wastes time. This helper produces *explicit* config files so:
  - each run is reproducible
  - configs can be copied to the server as-is
  - `evaluate.py` (which does not support arbitrary CLI overrides) can be used

Example
-------
  python scripts/make_config_variants.py \\
    --base configs/fspt_routeA_stage3_relocal_longocc_m20_v32_train.yaml \\
    --out-dir outputs/sweeps/occ_corr_thr/configs \\
    --set model.refiner.long_occlusion_occ_corr_threshold=0.05,0.08,0.10,0.12,0.15
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from omegaconf import OmegaConf


def _merge_with_defaults(config, config_dir: Path):
    """
    Merge Hydra-style `defaults:` into a single OmegaConf.

    This intentionally mirrors `train.py:_merge_with_defaults` so that generated
    variants can be evaluated from *any* folder without breaking relative
    defaults resolution.
    """

    defaults = config.pop("defaults", None)
    if not defaults:
        return config

    merged = OmegaConf.create()
    for entry in defaults:
        if isinstance(entry, str):
            if entry == "_self_":
                continue
            base_path = config_dir / entry
            if base_path.suffix == "":
                base_path = base_path.with_suffix(".yaml")
            if not base_path.exists():
                raise FileNotFoundError(f"Default config not found: {base_path}")
            base_cfg = OmegaConf.load(base_path)
            base_cfg = _merge_with_defaults(base_cfg, base_path.parent)
            merged = OmegaConf.merge(merged, base_cfg)
            continue

        if isinstance(entry, dict):
            for key, value in entry.items():
                if key == "_self_":
                    continue
                if value is None:
                    base_path = config_dir / key
                else:
                    base_path = config_dir / key / str(value)
                if base_path.suffix == "":
                    base_path = base_path.with_suffix(".yaml")
                if not base_path.exists():
                    raise FileNotFoundError(f"Default config not found: {base_path}")
                base_cfg = OmegaConf.load(base_path)
                base_cfg = _merge_with_defaults(base_cfg, base_path.parent)
                merged = OmegaConf.merge(merged, base_cfg)
            continue

    return OmegaConf.merge(merged, config)


def _parse_value(raw: str) -> Any:
    s = raw.strip()
    lower = s.lower()
    if lower in ("none", "null", "~"):
        return None
    if lower in ("true", "false"):
        return lower == "true"
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except Exception:
            pass
    if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", s) or re.fullmatch(r"[-+]?\d+([eE][-+]?\d+)", s):
        try:
            return float(s)
        except Exception:
            pass
    return s


def _safe_token(text: str) -> str:
    token = str(text).strip()
    token = token.replace("/", "_").replace("\\", "_")
    token = token.replace(" ", "")
    token = token.replace(":", "_")
    token = token.replace("[", "").replace("]", "")
    token = token.replace("(", "").replace(")", "")
    token = token.replace("{", "").replace("}", "")
    token = token.replace(",", "_")
    token = token.replace("=", "_")
    token = token.replace(".", "_")
    token = re.sub(r"[^A-Za-z0-9_\-]+", "_", token)
    token = re.sub(r"_+", "_", token)
    return token.strip("_") or "x"


def _iter_sweep_items(raw_sets: Iterable[str]) -> List[Tuple[str, List[Any]]]:
    items: List[Tuple[str, List[Any]]] = []
    for raw in raw_sets:
        if "=" not in raw:
            raise ValueError(f"Invalid --set '{raw}' (expected key=values)")
        key, values_raw = raw.split("=", 1)
        key = key.strip()
        values = [_parse_value(v) for v in values_raw.split(",") if v.strip() != ""]
        if not key or not values:
            raise ValueError(f"Invalid --set '{raw}' (empty key or values)")
        items.append((key, values))
    return items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate config variants (sweep).")
    p.add_argument("--base", required=True, help="Base YAML config path.")
    p.add_argument("--out-dir", required=True, help="Directory to write variants.")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="Sweep item: dotted.key=val1,val2,... (repeatable).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    out_dir = Path(args.out_dir)

    cfg = OmegaConf.load(base_path)
    cfg = _merge_with_defaults(cfg, base_path.parent)
    sweep_items = _iter_sweep_items(args.set)
    if not sweep_items:
        raise SystemExit("No --set provided. Nothing to sweep.")

    keys = [k for k, _ in sweep_items]
    values_lists = [vals for _, vals in sweep_items]

    out_dir.mkdir(parents=True, exist_ok=True)

    num_written = 0
    for combo in itertools.product(*values_lists):
        combo_dict: Dict[str, Any] = dict(zip(keys, combo))

        variant = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        for key, value in combo_dict.items():
            OmegaConf.update(variant, key, value, merge=True)

        suffix_parts = []
        for key, value in combo_dict.items():
            key_short = key.split(".")[-1]
            suffix_parts.append(f"{_safe_token(key_short)}-{_safe_token(value)}")
        suffix = "__".join(suffix_parts)

        out_path = out_dir / f"{base_path.stem}__{suffix}.yaml"
        if args.dry_run:
            print(out_path.as_posix())
        else:
            out_path.write_text(OmegaConf.to_yaml(variant), encoding="utf-8")
        num_written += 1

    print(f"Wrote {num_written} configs under {out_dir.as_posix()}")


if __name__ == "__main__":
    main()
