#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS_LOG="${STATUS_LOG:-$REPO_ROOT/outputs/megadepth_sceneinfo_pair_prep.status}"
SCENE_INFO_ROOT="/gemini/code/datasets/megadepth/extracted/scene_info"
OUT_PATH="${OUT_PATH:-$REPO_ROOT/data/two_view/megadepth_scene_info_pairs_sampled.jsonl}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS_LOG"
}

mkdir -p "$REPO_ROOT/outputs" "$REPO_ROOT/data/two_view"

while pgrep -f "tar -xf scene_info.tar.gz -C extracted" >/dev/null 2>&1; do
  log "Waiting for scene_info extraction to finish"
  sleep 60
done

log "Generating MegaDepth scene_info sampled pair manifest"
/root/miniconda3/bin/python "$REPO_ROOT/scripts/prepare_megadepth_pairs.py" \
  --scene-info-root "$SCENE_INFO_ROOT" \
  --output "$OUT_PATH" \
  --relative-paths \
  --max-pairs-per-scene 200

log "MegaDepth scene_info pair prep finished"
