#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/gemini/code/FSPT}"
cd "$REPO_ROOT"

CURRENT_LOG="${CURRENT_LOG:-outputs/full_lockin_queue_.log}"
SEED_LOG="${SEED_LOG:-outputs/full_lockin_queue_seed44_45.log}"
ANCHORBANK_PAYLOAD="${ANCHORBANK_PAYLOAD:-outputs/anchorbank_stage0_payload_20260314.tar.gz}"
ANCHORBANK_LOG="${ANCHORBANK_LOG:-outputs/anchorbank_stage0_autopilot.log}"
PAYLOAD_TAR="${PAYLOAD_TAR:-outputs/post_lockin_payload_20260312.tar.gz}"
STATUS_LOG="${STATUS_LOG:-outputs/remote_autopilot_after_current_queue.status}"
POST_LOG="${POST_LOG:-outputs/post_lockin_coarse_vs_refine_autopilot.log}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS_LOG"
}

validate_post_lockin_payload() {
  log "Validating post-lockin payload and extracted scripts"
  tar -tzf "$PAYLOAD_TAR" >/dev/null
  bash -n projects/mmp_tracker/run_full_lockin_local_vs_coarse_vs_refine.sh
  bash -n projects/mmp_tracker/run_post_lockin_coarse_vs_refine_queue.sh
  /root/miniconda3/bin/python -m py_compile \
    projects/mmp_tracker/mmp_tracker/config.py \
    projects/mmp_tracker/mmp_tracker/model.py \
    projects/mmp_tracker/train_mmp.py
  if [[ ! -x projects/mmp_tracker/run_remote_train.sh ]]; then
    log "Runner missing or not executable: projects/mmp_tracker/run_remote_train.sh"
    exit 1
  fi
  log "Post-lockin payload validation passed"
}

validate_anchorbank_payload() {
  log "Validating AnchorBank payload and extracted scripts"
  tar -tzf "$ANCHORBANK_PAYLOAD" >/dev/null
  bash -n projects/mmp_tracker/run_anchorbank_pairwise_stage0_queue.sh
  /root/miniconda3/bin/python -m py_compile \
    projects/mmp_tracker/mmp_tracker/config.py \
    projects/mmp_tracker/mmp_tracker/blocks.py \
    projects/mmp_tracker/mmp_tracker/global_relocator.py \
    projects/mmp_tracker/mmp_tracker/model.py \
    projects/mmp_tracker/train_mmp.py \
    aim_track/backbones/dinov2_wrapper.py \
    aim_track/adapters/minimal_vit_adapter.py \
    aim_track/global_matcher/pairwise_global_matcher.py
  if [[ ! -x projects/mmp_tracker/run_remote_train.sh ]]; then
    log "Runner missing or not executable: projects/mmp_tracker/run_remote_train.sh"
    exit 1
  fi
  log "AnchorBank payload validation passed"
}

wait_for_log_flag() {
  local file="$1"
  local pattern="$2"
  while true; do
    if [[ -f "$file" ]] && grep -q "$pattern" "$file"; then
      return 0
    fi
    sleep 300
  done
}

seed_queue_running() {
  if pgrep -af "run_full_lockin_local_vs_nocommit.sh 44 45" >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f "$SEED_LOG" ]] && grep -q "Launching full local seed=44" "$SEED_LOG" && ! grep -q "Full lock-in finished" "$SEED_LOG"; then
    return 0
  fi
  return 1
}

log "Autopilot waiting for current full local-vs-nocommit queue"
wait_for_log_flag "$CURRENT_LOG" "Full lock-in finished"
log "Current queue finished"

if [[ -f "$SEED_LOG" ]] && grep -q "Full lock-in finished" "$SEED_LOG"; then
  log "Seed44/45 queue already complete"
elif seed_queue_running; then
  log "Seed44/45 queue already running; waiting"
  wait_for_log_flag "$SEED_LOG" "Full lock-in finished"
  log "Seed44/45 queue finished"
else
  log "Launching seed44/45 queue directly from autopilot"
  bash projects/mmp_tracker/run_full_lockin_local_vs_nocommit.sh 44 45 > "$SEED_LOG" 2>&1
  log "Seed44/45 queue finished"
fi

if [[ -f "$ANCHORBANK_PAYLOAD" ]]; then
  log "Deploying AnchorBank payload"
  tar -xzf "$ANCHORBANK_PAYLOAD" -C "$REPO_ROOT"
  chmod +x projects/mmp_tracker/run_anchorbank_pairwise_stage0_queue.sh
  validate_anchorbank_payload
  log "AnchorBank payload deployed"

  log "Launching AnchorBank Stage-0 queue"
  if bash projects/mmp_tracker/run_anchorbank_pairwise_stage0_queue.sh > "$ANCHORBANK_LOG" 2>&1; then
    log "AnchorBank Stage-0 queue finished successfully"
  else
    log "AnchorBank Stage-0 queue failed; inspect $ANCHORBANK_LOG"
    exit 1
  fi
else
  log "AnchorBank payload not found; skipping AnchorBank Stage-0 queue"
fi

if [[ ! -f "$PAYLOAD_TAR" ]]; then
  log "Payload tar not found: $PAYLOAD_TAR"
  exit 1
fi

log "Deploying post-lockin payload"
tar -xzf "$PAYLOAD_TAR" -C "$REPO_ROOT"
chmod +x projects/mmp_tracker/run_full_lockin_local_vs_coarse_vs_refine.sh
chmod +x projects/mmp_tracker/run_post_lockin_coarse_vs_refine_queue.sh
validate_post_lockin_payload
log "Payload deployed"

log "Launching post-lockin coarse-vs-refine queue"
if bash projects/mmp_tracker/run_post_lockin_coarse_vs_refine_queue.sh > "$POST_LOG" 2>&1; then
  log "Post-lockin coarse-vs-refine queue finished successfully"
else
  log "Post-lockin coarse-vs-refine queue failed; inspect $POST_LOG"
  exit 1
fi
