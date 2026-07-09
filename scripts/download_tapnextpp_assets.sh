#!/usr/bin/env bash
set -euo pipefail

cd /gemini/code/FSPT

mkdir -p external/tapnextpp checkpoints/tapnextpp logs

LOG="logs/tapnextpp_download_inner_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TAPNext++ download probe started ==="
date -Is

echo "=== Fetch project page ==="
curl -L --retry 5 --retry-delay 10 \
  "https://tap-next-plus-plus.github.io/" \
  -o external/tapnextpp/project_page.html

echo "=== Extract candidate links ==="
python - <<'PY'
import re
from pathlib import Path

html = Path("external/tapnextpp/project_page.html").read_text(errors="ignore")
links = sorted(set(re.findall(r'https?://[^"\'<> )]+', html)))

Path("external/tapnextpp/candidate_links.txt").write_text("\n".join(links))

print("candidate links:")
for x in links:
    if any(k in x.lower() for k in ["github", "huggingface", "drive", "checkpoint", "ckpt", "model", "weights", ".pt", ".pth", ".npz", ".safetensors"]):
        print(x)
PY

echo "=== Try discover GitHub repo ==="
GITHUB_URL="$(grep -Eo 'https://github.com/[^"'"'"' <>)]+' external/tapnextpp/project_page.html | head -1 || true)"

if [ -n "$GITHUB_URL" ]; then
  echo "Found GitHub: $GITHUB_URL"
  rm -rf external/tapnextpp/repo
  git clone --depth 1 "$GITHUB_URL" external/tapnextpp/repo
else
  echo "No GitHub URL found in project page."
fi

echo "=== Save checkpoint/model candidate links ==="
grep -Eo 'https?://[^"'"'"' <>)]+' external/tapnextpp/project_page.html \
  | grep -Ei 'huggingface|drive|ckpt|checkpoint|weights|model|\.pt|\.pth|\.npz|\.safetensors' \
  | sort -u \
  > external/tapnextpp/checkpoint_candidate_links.txt || true

echo "=== Candidate checkpoint links ==="
cat external/tapnextpp/checkpoint_candidate_links.txt || true

echo "=== Done probe. Manual inspection may be required if checkpoint link is JS-rendered or gated. ==="
date -Is
