#!/usr/bin/env bash
set -euo pipefail

cd /gemini/code/FSPT

mkdir -p external/tapnextpp checkpoints/tapnextpp logs docs

LOG="logs/tapnextpp_acquire_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== TAPNext++ acquisition started ==="
date -Is

echo "=== Existing assets check ==="
ls -lh checkpoints/tapnext/bootstapnext_ckpt.npz 2>/dev/null || true
ls -lh baselines/track_on/checkpoints_trackon2_dinov2.pt 2>/dev/null || true
ls -lh baselines/track_on/checkpoints_trackon2_dinov3.pt 2>/dev/null || true

echo "=== Fetch TAPNext++ project page ==="
curl -L --retry 5 --retry-delay 10 \
  "https://tap-next-plus-plus.github.io/" \
  -o external/tapnextpp/project_page.html

echo "=== Extract links from project page ==="
python3 - <<'PY'
import re
from pathlib import Path

html_path = Path("external/tapnextpp/project_page.html")
html = html_path.read_text(errors="ignore")

links = sorted(set(re.findall(r'https?://[^"\'<> )]+', html)))

Path("external/tapnextpp/candidate_links.txt").write_text(
    "\n".join(links) + ("\n" if links else "")
)

ckpt_keywords = [
    "huggingface", "drive.google", "storage.googleapis",
    "checkpoint", "ckpt", "weights", "model",
    ".pt", ".pth", ".npz", ".safetensors"
]

ckpt_links = [
    x for x in links
    if any(k in x.lower() for k in ckpt_keywords)
]

Path("external/tapnextpp/checkpoint_candidate_links.txt").write_text(
    "\n".join(ckpt_links) + ("\n" if ckpt_links else "")
)

github_links = [x for x in links if "github.com" in x.lower()]
Path("external/tapnextpp/github_candidate_links.txt").write_text(
    "\n".join(github_links) + ("\n" if github_links else "")
)

print("links:", len(links))
print("github links:")
for x in github_links:
    print(" ", x)
print("checkpoint/model links:")
for x in ckpt_links:
    print(" ", x)
PY

echo "=== Clone first GitHub candidate if available ==="
GITHUB_URL="$(head -n 1 external/tapnextpp/github_candidate_links.txt 2>/dev/null || true)"

if [ -n "${GITHUB_URL}" ]; then
  echo "Found GitHub URL: ${GITHUB_URL}"
  rm -rf external/tapnextpp/repo
  git clone --depth 1 "${GITHUB_URL}" external/tapnextpp/repo
else
  echo "No GitHub URL found on project page."
fi

echo "=== Try direct checkpoint download only for obvious file links ==="
python3 - <<'PY'
from pathlib import Path
import subprocess
import shlex

links_path = Path("external/tapnextpp/checkpoint_candidate_links.txt")
if not links_path.exists():
    print("No checkpoint candidate file.")
    raise SystemExit(0)

links = [x.strip() for x in links_path.read_text().splitlines() if x.strip()]
direct = []
for x in links:
    xl = x.lower()
    if any(xl.endswith(ext) for ext in [".pt", ".pth", ".npz", ".safetensors", ".zip", ".tar", ".tar.gz"]):
        direct.append(x)

Path("external/tapnextpp/direct_download_links.txt").write_text(
    "\n".join(direct) + ("\n" if direct else "")
)

for url in direct:
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    if not name:
        continue
    out = Path("checkpoints/tapnextpp") / name
    print("Downloading", url, "->", out)
    subprocess.run([
        "curl", "-L", "--retry", "5", "--retry-delay", "10",
        "-o", str(out), url
    ], check=False)
PY

echo "=== Final tree ==="
find external/tapnextpp -maxdepth 3 -type f -printf "%p %s bytes\n" | sort || true
find checkpoints/tapnextpp -maxdepth 2 -type f -printf "%p %s bytes\n" | sort || true

echo "=== Write acquisition summary ==="
python3 - <<'PY'
from pathlib import Path
import json
import subprocess

def read(path):
    p = Path(path)
    return p.read_text(errors="ignore").splitlines() if p.exists() else []

summary = {
    "project_page_exists": Path("external/tapnextpp/project_page.html").exists(),
    "repo_exists": Path("external/tapnextpp/repo").exists(),
    "github_candidate_links": read("external/tapnextpp/github_candidate_links.txt"),
    "checkpoint_candidate_links": read("external/tapnextpp/checkpoint_candidate_links.txt"),
    "direct_download_links": read("external/tapnextpp/direct_download_links.txt"),
    "checkpoint_files": [
        str(p) for p in Path("checkpoints/tapnextpp").glob("*") if p.is_file()
    ],
}

Path("external/tapnextpp/acquisition_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)

md = []
md.append("# TAPNext++ Acquisition Summary\n")
md.append("\n## Status\n")
if summary["repo_exists"] and summary["checkpoint_files"]:
    md.append("\n```text\ncode_acquired = true\ncheckpoint_acquired = true\nnext = run import/checkpoint-load smoke\n```\n")
elif summary["repo_exists"]:
    md.append("\n```text\ncode_acquired = true\ncheckpoint_acquired = false\nnext = inspect README/model card for checkpoint link\n```\n")
else:
    md.append("\n```text\ncode_acquired = false or unknown\ncheckpoint_acquired = false or unknown\nnext = inspect project_page.html/candidate_links manually\n```\n")

md.append("\n## GitHub candidates\n\n")
for x in summary["github_candidate_links"]:
    md.append(f"- `{x}`\n")

md.append("\n## Checkpoint candidates\n\n")
for x in summary["checkpoint_candidate_links"]:
    md.append(f"- `{x}`\n")

md.append("\n## Downloaded checkpoint files\n\n")
for x in summary["checkpoint_files"]:
    md.append(f"- `{x}`\n")

Path("docs/tapnextpp_acquisition_summary_2026-07-05.md").write_text("".join(md))
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "=== TAPNext++ acquisition finished ==="
date -Is
