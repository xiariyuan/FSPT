# TAPNext++ Acquisition Summary

## Status: BLOCKED (Network)

```text
code_acquired    = false
checkpoint       = false (partial 66 MB / 2.4 GB)
domestic_mirror  = none confirmed
```

## What was checked

### TAPNext++ project page (`https://tap-next-plus-plus.github.io/`)
28 links extracted:
- **GitHub**: `https://github.com/google-deepmind/tapnet` (contains TAPNext++ code in `tapnet/tapnextpp/`)
- **arXiv**: `https://arxiv.org/abs/2604.10582` (paper)
- **CDN**: jsdelivr, cloudflare, font-awesome (assets only)
- **Author pages**: artemzholus, federicotombari, sebastian-jung, etc.
- **Related projects**: tap-next, tapvid, tapvid3d, bootstap, deepmind-tapir

### What was NOT found
- **No HuggingFace link** on project page or in paper
- **No ModelScope / Gitee / BaiduPan** mirror
- **No direct checkpoint download link** on project page (ckpt hosted on GCS, referenced only in repo README)
- **No gated access** — checkpoint is publicly hosted but on Google Cloud Storage

### Checkpoint location (from TAPNext++ README in repo)
```
https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
```
- Expected size: 2.4 GB
- Downloaded: 66 MB (2.8%) at ~50 KB/s
- Estimated full download time at current speed: ~13 hours

## Network blockers

| Target | Failure |
|--------|---------|
| `github.com` | GnuTLS recv error (-110): TLS connection non-properly terminated |
| `storage.googleapis.com` | Download speed ~50 KB/s (severe throttling) |

No git proxy (`http.proxy`, `https.proxy`) configured on this server.

## Recommended recovery

1. **Fix GitHub access** — configure a GitHub proxy/mirror on this server, then:
   ```
   git clone --depth 1 https://github.com/google-deepmind/tapnet external/tapnextpp/repo
   ```
2. **Download checkpoint** via a machine with better bandwidth:
   ```
   curl -L -o tapnextpp_ckpt.pt https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
   ```
   Then scp/rsync to `checkpoints/tapnextpp/`.

3. **Verify**: `file checkpoints/tapnextpp/tapnextpp_ckpt.pt` → should be a valid PyTorch checkpoint.

## Do NOT do
- Do not download TrackOn2
- Do not re-download BootsTAPNext (already at `checkpoints/tapnext/bootstapnext_ckpt.npz`, 741 MB)
- Do not train V26
- Do not use unofficial ModelScope/gitee checkpoints
- Do not confuse BootsTAPNext (JAX, 741 MB) with TAPNext++ (PyTorch, 2.4 GB)
