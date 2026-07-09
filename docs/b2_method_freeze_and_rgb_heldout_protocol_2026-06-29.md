# B2 Method Freeze and RGB Held-Out Protocol — 2026-06-29

## Decision

We now freeze the method family before any further RGB-Stacking evaluation.

The corrected method framing is:

```text
B2-W: Predicted Re-entry Windowed Override
```

## Method definition

Inputs:

```text
base = standard-strong tracker
reentry_branch / override = re-entry-strong tracker or fusion branch
```

Trigger:

```text
base invisible run >= 1 and override visible
```

Action:

```text
override from t - pre to t + W, then return control to base and continue monitoring
```

Special case:

```text
W = infinity gives full-post B2
```

## Frozen variants

For reporting and held-out evaluation, use predefined variants only:

```text
B2-W16
B2-W32
B2-W64
B2-full
```

The current robust default for RGB held-out evaluation is:

```text
B2-W16
```

No additional rule tuning should be performed on RGB held-out videos.

## Development vs held-out split

The first 10 RGB-Stacking videos have been used for development and diagnosis:

```text
dev: rgb_stacking_000000 ... rgb_stacking_000009
```

Therefore, subsequent cross-dataset evidence must use held-out videos:

```text
heldout-10: rgb_stacking_000010 ... rgb_stacking_000019
heldout-20: rgb_stacking_000010 ... rgb_stacking_000029
heldout-40: rgb_stacking_000010 ... rgb_stacking_000049
```

## Implementation correction

`datasets/tapvid_rgb_stacking.py` now supports:

```text
start_index
num_videos
```

and preserves original video names / sequence indices inside subsets.

This prevents held-out videos 10-19 from being mislabeled as 0-9.

## Current protocol

Next evaluation should be:

```text
1. export CoTracker3 offline on heldout-10
2. export CoTracker3 online on heldout-10
3. run B2-W16 on heldout-10
4. compare offline / online / B2-W16
5. do per-video audit only after scores are produced
```

## Paper-facing correction

RGB-Stacking so far validates the B2 mechanism, not the full DAVIS four-teacher B1 branch. A safe statement is:

```text
On RGB-Stacking, we instantiate the override branch with CoTracker3-online and test whether the B2-W mechanism generalizes to held-out videos.
```

Do not claim:

```text
The full DAVIS four-teacher B1 branch generalizes to RGB-Stacking.
```
