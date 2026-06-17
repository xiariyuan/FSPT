# Related Work Positioning Notes

## Safe Core Positioning

Use the following high-level positioning throughout the paper:

This work does not claim to be the first study of long-term occlusion, online point tracking, or point re-identification in general. Instead, it isolates a narrower event-level problem: first-frame causal re-entry after a long invisible interval, studies which signals are useful for that decision, and proposes a lightweight selector that improves a baseline tracker in that regime.

## What We Can Safely Claim

1. Existing point-tracking benchmarks and methods do not usually isolate first-frame causal re-entry as a primary evaluation target.
2. Long-occlusion return and point re-identification are already recognized as difficult by recent benchmark and method papers.
3. Our contribution is a dedicated problem formulation, signal analysis, and lightweight selector for this specific moment in the track.
4. Pre-occlusion support memory is the strongest usable cue among the signals we tested for this re-entry decision.

## What We Should Not Claim

1. Do not claim to be the first work on long-term point tracking under occlusion.
2. Do not claim to be the first work on online point recovery or retracking after occlusion.
3. Do not claim to be the first benchmark containing return-to-view or re-identification events.
4. Do not imply that prior work ignores occlusion entirely.

## How To Differentiate From Specific Papers

### TAP-Vid

Use:
TAP-Vid established the general Tracking-Any-Point task and benchmarked full-trajectory point tracking.

Do not use:
Prior work did not study difficult point-tracking scenarios.

Why:
TAP-Vid is the benchmark foundation, but it is broader than our event-specific setting.

### EgoPoints

Use:
EgoPoints highlights return-to-view and point re-identification challenges in egocentric videos, which supports the importance of our setting.

Do not use:
No prior benchmark considered point return or re-identification.

Why:
EgoPoints already discusses these phenomena explicitly.

### ITTO

Use:
ITTO emphasizes long-range motion, occlusion, and post-occlusion re-identification as a difficult benchmark setting.

Do not use:
We are the first to show that long-term occlusion remains hard.

Why:
ITTO already frames this as a benchmark challenge.

### Track-On

Use:
Track-On is a causal online tracker with explicit memory, close to our setting in spirit.

Do not use:
We are the first to use past visual information to help long-term causal point tracking.

Why:
Track-On already uses memory in an online setting.

### ReTracker

Use:
ReTracker directly addresses robust online point tracking under viewpoint change and long-term occlusion with a stronger end-to-end method.

Do not use:
No prior method tackled online re-entry after long occlusion.

Why:
This is the closest method-level overlap and the easiest paper for a reviewer to use against an overclaim.

## Recommended One-Sentence Summary

Prior work has improved general point tracking, built harder long-range benchmarks, and proposed stronger online trackers, but still lacks a focused, causal treatment of the first visible re-entry event as its own decision problem.

## Recommended Novelty Sentence

Our novelty lies in isolating the first-frame re-entry decision, showing that pre-occlusion support memory is the dominant practical cue for it, and turning that observation into a simple regime-aware selector rather than a new end-to-end tracker.
