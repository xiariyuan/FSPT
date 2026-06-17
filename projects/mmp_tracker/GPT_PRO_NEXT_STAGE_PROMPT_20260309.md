Please deeply review the attached MMP tracker code and the newest experiment evidence.

This is not an early brainstorming stage anymore.
We already completed the current internal evidence chain and obtained a fairly strong negative result for the current mainline.

Your job is to act like a research reviewer / co-designer and tell us what the next architecture should be.

Read these first:

1. `projects/mmp_tracker/GPT_PRO_NEXT_STAGE_BRIEF_20260309.md`
2. `projects/mmp_tracker/GPT_PRO_NEXT_STAGE_FILEGUIDE_20260309.md`
3. `projects/mmp_tracker/pro_review_artifacts/20260309/overnight_autopilot_decision.json`

Then inspect the core files and the raw metrics.

The important context is:

- `local` was the matched baseline
- `top1 rematch + nocommit` was tested
- `top1 rematch + deferred commit` was tested
- deferred commit is now a completed negative result at the current design level
- full local baseline was run overnight
- the autopilot explicitly decided **not** to run deferred full training

Please answer the following as directly and concretely as possible:

1. Is the current `top1 rematch + deferred commit` route decisively dead as a mainline?
2. What is the best next architecture?
   - Is it `top-k candidate rematch + candidate scoring / abstain`?
   - Or should we pivot to a different mechanism entirely?
3. What exactly is structurally wrong with the current design?
   - Is the problem primarily the single-candidate assumption?
   - the state-transition mechanism?
   - the training signal?
   - the retrieval representation?
4. What is the strongest remaining paper story?
   - mechanistic long-occlusion paper?
   - stronger retrieval / candidate reasoning paper?
   - or should the whole route be abandoned?
5. What are the top 3 highest-ROI next experiments?
   - please make them concrete and ordered
   - include stop rules
6. If we continue, what should be the new baseline ladder?
   - e.g. local -> top-k retrieval -> rematch -> abstain / selector -> commit
7. If you had to redesign this from the current codebase with minimum wasted effort, what exact files/modules would you keep, what would you rewrite, and what would you delete?

Please be blunt.
Do not protect the current idea if the evidence already says it is weak.

Also please pay attention to the fact that:

- the current route is in the online / forward-only point-tracking setting
- the public SOTA bar is already very high
- we do not want generic advice like "train longer" unless the evidence truly supports that
- we prefer a clean new mainline over repeatedly polishing a failed mechanism

Desired output format:

1. executive verdict
2. what the experiments already prove
3. what they do **not** prove
4. root-cause diagnosis of failure
5. best next architecture
6. exact next 3 experiments with stop rules
7. publication-potential judgment

