from pathlib import Path
p=Path('docs/REENTRY_TAP_PAPER_DRAFT_V1.md')
s=p.read_text()

# Update abstract: insert ReEntry-Guard after B2 sentence if absent.
old = "These results show that selective local override is an effective and lightweight way to improve TAP re-entry recovery without sacrificing standard tracking."
new = "In addition, we explore **ReEntry-Guard**, an oracle-guided runtime reliability gate trained only on dev stress labels. When frozen on RGB fresh20-49, ReEntry-Guard further improves over B2-W16-P2 on natural, translate, and occluder validation, closing 20--32% of the oracle routing gap while preserving or improving AJ. These results show that selective local override is an effective and extensible way to improve TAP re-entry recovery without sacrificing standard tracking."
if old in s and 'oracle-guided runtime reliability gate' not in s[:3000]:
    s=s.replace(old,new,1)

oldc = "4. **Controlled stress validation.** We introduce ReEntry-TAP, a controlled validation protocol for out-of-frame and occlusion-induced re-entry, and show that B2-W16-P2 generalizes to frozen RGB fresh20-49 stress with consistent AJ_RD gains."
newc = "4. **Controlled stress validation.** We introduce ReEntry-TAP, a controlled validation protocol for out-of-frame and occlusion-induced re-entry, and show that B2-W16-P2 generalizes to frozen RGB fresh20-49 stress with consistent AJ_RD gains.\n\n5. **Learned reliability extension.** We further train ReEntry-Guard, a lightweight runtime gate supervised by oracle routing labels on dev stress. ReEntry-Guard improves beyond the fixed B2-W16-P2 rule on all three fresh evaluations and closes 20--32% of the oracle_b2 AJ_RD gap."
if oldc in s and 'Learned reliability extension' not in s:
    s=s.replace(oldc,newc,1)

# Insert method subsection before design properties.
marker = "### 4.4 Design Properties\n"
insert = """### 4.4 ReEntry-Guard: Learned Reliability Extension\n\nB2-W16-P2 is a training-free rule: once the trigger fires, the local override window is accepted. Oracle analysis shows that this fixed rule is strong but not optimal. For some triggered re-entry queries, the override improves AJ_RD; for others, preserving the base branch is better. This motivates an optional learned reliability gate, **ReEntry-Guard**.\n\nReEntry-Guard keeps the same candidate local override windows as B2-W16-P2, but adds an accept/reject decision for eligible re-entry candidates. The gate is trained only on dev stress data using oracle routing labels:\n\n```text\nlabel = 1 if B2 candidate AJ_RD_256 > offline AJ_RD_256 for the query\nlabel = 0 otherwise\n```\n\nAt test time, the gate uses only runtime features available from the two predicted branches, including visibility disagreement, base invisible-run length, override visible persistence, base/override geometry distance, local motion smoothness, and trigger timing. No ground-truth features are used at inference.\n\nThus the full framework has two instantiations:\n\n```text\nB2-W16-P2:\n  training-free fixed local override\n\nReEntry-Guard:\n  oracle-guided learned reliability gate over the same local override candidates\n```\n\nIn our experiments, ReEntry-Guard is trained on RGB dev0-9 translate/occluder L16 stress and then frozen for RGB fresh20-49 natural and frozen stress validation.\n\n"""
if marker in s and '### 4.4 ReEntry-Guard' not in s:
    s=s.replace(marker, insert+marker,1)
    # Renumber design properties? Leave 4.4 and ReEntry-Guard 4.4 duplicate? Let's fix.
    s=s.replace('### 4.4 Design Properties','### 4.5 Design Properties',1)

# Insert experiment subsection after oracle upper-bound block before Analysis.
marker = "## 7. Analysis\n"
exp = """### 6.10 ReEntry-Guard Learned Reliability Gate\n\nWe train ReEntry-Guard on dev translate_L16 and dev occluder_L16 using oracle accept/reject labels. The training set contains 7,659 eligible re-entry samples, with 3,712 positive and 3,947 negative examples. We evaluate the frozen gate on three fresh20-49 settings: natural RGB fresh20-49, frozen translate_L16, and frozen occluder_L16.\n\nThe best v2 model is a random forest gate with threshold 0.40.\n\n| Setting | B2 AJ_RD | B2 AJ | ReEntry-Guard AJ_RD | ReEntry-Guard AJ | ΔAJ_RD vs B2 | ΔAJ vs B2 |\n|---|---:|---:|---:|---:|---:|---:|\n| RGB fresh20-49 natural | 0.4454 | 79.0664 | 0.4499 | 79.1749 | +0.0045 | +0.1085 |\n| fresh20-49 translate L16 | 0.5310 | 74.8457 | 0.5336 | 74.8490 | +0.0026 | +0.0033 |\n| fresh20-49 occluder L16 | 0.6588 | 76.7991 | 0.6632 | 77.0761 | +0.0044 | +0.2770 |\n\nVideo-level statistics show that the gain is small but consistent. On natural RGB fresh20-49, ReEntry-Guard improves AJ_RD over B2 on 19/30 videos with mean video-level gain +0.0045 and 95% CI [0.0006, 0.0091]. On frozen occluder L16, it improves AJ_RD on 28/30 videos with mean gain +0.0043 and 95% CI [0.0032, 0.0057], while also improving AJ by +0.277 on average.\n\nCompared with the oracle_b2 upper bound, ReEntry-Guard closes about 31.5% of the natural AJ_RD gap, 20.5% of the translate stress gap, and 30.6% of the occluder stress gap. This confirms that runtime reliability gating can improve beyond the fixed B2-W16-P2 rule, although the remaining oracle gap suggests room for stronger tracker-native uncertainty features.\n\n"""
if marker in s and '### 6.10 ReEntry-Guard' not in s:
    s=s.replace(marker, exp+marker,1)

# Update analysis section after Headroom and Future Learned Gates maybe add gate interpretation.
old = "However, prior generic appearance-feature pilots did not meaningfully improve harmful override detection, suggesting that future gates may need tracker-specific correspondence features or better temporal uncertainty modeling."
new = "ReEntry-Guard v2 validates this direction: a lightweight random-forest gate trained on dev stress improves over B2-W16-P2 on all three fresh evaluations and closes 20--32% of the oracle_b2 gap. However, the gains remain modest. Prior generic appearance-feature pilots also did not meaningfully improve harmful override detection, suggesting that future gates may need tracker-specific correspondence uncertainty, dense correspondence confidence, or better temporal uncertainty modeling."
if old in s:
    s=s.replace(old,new,1)

# Update limitations heuristic trigger.
old = "3. **Heuristic trigger.** The trigger is simple and training-free. Oracle analysis shows that better gating could further improve AJ_RD and AJ."
new = "3. **Heuristic trigger and modest learned-gate gains.** The fixed B2 trigger is simple and training-free. ReEntry-Guard shows that learned reliability gating can improve beyond B2, but the gains are modest and do not close the full oracle gap."
if old in s:
    s=s.replace(old,new,1)

# Add artifact refs
old = "docs/reentry_tap_method_ablation_statistics_oracle_2026-07-01.md\n"
new = "docs/reentry_tap_method_ablation_statistics_oracle_2026-07-01.md\ndocs/reentry_guard_v1_v2_results_2026-07-01.md\n"
if old in s and 'docs/reentry_guard_v1_v2_results_2026-07-01.md' not in s:
    s=s.replace(old,new,1)

p.write_text(s)
print('updated', p, 'len', len(s))
