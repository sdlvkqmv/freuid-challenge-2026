# Attempts 15/16 — multi-crop max-aggregation TTA (direction E / ROI-lite)

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

## Hypothesis
Tampered regions (physical manipulation, GenAI face/field edits) are spatially tiny; one
global-pooled forward dilutes a small high-evidence region. Score several spatial crops, aggregate
by **max** (or top-k mean) so a crop landing on the tampered region keeps its high score. Detector-
free first test of the spatial-focus hypothesis before building a real face/text ROI pipeline
(`freuid/multicrop.py`, bolt onto 06's `best.pt`). Crops = full-resize global view + a grid of tiles
near 06's RRC 0.7–1.0 scale.

## Result — both REGRESS vs 06
| attempt | crops | agg | spearman vs 06 | **Kaggle public LB** |
|---|---|---|---|---|
| 06 baseline | 1 (global) | — | 1.000 | 0.15185 |
| 15 multicrop max | global + 2×2 @0.7 | max | 0.934 | **0.18077** 🔴 |
| 16 multicrop topk | global + 3×3 @0.6 | top3-mean | 0.874 | **0.21898** 🔴 |

## Takeaway — spatial cropping discards the dominant (global recapture) cue
Spatial focus **hurts**, and the more the score relies on crops (finer grid 3×3 + top-k), the worse
(0.219). Two mechanisms: (1) the public test is **recapture-dominated** and recapture is a *global*
artifact (whole-image moiré/halftone/double-JPEG) — cropping throws away exactly the cue 06 keys on;
(2) **max over crops inflates bona-fide scores too**, raising the 1%BPCER threshold → worse APCER.
The localized-tamper benefit never materializes (localized families are a minority of public test,
and sub-tiles push 06 out of its training scale → noisy per-crop scores that max amplifies).

**Decision:** detector-free spatial focus is dead. Note this only refutes *replacing* the global
score with crop-max — an **additive face-ROI stream that keeps the global context** (research §6) is
a different design, still open. But the evidence that the global recapture cue dominates lowers its
priority. Code kept (`freuid/multicrop.py`). 06 still champion across G, F, I, E.
