# Attempt 28 — large-backbone scale pivot (convnext_large)

Back to [[SUMMARY]].

## Hypothesis

effb3 (12M) plateaued at LB 0.15185 after 11 bolt-on directions; public LB top ~0.0004
(~390× gap). Hypothesis: **effb3 capacity caps recapture-domain generalization** → scale the
backbone, hold the winning attempt06 signal recipe (SRM + moderate recapture aug) fixed so the
backbone is the only variable.

## Setup

- **convnext_large.fb_in22k_ft_in1k** (196M, LayerNorm — robust to the small batch 10GB forces).
- attempt06 recipe UNCHANGED: streams `[rgb, srm]`, recapture aug `prob 0.7` (moderate optimum).
- 384px, bs16, AdamW lr 8e-5, wd 0.05, cosine+warmup, 8ep (early-stop patience 3).
- `model.grad_checkpointing` added (timm `set_grad_checkpointing`) → 196M fits 10GB (peak ~6GB).
- Selection by recap_freuid. Best @epoch4.

## Result — FAILED (LB 0.23531, worse than 06's 0.15185)

| | value |
|---|---|
| in-domain val FREUID | **0.00026** (≈ 06's 0.00018 — digital domain fit near-perfect) |
| recap-val FREUID | 0.00069 |
| **Kaggle public LB** | **0.23531** |

**16× more params, near-identical in-domain val, yet OOD LB is the WORST large-model result
(worse than 06 0.152, effb3 seeds 0.19–0.21, resnet/resnext 0.21–0.22).**

## Interpretation — the scale hypothesis is REFUTED (capacity↔OOD inversion)

Bigger backbone → more capacity to memorize the **digital-domain shortcut** (train is 69,332
digital / 20 recaptured) → **worse** collapse on the recapture-heavy test. effb3's smaller
capacity acts as an implicit regularizer against the shortcut. This is the strongest confirmation
yet of root finding #0 + attempts 20–25: **the backbone itself carries OOD-robustness, and
scaling capacity on the same (digital) training distribution makes generalization worse, not
better.** In-domain val being identical (0.00026) while LB diverges by 90× re-proves the proxy
is blind to the domain gap.

## Consequence

Architecture / capacity lever is now fully exhausted (small effb3 = best; larger = monotonically
worse). The remaining lever is **DATA**: the gap to LB-top is a training-distribution gap
(digital→recapture), not a model-capacity gap. Next = real/large recapture training data
(external allowed, guide §External Data) or a recapture-domain corpus, NOT more scale.
→ champion remains **attempt 06 (LB 0.15185)**.
