# Attempt 07 — push the SRM×recapture winner (harder aug + hi-res)

Back to [[SUMMARY]] · proxy [[eval_harness]] · best so far [[attempts/06_effb3_srm_recap]]

## Hypothesis
Finding #1 (SRM×recapture synergy) won at LB 0.15185 with *moderate* recapture aug (prob 0.7).
Push it: (a) stronger/wider aug, (b) higher resolution (recapture artifacts — moiré, raster —
are high-frequency, so more pixels should expose them to the SRM stream).

## Setup
Config `configs/srm_recap_strong.yaml` — recapture prob 0.85 (vs 0.7), wider ranges
(downscale 0.4–1.0, blur 0–1.8, moiré_p 0.4, noise 0–12, jpeg 35–92). Two variants:
- **07a** 384px, bs24, 12ep (GPU2)
- **07b** 448px, bs16, 10ep (GPU3)
Selection by recap_freuid.

## Result
| variant | recap-val | **Kaggle public LB** | vs 06 (0.15185) |
|---|---|---|---|
| 07a srm+recap STRONG/WIDE 384 | 0.00058 | **0.19546** | 🔴 much worse |
| 07b srm+recap 448 hi-res | 0.00032 | **0.16440** | 🔴 worse |

## Takeaway
**Both pushes regressed.** The dominant cause is the **stronger/wider recapture aug**: 07a (same
384px as 06, only aug changed) collapsed to 0.19546 → over-augmentation destroys signal. 07b adds
hi-res but still uses the strong aug, landing 0.16440 (hi-res partly compensates the bad aug but
doesn't recover 06). → **06's moderate prob-0.7 aug is near-optimal; do not crank it.** Resolution
is at best neutral here. recap-val again mis-ranked (07b 0.00032 < 06 0.00042 yet LB worse) —
proxy still untrustworthy ([[eval_harness]]).
