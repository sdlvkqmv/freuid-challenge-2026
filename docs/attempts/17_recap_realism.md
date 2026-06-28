# Attempts 17/18 — enhanced recapture-sim realism (chromatic moiré + aberration + vignette)

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

## Hypothesis
Recapture augmentation is the ONLY lever that ever helped (06). 07a showed *wider ranges* hurt, but
*more physically faithful* artifacts were untried. Add the hallmark recapture cues missing from 06's
sim, at 06's moderate strength (`freuid/augment.py`, gated flags):
- **17 / V1** — `chroma_moire`: per-channel **multiplicative** moiré (RGB-subpixel × sensor-grid
  beat, two superimposed frequencies). The single distinctive change vs 06.
- **18 / V2** — V1 + `chroma_ab=2.0` (chromatic aberration) + `vignette=0.25` (lens falloff). Full
  realism bundle.

## Result — both REGRESS hard vs 06
| attempt | added realism | recap-val | spearman vs 06 | **Kaggle public LB** |
|---|---|---|---|---|
| 06 baseline | (moderate generic) | 0.00042 | 1.000 | **0.15185** |
| 17 V1 chroma-moiré | per-ch mult. moiré | 0.00034 | 0.925 | **0.21221** 🔴 |
| 18 V2 full bundle | + aberration + vignette | **0.00030** | 0.942 | **0.20846** 🔴 |

## Takeaway — realism ≠ generalization; the proxy ACTIVELY misleads here
1. **More faithful synthetic artifacts overfit the synthetic domain.** The model latches onto *my
   specific* chromatic-moiré/aberration signature, which differs from real test recaptures → a new
   shortcut that doesn't transfer. 06's simpler, generic perturbation wins precisely because it
   imprints **no learnable synthetic signature** — it broadens the input distribution without
   teaching a fake cue.
2. **recap-val is worse than useless for augmentation changes — it is anti-correlated.** V2 had the
   **best recap-val of every run (0.00030)** yet near-worst LB (0.20846). Because recap-val evaluates
   on the *same synthetic sim*, it rewards exactly the synthetic-sim overfitting that destroys real-LB
   performance. → [[eval_harness]]: do NOT use recap-val to rank aug variants; it measures fit to the
   fake distribution. Only public LB ranks.

**Decision:** recapture-realism is a dead lever — 06's moderate generic aug is optimal and cannot be
improved by adding faithful artifacts. Code kept (gated, default-off in `augment.py`). This is the
**9th consecutive direction to regress vs 06** (07/08/09/10/11/12/13/14/15/16/17/18). The only
remaining genuinely-different family is **non-pixel field-consistency (D)** — independent of the
recapture texture entirely. 06 remains champion.
