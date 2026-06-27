# Attempts 13/14 — operating-point hard-negative mining (direction I)

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

## Hypothesis
The FREUID metric is driven by **APCER@1%BPCER**: threshold set by the top ~1% of bona-fide scores,
APCER = fraction of fraud below it. So the decisive samples are **hard negatives** (high-scoring
bona-fide that raise the threshold) and **hard positives** (low-scoring fraud that get missed).
Plain BCE (06) spends most gradient on easy examples. Concentrate it on the hard tail → squeeze the
operating point. Two mechanisms on 06's exact recipe (effb3+SRM, moderate recapture aug, `freuid/losses.py`):
- **13 / I-focal** — asymmetric focal loss, `gamma_pos=2, gamma_neg=3` (focus hard bona-fide harder,
  since they set the 1%BPCER threshold). `configs/hardneg_focal.yaml`, 8ep GPU6.
- **14 / I-OHEM** — plain BCE+pos_weight but backprop only the **hardest 50%** of each batch.
  `configs/hardneg_ohem.yaml`, 8ep GPU7.

## Result — both REGRESS vs 06
| attempt | recap-val | spearman vs 06 | **Kaggle public LB** | vs 06 (0.15185) |
|---|---|---|---|---|
| 06 baseline | 0.00042 | 1.000 | 0.15185 | — |
| 13 I-focal (gp2/gn3) | 0.00033 | 0.900 | **0.16161** | 🔴 worse |
| 14 I-OHEM (top50%) | 0.00032 | 0.937 | **0.18855** | 🔴 much worse |

## Takeaway — hard-neg mining AMPLIFIES the domain shortcut (proxy lied again)
Both variants had **better recap-val than 06** (0.00032/0.00033 < 0.00042) yet **worse LB** — the
proxy mis-ranked them, exactly the [[eval_harness]] pattern. Mechanism (root finding #0): the hard
examples live in the **digital train domain**; concentrating gradient on them sharpens the decision
boundary *on the digital shortcut*, which does not transfer to the recapture-heavy test and likely
**overfits digital-specific artifacts** → the train/test gap widens. OHEM (most aggressive, top-50%)
regressed most (0.18855). Operating-point squeezing presumes a well-ranked model that just needs a
sharper threshold; here ranking quality is gated by the **domain gap**, not the operating point, so
hard-mining the wrong domain backfires.

**Decision:** drop hard-negative mining. Metric-squeeze tricks don't help while the domain gap
dominates. Code kept (`freuid/losses.py`, focal + OHEM). Last lever left = a **new signal family**
(ROI face/text crops + max-agg, or field-consistency) that is itself less domain-gap-sensitive —
NOT another reshaping of the same digital-domain loss. 06 still champion across directions G, F, I.
