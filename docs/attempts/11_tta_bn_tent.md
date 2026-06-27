# Attempt 11 — test-time adaptation (direction G): AdaBN + Tent

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

## Hypothesis
Root finding #0: train is ~all digital, test emphasizes print-and-capture. The 06 model may
collapse on the recapture domain partly because its BatchNorm running stats are calibrated to the
digital train distribution. **Re-align BN to the unlabeled test distribution — no labels, no
retraining** — the cheapest possible domain-gap fix. Two variants (`freuid/tta.py`, bolt onto 06's
`best.pt`):
- **AdaBN** — reset BN running mean/var, recompute (cumulative avg, momentum=None) over one pass of
  the whole public_test set, then predict with the test-domain stats.
- **Tent** (Wang 2021) — freeze all but BN affine (γ/β); online entropy-minimize predictions per
  test batch (lr 1e-3, 1 step, bs16, autocast fwd).

## Result
| variant | spearman vs 06 | **Kaggle public LB** | vs 06 (0.15185) |
|---|---|---|---|
| AdaBN | 0.990 | **0.15598** | 🔴 slightly worse |
| Tent  | 0.898 | **0.15854** | 🔴 worse |

## Takeaway — test-time BN adaptation does NOT help (marginal regression)
- **AdaBN** barely changed ranking (spearman 0.99) → tiny regression. 06 was trained *with recapture
  augmentation*, so its BN stats are **already broadened toward the recapture domain**; recomputing
  on raw test adds noise rather than correcting a mismatch.
- **Tent** entropy loss was **~0 from the first batch** (0.0019→0.0000) — the model is already
  overconfident on test, so there is nothing to minimize. The few non-trivial batches only injected
  drift → larger ranking change (spearman 0.90) but worse LB. Classic Tent failure mode on a
  saturated model.
- **Decision:** direction G exhausted. Recapture-augmentation at *train* time already does the BN
  re-alignment that test-time adaptation tries to do post-hoc, so there is no slack left to recover.
  Implementation kept (`freuid/tta.py`) — could revisit if a *non*-recapture-trained backbone needs
  domain alignment.
