# Local Eval Harness — FREUID proxy

Back to [[SUMMARY]].

## Metric (exactly as scored)

Positive class = fraud (label 1); score = fraud probability (higher = more fraud).
Decision rule at threshold t: flag fraud if `score >= t`.

```
APCER(t) = P(score <  t | label==1)   # attacks missed      = FNR of positive class
BPCER(t) = P(score >= t | label==0)   # bona-fide flagged   = FPR of positive class
DET curve = APCER vs BPCER, swept over all t. Both in [0,1], lower better.

AuDET         = area under APCER-vs-BPCER curve            (∫ APCER dBPCER)
APCER@1%BPCER = APCER interpolated at BPCER = 0.01
g1 = 1 - AuDET ;  g2 = 1 - APCER@1%BPCER
FREUID = 1 - 2*g1*g2/(g1+g2)            # 1 - harmonic mean of the two "goodness" terms
```

Implemented in `freuid/metrics.py:freuid_score` via `sklearn.metrics.roc_curve`
(fpr→BPCER, 1-tpr→APCER). Verified against the competition's published formula in the guide.

## ⚠️ Calibration is a NO-OP for this metric (key finding)

AuDET and APCER@1%BPCER are computed by **sweeping thresholds over the submitted scores**.
A *monotonic* transform f (isotonic, Platt, temperature, min-max, rank) preserves the order of
every pair of scores → for every threshold t there is a t'=f(t) giving the identical confusion
matrix → the entire DET curve is unchanged → **AuDET, APCER@1%BPCER, and FREUID are invariant.**

Consequence:
- Do **not** spend submissions on calibration to move the LB. It cannot.
- The research doc's "calibration is the bottleneck" is about **fixed-τ=0.5 Pixel-F1** doc-forgery
  benchmarks (DOCFORGE-BENCH), where a single threshold is applied. Different metric family.
- What *does* move FREUID: (a) **ranking quality** (separating fraud from bona-fide — AUC-like),
  and (b) **shape of the score distribution near the 1% BPCER operating point** (the hardest 1%
  of bona-fide false alarms). Hard-negative emphasis / loss design near that slice can help (b).
- Ensembling helps **only because rank-fusion changes the ordering**, not via calibration.

## Validation procedure

`freuid/data.py:make_split` — two schemes (config `val.scheme`):

- **stratified** (default): StratifiedKFold on `(type, label)`, `n_folds=5`, `fold=0`.
  In-distribution estimate. **Optimistic** vs the private OOD split.
- **group_holdout**: hold out entire `type`(s) listed in `val.holdout_types`.
  Directly simulates the private LB's unseen-doc-type setting. Use as the OOD proxy.
  The cross-type collapse (~100%→~50%, research §1) is what this catches.

Selection: checkpoint with **lowest val FREUID** is saved to `checkpoints/best.pt`;
OOF val predictions saved to `oof_val.csv` (for ensembling / error analysis).

## 🔴 Measured: the in-domain proxy is BROKEN (attempt 02)

| | value |
|---|---|
| attempt 02 in-domain stratified val FREUID | 0.00011 |
| attempt 02 Kaggle public LB | **0.18471** |

~1700× gap. Cause = **train/test domain shift**: train is 69,332 digital / **20 recaptured**
(`is_digital=False`), test emphasizes **print-and-capture**. The model overfits the digital domain;
stratified CV (which is ~100% digital on both sides) cannot see the collapse. **Do not rank methods
by stratified val.** Build a recapture/OOD holdout (see [[SUMMARY]] remaining directions) before
trusting any local number again.

## Trusting the proxy

- In-domain `stratified` proxy ≫ optimistic. Treat absolute numbers as upper bounds.
- The honest proxy for the private LB is `group_holdout` (leave-one-type-out). When both are
  available, **rank methods by group_holdout**, not stratified.
- Public Kaggle LB currently scores only the 7,821 available public_test images (the rest of the
  142,818 ids are placeholder-filled and not yet imaged) → public LB is itself a partial signal.

## Submission mechanics

`freuid/predict.py`: predict all available public_test images → map onto `sample_submission.csv`
→ fill missing ids with `predict.fill_value` (0.5). Output `[0,1]` continuous scores.
Ensemble: `freuid/ensemble.py` rank-averages multiple submissions (scale-free, targets ranking).

5 submissions/day cap → submit only distinct, proxy-validated candidates.
