# Attempt 19 — direction D: photo-identity (main↔ghost) consistency

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

## Hypothesis & recon
Inspecting images revealed the dominant fraud family = **photo substitution**: the main portrait
is swapped to a different person while the secondary "ghost" portrait + stated gender stay → the two
on-card faces no longer match (e.g. a Mauritius ID with a *woman* main photo but a *man* ghost and
Gender=M). This identity mismatch is **semantic** (orthogonal to SRM pixels) and should **survive the
recapture domain gap**. Built a face-consistency feature (`freuid/face_consistency.py`): MTCNN detect
+ InceptionResnetV1/vggface2 embed the two largest faces, `face_incons = 1 - cos(main, ghost)`.

## Signal IS real (validated standalone)
Measured on a 6,000-image stratified train sample:
| type | 2-face coverage | AUROC(face_incons) |
|---|---|---|
| **MAURITIUS/ID** | **94.8%** | **0.8835** |
| EGYPT/DL | 6.9% | 0.65 (faint ghost rarely detected) |
| BENIN/GUINEA/MOZAMBIQUE DL | <1% | n/a (single photo) |

So the signal is strong and reliable **only on MAURITIUS/ID** (dual clear photos); other types are
single-photo or the ghost is too faint for MTCNN (lowering thresholds added false faces). Test-set
coverage 1,695/7,821 (21.7%).

## Fusion REGRESSES, and the diagnostic shows WHY: D is redundant with 06
Late rank-fused into 06 (`freuid/fuse_face.py`, covered images get `(1-w)·rank06 + w·face_pct`),
w=0.4 → **LB 0.20334** (06 = 0.15185). 🔴 Regresses. Free diagnostic on the 1,695 covered test images:
- **pearson(06 score, face_incons) = 0.658**, spearman 0.431 — 06 already tracks the swap signal.
- 222 likely-swaps (incons>0.7): **06 mean score 0.883** → 06 already flags them as fraud.
- 1,191 consistent faces (incons<0.2): 06 mean 0.068 → already bona-fide.
- Swaps 06 *missed* (incons>0.7 yet 06<0.5): only **26/222 (12%)**.

**06's SRM pixel-forensic stream already captures photo substitution** — the swapped/blended face
leaves residual artifacts 06 keys on, so the "orthogonal" semantic signal is ~65% correlated with it.
Naive fusion injects the face signal's ~12% error across all 1,695 covered images (raising some true
bona-fide, corrupting the 1%BPCER operating point) to recover only 26 genuinely-missed swaps → net
negative.

## Takeaway
Direction D's strongest realization is a **validated but redundant** signal: it works (AUROC 0.88 on
Mauritius) yet adds nothing over 06 because 06 already detects the same fraud family. This is the
clearest explanation yet for 06's robustness — **its forensic stream is more semantically complete
than expected** (it "sees" photo swaps without an explicit identity check). Ceiling for any smarter
fusion is ~26 images → not worth more submissions. Code kept (`face_consistency.py`, `fuse_face.py`).
A confident-only gated boost (only push incons>0.8 swaps up, never move others) is the one cheap
follow-up if revisited, but the upside is marginal. 06 remains champion (10th direction to not beat it).

**Untried D sub-signal:** OCR date-logic (issue≤expiry, DOB plausibility) — universal across types but
needs OCR infra and likely also redundant (field-value edits leave pixel traces too). Low priority.
