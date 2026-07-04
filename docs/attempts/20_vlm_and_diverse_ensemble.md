# Attempt 20-25 — two genuinely-new paradigms: VLM external-prior (T1) + diverse-backbone ensemble (T2)

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

Session goal (2026-07-02): stop bolt-ons; find a paradigm decorrelated from 06. Creative-thinking pass
isolated the shared hidden constraint of all 10 prior failures (F5-negation): **"the fraud signal must
be learned from the 69k train set"** — which is 99.97% digital, so every learned signal is
digital-domain and collapses on the recapture-heavy test (root finding #0). Two moves negate it.

## T1 — VLM zero-shot forgery score (external prior, NO train-set learning) — FAILED

Idea (F7 adjacent-possible): an open VLM (Qwen2-VL-2B-Instruct) is pretrained on web-scale REAL
photos → recapture/print-and-capture is *in-distribution* for it (unlike our train), and it is a
totally different function from a pixel-forensic CNN → error-decorrelated. Score = P("Yes") vs
P("No") on the first generated token of a forensic yes/no prompt, read from logits (no sampling, no
labels). Code: `freuid/vlm_score.py`.

**Result: AUROC 0.4452 on a 500-image stratified train sample — WORSE than random.** Per type all
≤0.51 (BENIN 0.51, EGYPT 0.44, GUINEA 0.45, MOZAMBIQUE 0.42, **MAURITIUS 0.31**). The 2B VLM cannot
judge these forgeries zero-shot — photo swaps done well + recapture are too subtle, and the model is
biased by document familiarity. A barely-sub-0.5, per-type-inconsistent signal is noise; inverting it
would overfit. **Not submitted** (free diagnostic killed it). The most promising "different paradigm"
does not work at 2B zero-shot. (Untried: 7B/72B VLM, or fine-tuning a VLM — high cost, uncertain, and
the fine-tune reintroduces the digital-domain-learning constraint we were trying to escape.)

## T2 — diverse-backbone ensemble (same winning recipe, different arch) — FAILED (11th direction)

Idea: rank-fusion only helps if members are error-decorrelated (all 10 bolt-ons were >0.9 corr → the
fusion attempt 10 gave 0.15564 ≈ 06). Train NEW backbones on 06's exact recipe (SRM + recapture aug)
to get genuinely independent weights, then fuse. This is the "ensemble of diverse recapture-trained
streams" the SUMMARY had flagged as untried.

Trained 3 backbones (12ep, recap-freuid selection):
- **convnext_tiny.in12k** — DUD: val FREUID flat 0.187-0.193 across 3 epochs (never fit even the easy
  digital domain), same under-convergence as attempt03. Killed. ConvNeXt+6ch-input+this recipe doesn't
  train without a bespoke schedule.
- **resnet50** (attempt21) — converged perfectly (in-domain 0.00000, recap 0.00043 @ ep10).
- **resnext50_32x4d** (attempt22) — converged perfectly (in-domain 0.00000, recap 0.00053 @ ep9).

Correlations of test preds vs 06: r50 spearman **0.896**, rx50 **0.821** (rx50 more decorrelated);
r50↔rx50 **0.946** (same family → near-duplicate, no point fusing both).

Fusion curve (rank-fuse, LB — lower better):

| candidate | 06 weight | spearman vs 06 | LB |
|---|---|---|---|
| resnet50 solo (attempt21) | 0% | 0.896 | 0.21612 |
| 06 + resnet50 equal (attempt23) | 50% | 0.896 | 0.17868 |
| 06 : resnext50 = 2:1 (attempt25) | 67% | 0.821 | 0.17409 |
| 06 : resnext50 = 3:1 (attempt24) | 75% | 0.821 | 0.17064 |
| **06 champion** | 100% | — | **0.15185** |

**Monotonic in the 06 weight — every fusion regresses, asymptoting to 0.15185 from above.** Zero
variance-reduction benefit at the 1%BPCER operating point. Root cause, now airtight: **effb3 is not a
generic backbone — it is specifically OOD-robust on the recapture test domain** (research §1: effb3 ≫
ResNet/ViT on ID forgery). resnet50/resnext50 fit the training (digital) distribution *identically*
(in-domain 0.0000) yet generalize far worse to recapture (solo 0.216 vs 0.152) → fusing them can only
drag 06 toward their error. Held the 5th daily submission: a 5:1+ fuse would land ~0.155-0.16 (still
> 06) — no positive-EV move remained.

## T3 — same-backbone (effb3) reseed ensemble — FAILED, but exposes the real story

T2 fused *different* backbones (worse on recapture → regressed). The cleaner test: fuse **effb3 with
effb3**, same exact 06 recipe, only the seed differs → equal-strength members, decorrelated by init,
the textbook variance-reduction ensemble. Trained two clean reseeds (06's config verbatim, seed 123 &
456, 6ep). Both converged strongly: recap-val **0.00009** each — *better* than 06's 0.00042.

Correlations vs 06: s123 **0.920**, s456 **0.916** (s123↔s456 0.962). Results (LB, lower better):

| candidate | LB |
|---|---|
| **06 seed42 (champion)** | **0.15185** |
| seed456 solo | 0.19178 |
| seed123 solo | 0.21497 |
| 06 + s123 (2-way) | 0.17400 |
| 06 + s123 + s456 (3-way) | 0.18088 |

🔴 **Both reseeds are far worse solo (0.19-0.21) than 06 (0.152) — same recipe, only the seed differs.**
So 06 is not a reproducible recipe outcome; it is a **fortunate seed** at the low end of a wide LB
seed-variance band (~0.152-0.215). The ensembles regress for the exact T2 reason (members individually
much worse), now with backbone held fixed. Two hard consequences:
1. **We have NO way to select the good seed.** recap-val ranked the reseeds *better* than 06 (0.00009 <
   0.00042) yet they scored *worse* on LB — the anti-correlation ([[eval_harness]]) reconfirmed at the
   seed level. In-domain is 0.0000 for all (useless). Only the public LB reveals the good seed, one
   submission at a time.
2. **06's 0.15185 is partly public-LB luck.** On the private LB (142k imgs, +2 unseen types) the lucky
   seed may regress toward the band mean. → For the FINAL submission, an effb3 reseed *ensemble* (lower
   variance) may be MORE robust on private even though it is worse on public — a genuine
   selection-time trade-off to revisit before the 2026-07-14 deadline (competition allows picking final
   subs; can't measure private now).

## Takeaway
Three independent, genuinely-different paradigms all fail:
- **External-prior (VLM)** — the prior is too weak zero-shot at 2B; the forgeries need forensic pixels,
  not world knowledge. Same lesson as F frozen-CLIP (semantics ≠ forensics), from the opposite end.
- **Diverse ensemble** — decorrelation is real (down to 0.82) but useless because the alternative
  backbones are *strictly worse on the test domain*; ensembling a worse+correlated member never helps.
  This is the sharpest statement yet of why 06 wins: **the backbone itself (effb3) carries the
  OOD-robustness**, not just the SRM stream or the recapture aug.

- **Reseed ensemble** — the recipe has huge LB seed-variance and 06 is the lucky low end; no proxy
  selects the good seed, and reseeds are individually too weak to fuse without regressing.

**12 CONSECUTIVE DIRECTIONS DO NOT BEAT 06.** The signal accessible from the labeled train set is
captured; the VLM shows the accessible *external* zero-shot prior is too weak; and the recipe's own seed
variance means 06 cannot even be *reproduced* on demand. A real gain likely needs either (a) a stronger
VLM fine-tuned on genuine recapture data, or (b) genuinely real recapture training data — both are
new-data / big-build efforts, not a same-day bolt-on. **Accept 06 (0.15185) as the standing final public
submission**, but revisit an effb3-reseed *ensemble* as a lower-variance hedge for the PRIVATE LB before
the deadline (see T3.2 above).

Env note: facenet-pytorch (installed for D) pins `torch<2.3`, which had silently downgraded torch to
2.2.2 and broke training (`torch.amp.GradScaler` missing). Restored torch 2.4.1 + torchvision 0.19.1
(cu121) for this session; facenet-pytorch now shows a harmless version-conflict warning (D is done, not
needed for training/VLM). Re-pin torch<2.3 only if re-running direction D.
