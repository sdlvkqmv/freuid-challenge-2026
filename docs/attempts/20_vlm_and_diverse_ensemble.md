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

## Takeaway
Two independent, genuinely-different paradigms both fail:
- **External-prior (VLM)** — the prior is too weak zero-shot at 2B; the forgeries need forensic pixels,
  not world knowledge. Same lesson as F frozen-CLIP (semantics ≠ forensics), from the opposite end.
- **Diverse ensemble** — decorrelation is real (down to 0.82) but useless because the alternative
  backbones are *strictly worse on the test domain*; ensembling a worse+correlated member never helps.
  This is the sharpest statement yet of why 06 wins: **the backbone itself (effb3) carries the
  OOD-robustness**, not just the SRM stream or the recapture aug.

**11 CONSECUTIVE DIRECTIONS DO NOT BEAT 06.** The signal accessible from the labeled train set is
captured; the VLM shows the accessible *external* zero-shot prior is too weak. A real gain likely needs
either (a) a stronger VLM fine-tuned on genuine recapture data, or (b) genuinely real recapture training
data — both are new-data / big-build efforts, not a same-day bolt-on. **Accept 06 (0.15185) as the
standing final submission** absent one of those.

Env note: facenet-pytorch (installed for D) pins `torch<2.3`, which had silently downgraded torch to
2.2.2 and broke training (`torch.amp.GradScaler` missing). Restored torch 2.4.1 + torchvision 0.19.1
(cu121) for this session; facenet-pytorch now shows a harmless version-conflict warning (D is done, not
needed for training/VLM). Re-pin torch<2.3 only if re-running direction D.
