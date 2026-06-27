# Attempt 12 — frozen foundation features (direction F): CLIP + DINOv2 ViT-L

Back to [[SUMMARY]] · best so far [[attempts/06_effb3_srm_recap]] · proxy [[eval_harness]]

## Hypothesis
Direction F, the biggest OOD bet. UnivFD precedent: a **frozen CLIP-ViT-L/14 + linear probe**
generalizes to *unseen* generators — analogous to the private-LB objective (2 unseen doc types).
The pretrained prior is the OOD lever; learn only a cheap head on top. Two variants
(`freuid/model.py::FrozenFoundation`, backbone permanently eval + no_grad), both with 06's proven
moderate recapture aug:
- **F1** `vit_large_patch14_clip_224.openai`, **linear** head, CLIP normalization (1025 trainable).
- **F2** `vit_large_patch14_dinov2.lvd142m` @224 (pos-embed interpolated), **MLP** head, ImageNet
  norm (525k trainable).

## Result — KILLED at epoch 2 (clear failure)
In-domain val FREUID (lower better; fine-tuned 06 = **0.00018**):

| variant | ep0 | ep1 | ep2 | trajectory |
|---|---|---|---|---|
| F1 CLIP linear | 0.815 | 0.758 | 0.666 | crawling, nowhere near usable |
| F2 DINOv2 mlp  | 0.392 | 0.409 | 0.402 | **plateaued flat** |

Both **catastrophic** vs every fine-tuned attempt (in-domain 0.0001–0.0003). Not submitted —
a ~random model (≈0.4 in-domain → would map ≳0.4 LB by the convnext data point 0.18→0.354) would
waste a submission.

## Takeaway — frozen semantic features are the WRONG signal family for ID forgery
CLIP/DINOv2 are trained for **semantic content** and deliberately discard low-level texture/noise —
which is **exactly** the forensic signal ID-forgery detection needs (noise residuals, JPEG-grid /
double-compression periodicity, pixel-level splice/recapture traces; research §2,§6). The UnivFD
precedent works because *GAN/diffusion* artifacts are partly **semantic** and survive in CLIP space;
**physical-manipulation + print-and-capture ID forgery is a different, low-level signal family** that
frozen semantic encoders miss. Consistent with research §1 (effb3 ≫ ViT on ID forgery, SIDTD
0.994 vs 0.552): the winning recipe is a **full fine-tune of a CNN** with forensic streams, not a
frozen transformer prior.

**Decision:** drop frozen-foundation linear/MLP probing. If revisiting transformers, would need
(a) **partial fine-tune** of late blocks (not frozen) or (b) **low-level/early-layer** features, not
the pooled semantic embedding — but research already says effb3 full-finetune beats ViT here, so
this is low priority. Next lever = **D field-consistency / MRZ** (orthogonal non-pixel signal,
domain-gap-robust) and ROI crops, per [[research/directions]]. Code kept (`FrozenFoundation`,
configs `frozen_clip.yaml`/`frozen_dinov2.yaml`).
