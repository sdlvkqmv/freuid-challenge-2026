# Forged ID Document Detection — Methodology Survey

> Deep-research harvest for **FREUID Challenge 2026** (IJCAI-ECAI). Binary detection of forged/manipulated identity documents; lower competition score is better; metric = harmonic mean of **AuDET** and **APCER@1%BPCER**.

## Verification status (read first)

- Source: a fan-out deep-research run (6 angles → 18 sources fetched → 132 candidate claims → adversarial verification).
- The 3-vote adversarial verify pass was **stopped early**: ~116 of ~396 verdicts completed; **20 votes refuted**. Claim↔verdict mapping was lost when the run was halted, so claims below are **sourced but NOT all individually triple-verified**.
- **Action:** treat every specific number as "re-confirm against the cited source before relying on it in a writeup." Architecture/approach-level findings (what works, what fails) are higher-confidence than exact metric digits.
- Several arXiv IDs are recent (2025-08 .. 2026-05); plausible given today (2026-06) but unverified for existence. Flagged inline.

---

## TL;DR — what matters for FREUID 2026

1. **Field localization first.** Crop face + text ROIs (YOLO/detector) before classifying. Whole-image classification leaves large accuracy on the table on ID forgeries. [FLiD, EdgeDoc, TwoHead-SwinFPN]
2. **Noise-residual / camera-fingerprint streams beat raw RGB** for tamper cues. NoisePrint(++), guided-filter residuals, DCT/JPEG-artifact streams all add signal. [EdgeDoc, TruFor, CAT-Net, guided-noise]
3. **Generalization gap is THE problem.** Models hit ~100% in-domain and collapse to ~50% cross-type / cross-nationality / cross-generator. The private test will punish overfit. Validate on held-out doc types & `is_digital` splits.
4. **Calibration, not capacity, is the bottleneck** for the metric. AUC can be high while thresholded F1 is near-zero because tampered regions are tiny. Output well-calibrated continuous scores; tune the operating point. [DOCFORGE-BENCH]
   - ⚠️ **Correction (verified on FREUID metric):** this applies to **fixed-0.5-threshold F1** benchmarks, NOT to FREUID's DET-based AuDET/APCER@1%BPCER. Those sweep thresholds over submitted scores, so any **global monotonic** calibration is a **no-op**. The lever is ranking quality + behavior near the 1%-BPCER point. See `SUMMARY.md` finding #2 and `directions.md`.
5. **GAN-era and face-forgery detectors do NOT transfer to diffusion edits.** If the forgery family includes GenAI multimodal edits, use diffusion-aware detectors (reconstruction error / frequency). [DIRE, FIRE]
6. **Ensemble/score-fusion of complementary detectors** (e.g. edge/noise model + transformer forensics model) beat either alone on the private split. [EdgeDoc+TruFor fusion]

---

## 1. Forged ID-document detection — SOTA, datasets, methods

### Key systems
- **FLiD** (frozen-backbone + ROI) — `arXiv:2605.09089` *(recent, unverified existence)*
  - Frozen **MobileNetV3-Small** encoder (2.5M params) → text-forgery AUC 0.954 on FantasyID; **fine-tuning the backbone collapses it to ~0.507** (catastrophic — freeze the encoder).
  - YOLOv8 ROI localization of face/text before classification is worth **~0.39 AUC** on text forgery vs whole-image.
  - EER: text 11.61% / face-swap 18.05% / combined 15.16% vs ~47–50% for a from-scratch full-document MobileNetV2 baseline.
  - Classification head only 191K params / 119M FLOPs — tiny.
- **EdgeDoc** (ICCV 2025 DeepID challenge) — `arXiv:2508.16284`
  - Input = **NoisePrint camera-fingerprint map + green channel** (2-channel), not full RGB.
  - **EdgeNeXt-XXS** backbone + U-Net decoder (depthwise-separable). AdamW (wd 5e-4), cosine over 20 epochs, mask-loss λ=3.0.
  - Perfect on FantasyID public val (AUC 1.00) but **0.59 aggregate F1 on private test** — classic overfit. **Score-fusion EdgeDoc+TruFor → 0.79 F1**, beating either alone (0.59 / 0.71).
- **TwoHead-SwinFPN** — `arXiv:2601.12895` *(recent, unverified existence)*
  - **Swin Transformer + FPN + CBAM + UNet decoder**, uncertainty-weighted multi-task (detect + localize). 84.31% acc / 90.78% AUC on FantasyIDiap.
  - Ablation: **swapping ResNet-50 → Swin is the single biggest gain** (+3.3 acc). FPN/CBAM/multitask add smaller increments.
  - **Optimal classification threshold 0.80, not 0.50.** Segmentation threshold 0.10.
  - Failure modes: false-negatives = subtle text inpainting / high-quality face swaps / compression; false-positives = JPEG artifacts (43%), scanner distortion, natural variation → **detector conflates acquisition artifacts with manipulation**.
- **SIDTD-benchmarked models** — `arXiv:2401.01858`
  - **EfficientNet-B3 crushes ViT-L/16** on template-based ID forgery: 0.994 acc / 1.000 ROC-AUC vs ViT 0.552 / 0.501 (near-random). ViT only competitive on video-based variant.
  - **Few-shot cross-nationality collapse:** EfficientNet 0.994 → 0.593 when testing on held-out nationalities. Every model collapses.

### Datasets (for pre-training / synthetic forgery generation)
| Dataset | Scale | Notes |
|---|---|---|
| **IDNet** `2408.01690` | 837,060 synthetic ID images, 20 doc types, 6 fraud patterns | Generation <\$0.0001/doc, 0.14s/doc, \$69 total. SSIM-to-real up to 0.99. **IDNet→SIDTD transfer 100%; reverse only ~67%** → IDNet trains more general models. |
| **MIDV-2020** `2107.00396` | 72,409 imgs, 1,000 mock docs, 10 types | Bona-fide source. Fine-grained GT (face oval, signature, text fields, MRZ). 2160×3840 / 2480×3507. Tesseract OCR weak (MRZ 10.5%). |
| **SIDTD** `2401.01858` | 1,222 forged templates, 10 EU nationalities | Physical print-and-capture via HP LaserJet + 100µm laminate; "Crop & Replace" composite forgeries. Severe class imbalance in video clips. |
| **FantasyID / FantasyIDiap** | 262–362 designs, ~786–1,086 bona-fide | Forgeries: InSwapper/FaceDancer (face-swap) + DiffSTE/TextDiffuser2 (AI text). Small-scale. |

**Cross-type morphing collapse** (`2408.01690`): MixFaceNet/PW-MAD/Inception morphing detectors hit 97–100% same-type, **drop to 50–55% cross-type** (e.g. WV→AZ driver license). Domain shift is brutal for morphing.

---

## 2. General image manipulation localization (transferable tamper cues)

- **IML-ViT** `2307.14863` — ViT, **1024×1024 input, patch 16**, edge-supervision. CASIA-v1 F1 0.721 (vs MVSS-Net++ 0.513).
  - Ablation lessons (each independently necessary): **MAE pre-training is critical** (remove → mean F1 0.56 → 0.07, total failure); high-res matters (0.56 → 0.42); **manipulation-edge supervision** matters (0.56 → 0.47).
  - Cost: 91M params, 445 GFLOPs, 0.094s/img; train 12h on 4×3090, 22GB/card, 200 epochs.
- **TruFor** `2212.10957` — **Noiseprint++** (DnCNN 15-layer, self-supervised contrastive on 24,757 pristine imgs / 1,475 cameras, NO manipulated data) + **SegFormer** fusion. 68.7M params (lean vs CAT-Net 114M, MVSS 147M). Image-AUC 0.857 avg over 7 datasets. Detects generative forgeries too (CocoGlide F1 0.523). Strong, generalizes via camera-fingerprint inconsistency.
- **PSCC-Net** `2103.10596` — progressive spatio-channel correlation, **3.6M params, 50+ FPS** (8× faster than ManTra-Net/SPAN). NIST16 localization AUC 99.1%/F1 74.2% fine-tuned. Robust under JPEG q=50. Trained on ~377K synthetic (splice/copy-move/removal/pristine).
- **CAT-Net (v2)** `IEEE 9423390 / 2108.12947` — **dual-stream RGB + DCT**; DCT stream pre-trained on double-JPEG detection. Exploits JPEG compression-history mismatch between spliced and authentic regions. Its 5-dataset balanced "CAT protocol" (tampCOCO, compRAISE, FantasticReality, CASIA-v2, IMD2020) is a de-facto standard training setup.
- **ManTra-Net** `CVPR 2019` — self-supervised over **385 manipulation types**; VGG trace extractor + LSTM **local-anomaly (Z-score)** head; fully-conv, arbitrary size, no pre/post-proc. Generalizes via "local manipulation inconsistency" assumption. (Note: released weights have 32 filters in block 1 vs 16 in paper — reproducibility gap.)
- **Guided-noise residual** `2412.01622` — guided-filter + Sobel noise extractor beats BayarConv/SRM by 3–4% AUC; ARPM dilated pyramid {6,12,18} adds multi-scale. Robust to Gaussian blur k=15. Strong on **small forged regions (<1% area)** — relevant to ID field edits.

**Transferable takeaways for ID fraud:** noise residuals (NoisePrint/guided/SRM) + DCT/JPEG streams + edge supervision + high-res input. Manipulation borders and tiny-region sensitivity matter because ID forgeries edit small fields (photo, DOB, name).

---

## 3. GenAI / diffusion / deepfake edit detection

- **GAN-trained detectors fail on diffusion** (`2303.09295`): CNNDetection/GANDetection/Patchforensics ~51–58% acc (≈chance) on diffusion images. **Face-forgery detectors (SBI) also fail (~59%).** Do not reuse deepfake-face detectors blindly.
- **DIRE** `2303.09295` — **DIffusion Reconstruction Error**: reconstruct image via pretrained diffusion (DDIM inversion, S=20), pixel error = signal. 99.9% acc in-domain; generalizes to unseen diffusion models (PNDM 98.7%, iDDPM 98.6%) and cross-dataset (LSUN→ImageNet SD-v1 97.2%). Robust to JPEG q=30, blur σ=3.
- **FIRE** `2412.07140` *(recent)* — **F**requency-guided reconstruction error. Hypothesis: diffusion models mis-reconstruct **mid-band frequencies**. Uses only LDM encoder/decoder (skips full denoising → fast, end-to-end). Beats FakeInversion on DALL-E 3 (78.3 vs 74.1 AUC) and Kandinsky-3 (+7.6 AUC). Learned frequency mask matters (57→78 AUC across masks). **Weak under severe Gaussian blur** — known attack vector.

**For FREUID's GenAI multimodal-edit family:** consider a diffusion-reconstruction-error or frequency branch in addition to the noise/RGB forensic streams.

---

## 4. Print-and-capture / presentation-attack detection (the `is_digital=False` family)

- FREUID explicitly includes **print-and-capture forgeries to "close the analog hole"** most detectors rely on. (`freuid2026.microblink.com`)
- **PRNU / sensor-noise** localization works but needs many frames per camera model — impractical per-image. (`1910.08993`)
- **SIDTD** provides a reproducible physical PAI recipe (HP LaserJet E65050 + 100µm laminate) — useful to understand the print-recapture artifact distribution.
- Acquisition artifacts (JPEG, scanner distortion, moiré) are the **dominant false-positive source** (TwoHead-SwinFPN: 43% of FPs from JPEG). A print-and-capture detector must separate *capture* artifacts from *manipulation* artifacts — augment heavily with benign recapture/compression so the model doesn't learn "any artifact = fraud."
- Classic survey (`1910.08993`): SVM dominates legacy work; texture (SIFT-FV+SVM F1 0.98), wavelet (2D-SIWPT+BBA), IR-segmentation. Mostly closed-world; not directly competitive with deep methods but informs feature design.

---

## 5. Metrics (AuDET, APCER@1%BPCER) & score calibration

- **FREUID metric** (`freuid2026.microblink.com`): primary **AuDET** = area under the **Detection Error Trade-off** curve (single scalar over all operating points); secondary **APCER@1%BPCER** = attack-presentation error at a fixed 1% bona-fide rejection (production-relevant DET slice). Both need the **full DET curve → output calibrated continuous scores, never hard 0/1**. (Matches `CLAUDE.md`.)
- ISO/IEC 30107-3 vocabulary: **APCER** = attacks classified as bona-fide; **BPCER** = bona-fide classified as attacks. Higher fraud-score = more likely attack.
- **Calibration is the lever** (`2603.01433`, DOCFORGE-BENCH): tampered regions are tiny (**0.27–4.17% of pixels** in docs vs 10–30% in natural images), so a fixed τ=0.5 is catastrophically miscalibrated — **Pixel-AUC ≥0.76 coexists with near-zero Pixel-F1**. Oracle-F1 (best threshold) is **2–10× higher** than fixed-threshold F1 → "calibration, not representation, is the bottleneck."
  - **Adapting the threshold on as few as 10 in-domain images recovers 39–55% of the gap.** Cheap, high-value.
- For imbalanced fraud, report/optimize **FPR & FNR (distribution-insensitive)**, not accuracy (`1910.08993`).
- Practical: fit **isotonic / Platt / temperature** scaling on a held-out fold; the *ranking* (DET/AUC) is what AuDET rewards, but APCER@1%BPCER is threshold-sensitive — calibrate so scores are monotone and well-spread near the operating point.

---

## 6. Practical competition recipe

**Backbones:** EfficientNet-B3 / ConvNeXt / Swin beat plain ViT on ID forgery (ViT underperforms unless lots of data or video). For localization, SegFormer/EdgeNeXt/IML-ViT-style high-res transformers.

**Inputs / streams (multi-stream beats RGB-only):**
- RGB (high-res, ≥512, ideally 1024 for localization) +
- Noise residual: NoisePrint++ / guided-filter+Sobel / SRM +
- Frequency/DCT (JPEG-artifact) stream +
- (if GenAI family) diffusion reconstruction-error branch.

**Pipeline:** detect/crop **face + text ROIs** (YOLO) → per-region forensic classification → aggregate. Whole-image is a weak baseline.

**Training:**
- **Freeze pretrained encoders** when data is small (FLiD: fine-tuning collapsed it). MAE/self-supervised pre-training is high-value (IML-ViT).
- Multi-task: binary detect + pixel localization (edge/mask supervision) regularizes and improves both.
- Forgery-aware augmentation: **synthetic copy-move/splice/inpaint** tamper generation, JPEG/compression, **print-recapture & moiré simulation**, scanner distortion — so benign acquisition artifacts don't read as fraud.

**Validation (critical):** stratify by **`is_digital`** AND **`type` (country/doctype)**. Hold out **entire doc types / nationalities** to measure the generalization gap directly (cross-type collapse to ~50% is the documented failure). In-domain CV will be wildly optimistic.

**Calibration & output:** continuous fraud score; isotonic/Platt on held-out fold; tune operating point toward 1% BPCER; per-doc-type threshold adaptation if test distribution is known.

**Ensembling:** **score-level fusion of complementary detectors** (noise-stream model + DCT model + diffusion-error model) — documented to beat any single model on private test (EdgeDoc+TruFor 0.59/0.71 → 0.79).

---

## Recommended approach for FREUID 2026 (prioritized)

1. **Baseline fast:** EfficientNet-B3 (frozen/lightly-tuned) on full-image, output calibrated probs, submit to lock the pipeline & confirm the 7,821-vs-142,818 id-set question. Establish DET/AuDET measurement locally.
2. **ROI pipeline:** add YOLO face/text crops → per-region classifier. Expect the biggest single jump.
3. **Add a noise/forensic stream:** NoisePrint++ or guided-filter residual concatenated with RGB; or run **TruFor** off-the-shelf as a second opinion.
4. **Localization multi-task head** with edge/mask supervision (synthetic tamper masks) for regularization + interpretable FP/FN analysis.
5. **GenAI branch:** if `is_digital=True` family is diffusion-edited, add DIRE/FIRE-style reconstruction-error features.
6. **Stratified hold-out by type & is_digital** from day 1; track the cross-type gap as the real metric proxy.
7. **Calibrate hard:** isotonic on OOF, threshold-adapt toward APCER@1%BPCER; consider per-type calibration.
8. **Ensemble** 2–3 complementary streams by score fusion before final submission.

---

## Sources

| Topic | Source |
|---|---|
| FLiD frozen-backbone + ROI | arXiv:2605.09089 *(unverified)* |
| EdgeDoc + TruFor fusion (ICCV DeepID) | arXiv:2508.16284 |
| TwoHead-SwinFPN | arXiv:2601.12895 *(unverified)* |
| SIDTD models (EfficientNet vs ViT, few-shot) | arXiv:2401.01858 |
| IDNet synthetic dataset, morphing cross-type | arXiv:2408.01690 |
| MIDV-2020 dataset | arXiv:2107.00396 |
| IML-ViT | arXiv:2307.14863 |
| TruFor / Noiseprint++ | arXiv:2212.10957 |
| PSCC-Net | arXiv:2103.10596 |
| CAT-Net | IEEE 9423390 / arXiv:2108.12947 |
| ManTra-Net | CVPR 2019 |
| Guided-noise localization | arXiv:2412.01622 |
| DIRE | arXiv:2303.09295 |
| FIRE (frequency reconstruction error) | arXiv:2412.07140 |
| DOCFORGE-BENCH (calibration bottleneck) | arXiv:2603.01433 *(recent)* |
| Counterfeit-detection survey (metrics, PRNU) | arXiv:1910.08993 |
| FREUID metric/dataset facts | freuid2026.microblink.com |

*Generated from a stopped deep-research run; adversarial verification was partial. Re-confirm specific numbers against the cited source before use.*
