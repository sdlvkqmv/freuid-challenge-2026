# FREUID Challenge 2026 — Field-Crop MIL Forgery Detector

Solution for the [FREUID Challenge 2026](https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai)
(IJCAI-ECAI): binary detection of forged/manipulated identity documents.

**Method in one paragraph.** All five document types in the dataset are
template-locked: after resizing to a per-type canonical resolution, every
document's fields (photo, name, DOB, document number, MRZ, holograms) sit at
fixed pixel locations. Instead of scoring a downsampled whole image — where
document text is ~10 px tall and character-level tampering is physically
invisible — we cut K=13 native-resolution field crops per document (boxes
derived from per-type pixel-STD maps over bona-fide images, plus 3
static-zone crops) and score each crop with an EfficientNet-B3 fed
channel-concatenated RGB + SRM noise residual (6 input channels). The
document fraud score is the **maximum** over crop scores
(multiple-instance learning), with horizontal-flip test-time augmentation.
Test-image document types are assigned by 1024-bit dhash nearest-neighbour
matching against the 69,352 training images (template dominance makes this
near-perfect).

Public leaderboard: **0.03905** (FREUID score, lower is better).

## Repository layout

```
freuid/                  training library (config, data, metrics, model: RGB+SRM streams)
scripts/train_mil_v2.py  final-submission training script (single file)
configs/mil_fieldcrop_v2.yaml  training config
assets/
  best.pt                frozen final model weights (43 MB, epoch-4 best-val checkpoint)
  field_boxes_v2.json    per-type field crop boxes (canonical-resolution pixel coords)
  train_dhash_refs.npz   packed 1024-bit dhashes + types of all training images
  test_type_assignment.csv  public-test document types (dhash NN, reproducible)
submission/
  prepare_submission.py  self-contained inference entrypoint (sandbox contract)
  Dockerfile             reproducibility container
  requirements.txt       pinned inference dependencies
docs/provenance/         training logs/configs of the two selected runs
```

## Docker (organizer verification)

Build from the repository root:

```bash
docker build -t freuid-repro:local -f submission/Dockerfile .
```

Run under the no-network sandbox contract (`/data` = flat read-only directory
of test images, `/submissions` = writable output directory):

```bash
docker run --rm --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Writes `/submissions/submission.csv` with columns `id,label`
(label = fraud score in [0,1], higher = more likely fraud).

**Model selection.** `FREUID_MODEL=a31` (default) reproduces our primary
final submission (public LB 0.03905). `docker run -e FREUID_MODEL=a30 ...`
reproduces our second final selection (field-crop MIL v1: 10 crops @256px,
no TTA; public LB 0.08023). Both checkpoints were trained on 2026-07-08,
before the July 13 code freeze (training logs archived under
`docs/provenance/`); post-release work was inference-only (private-row
prediction with the frozen checkpoints).

Precision note: on a CUDA host (`--gpus all`) the container reproduces the
submitted scores bit-exactly (inference ran under fp16 autocast). On CPU
(fp32) scores differ by < 3e-4 absolute, which does not alter score ranking
in any meaningful way for the DET-based metric.
Everything needed at inference (weights, field boxes, dhash references) is
baked into the image; no network access is used. The container runs on CPU;
if the host passes `--gpus all` it uses CUDA automatically (recommended:
inference over 7,821 images takes a few minutes on one 10 GB GPU, CPU is
roughly 20-30x slower).

## Model weights

`assets/best.pt` (43 MB) is the best-validation checkpoint (epoch 4 of 6)
of the final v2 training run (seed 42); `assets/best_a30.pt` is the v1
checkpoint (second final selection). SHA-256 hashes are recorded in
`assets/CHECKSUMS.txt`.

## External data / pretrained models

- No external document data. Training uses only the competition training set.
- The backbone is initialized from ImageNet-pretrained `tf_efficientnet_b3`
  weights via `timm` (downloaded at **training** time only; inference uses the
  committed checkpoint and never touches the network).

## Reproducing training

Hardware used: 1x NVIDIA GPU (10 GB VRAM), ~6 h wall-clock, 64 GB RAM
recommended for data loading. Python 3.11, PyTorch 2.4.1, timm 1.0.27.

```bash
# 1. download competition data into data/ (kaggle CLI) and extract so that
#    data/extracted/train/train/*.jpeg and train_labels.csv exist
# 2. train (writes run dir under experiments/, ends with a full
#    142,818-row submission.csv using best-val weights):
python scripts/train_mil_v2.py <GPU_INDEX>
```

The training script is a verbatim copy of the one archived inside the winning
run directory (only paths rewritten repo-relative). Validation is a
stratified 5-fold split (fold 0 held out). The final Kaggle submission's
public-test rows come from this run; private-test rows were produced after
the July 13 private release by running the same frozen checkpoint via
`submission/prepare_submission.py` (inference only — no weight or code
changes).

## License

MIT — see [LICENSE](LICENSE).
