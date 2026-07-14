# Reproducibility container (team jgshin22)

Implements the organizer sandbox contract
([reproducibility page](https://freuid2026.microblink.com/reproducibility.html)):
`/data` = read-only flat directory of test images, `/submissions` = writable
output, no network at runtime.

## Build (from the repository root)

```bash
docker build -t freuid-repro:local -f submission/Dockerfile .
```

Everything needed at inference is baked into the image: model weights
(`assets/best.pt`, 43 MB), per-type field boxes, and the training-image dhash
references used for document-type assignment. No runtime downloads.

## Run

```bash
docker run --rm \
  --network none \
  -v /path/to/flat/test/images:/data:ro \
  -v "$(pwd)/out:/submissions" \
  freuid-repro:local
```

Writes `/submissions/submission.csv` (`id,label`, label = fraud score,
higher = more likely fraud; one row per image file). Runs on CPU by default;
pass `--gpus all` for CUDA (a 10 GB GPU scores the 7,821 public images in a
few minutes, CPU is ~20-30x slower).

## Files

| File | Purpose |
| ---- | ------- |
| `Dockerfile` | inference image (python:3.11-slim + pinned pip deps) |
| `prepare_submission.py` | entrypoint: dhash type assignment → 13 field crops @320px → EffB3+SRM MIL scorer → hflip-TTA max-agg |
| `requirements.txt` | pinned inference dependencies |
| `freuid_technical_report.tex` / `.pdf` | technical report |
| `REPLY_TEMPLATE.txt` | filled Kaggle discussion reply |

The same script also ran our private-test inference (multi-GPU via
`--shard K --nshards N`; shards concatenated afterwards).
