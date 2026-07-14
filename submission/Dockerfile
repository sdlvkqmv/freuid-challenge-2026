# FREUID Challenge 2026 — reproducibility Dockerfile (team submission)
#
# Build (from repository root):
#   docker build -t freuid-repro:local -f submission/Dockerfile .
# Run (organizer sandbox contract):
#   docker run --rm --network none \
#     -v /path/to/flat/test/images:/data:ro \
#     -v "$(pwd)/out:/submissions" \
#     freuid-repro:local
#
# Runs on CPU out of the box; uses CUDA automatically when the host passes
# --gpus (torch wheel bundles cu121).

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    FREUID_DATA_DIR=/data \
    FREUID_OUTPUT_DIR=/submissions \
    FREUID_SUBMISSION_PATH=/submissions/submission.csv

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY submission/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY submission/prepare_submission.py .
# Frozen artifacts: model weights (43 MB), per-type field boxes,
# train-image dhash references for document-type assignment.
COPY assets/ /app/assets/

RUN useradd --create-home --uid 1000 runner
USER runner

ENTRYPOINT ["python", "/app/prepare_submission.py"]
CMD []
