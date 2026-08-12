# Flicker detection, one container per shard.
#
# The image needs no FFmpeg and no CUDA toolkit. PyAV's wheel vendors its own
# FFmpeg (libavcodec 62 and 31 companion libraries), and its NVDEC support comes
# from that build rather than from anything installed here -- it dlopens
# libnvcuvid.so from the *host* driver at run time. So the container carries the
# codec stack, the host supplies only the driver, and a plain slim base is enough.
#
# See DOCKER.md for the full run book. In short:
#
#   docker build -t flicker:0.2.0 .
#   docker run -d --name flicker --restart unless-stopped --gpus all \
#     -v /data/input:/app/input:ro -v /data/output:/app/output \
#     flicker:0.2.0 --from-row 0 --to-row 23170
#
# The entry point is scripts/run_batch.sh, so container arguments are the shard's
# row range and nothing else. That script pins the thread environment, picks a
# worker count, checks S3 credentials before decoding anything, and names its
# outputs after the range -- all of which would have to be reproduced by hand if
# the entry point were main.py directly.

FROM python:3.10-slim-bookworm

# libgl1/libglib2.0-0: OpenCV's runtime dependencies.
# procps: run_batch.sh and stop_batch.sh identify the batch with `ps`, which the
#         slim base does not ship; without it the guard fails under `set -e`
#         before any work starts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv resolves from the committed lock file, so the image pins the same versions
# the pipeline was measured with -- including av, whose bundled FFmpeg decides
# whether NVDEC is available at all.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY main.py run.py upload_output.py ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Mount points. input/ holds the link sheet and is read-only; output/ must be a
# volume or the results die with the container.
RUN mkdir -p input output reports && chmod +x scripts/*.sh scripts/*.py

# FLICKER_CONFIG: main.py's --config default. Without it every invocation that
# bypasses the entry point (--print-shards, for one) would fall back to
# configs/detector.yaml, which points at the reference corpus and has no s3.links.
#
# NumPy is on scipy-openblas, which spawns one thread per core in *every* worker.
# Unpinned, the detector stage measured 20.4 ms wall / 308 ms CPU across 15.1
# threads; pinned it is 10.6 ms wall / 10.6 ms CPU -- faster and 29x cheaper,
# because the thread thrash cost more than the parallelism bought. Scores are
# bit-identical either way. run_batch.sh exports these too; setting them here as
# well means they hold even if the entry point is overridden.
ENV FLICKER_CONFIG=configs/detector_1.yaml \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Arguments are passed through to run_batch.sh, and from there to main.py:
#   docker run ... flicker:0.2.0 --from-row 0 --to-row 23170
#   docker run ... flicker:0.2.0 --from-row 0 --to-row 99 --no-s3-output
ENTRYPOINT ["scripts/run_batch.sh"]
