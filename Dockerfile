# Flicker detection, reproducibly.
#
# The image needs no FFmpeg and no CUDA toolkit. PyAV's wheel vendors its own
# FFmpeg (libavcodec 62 and 31 companion libraries), and its NVDEC support comes
# from that build rather than from anything installed here -- it dlopens
# libnvcuvid.so from the *host* driver at run time. So the container carries the
# codec stack and the host supplies only the driver, which is what makes a plain
# slim base sufficient and keeps the image small.
#
# Build:
#   docker build -t flicker:0.2.0 .
#
# Run with the GPU (needs nvidia-container-toolkit on the host):
#   docker run --rm --gpus all \
#     -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN \
#     -v "$PWD/output:/app/output" -v "$PWD/input:/app/input:ro" \
#     flicker:0.2.0 --links --workers 8 --resume
#
# Run without a GPU -- the same command, no --gpus:
#   docker run --rm -e AWS_ACCESS_KEY_ID ... flicker:0.2.0 --links --workers 16
#
# VERIFY NVDEC IS ACTUALLY BEING USED. A container that cannot reach the driver
# falls back to software and only says so in one log line, which at corpus scale
# means quietly spending ~6x the CPU. Prove it before a long run:
#   docker run --rm --gpus all flicker:0.2.0 --decode-backend cuda \
#     s3://bucket/key.mp4 --output /tmp/probe.jsonl --no-window-dir --no-video-csv
# and check the metadata line says `decode=cuda`. Setting
# `decode.allow_fallback: false` turns a missing GPU into a hard failure, which
# is what you want in a smoke test and never in production.

FROM python:3.10-slim-bookworm

# libnvcuvid/libnvidia-encode arrive from the host via the container toolkit, so
# only the ordinary shared libraries PyAV and OpenCV link against are needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv resolves from the committed lock file, so the image pins the same versions
# the pipeline was measured with -- including av, whose bundled FFmpeg decides
# whether NVDEC is available at all.
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY main.py run.py ./
COPY src/ ./src/
COPY configs/ ./configs/

# Written to by the batch, and the mount point for results.
RUN mkdir -p output reports

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Arguments go straight through, so the container is the CLI:
#   docker run ... flicker:0.2.0 --links --workers 8 --resume
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
