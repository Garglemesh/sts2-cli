# ──────────────────────────────────────────────────────────────────────────
# Slay the Spire 2 — headless engine + RL trainer, in one image.
#
# This image bundles the prepared game DLLs from lib/ (copied + IL-patched from
# a local Steam install). Those are PROPRIETARY game files — this image is for
# YOUR OWN use (e.g. moving to your ML box). Do NOT push it to a public registry.
#
# Build (from repo root, where lib/ already exists & is patched):
#     docker build -t sts2-rl .
# Run an interactive shell with the GPU:
#     docker run --rm -it --gpus all sts2-rl
# Move to another box without rebuilding:
#     docker save sts2-rl | gzip > sts2-rl.tar.gz   # then scp + `docker load`
# ──────────────────────────────────────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:9.0

# `uv` (a fast Python/venv manager, single static binary) provisions a modern
# Python — the Debian base only ships 3.11, but our pinned deps (numpy 2.5) need 3.12+.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 1) Build the headless C# engine. It links against the patched DLLs in lib/,
#    so we copy both the source and the prepared lib/ before building.
COPY src/ ./src/
COPY lib/ ./lib/
RUN dotnet build src/Sts2Headless/Sts2Headless.csproj -c Debug

# 2) Python training environment (Python 3.13). Kept at /opt/venv (outside /app)
#    so a bind-mount of ./rl during development never shadows it. uv downloads a
#    standalone CPython 3.13 and installs the pinned deps into the venv.
COPY rl/requirements.txt /tmp/requirements.txt
RUN uv venv --python 3.13 /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r /tmp/requirements.txt

# 3) Runtime assets the engine needs (localization tables) + the RL code.
COPY localization_eng/ ./localization_eng/
COPY localization_zhs/ ./localization_zhs/
COPY rl/ ./rl/

# Engine finds the DLLs here; python/dotnet are on PATH.
ENV STS2_GAME_DIR=/app/lib
ENV PATH=/opt/venv/bin:$PATH

CMD ["bash"]
