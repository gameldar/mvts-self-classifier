# syntax=docker/dockerfile:1

# The default base runs anywhere. The CUDA torch wheels selected via TORCH_EXTRA
# bundle their own CUDA userspace, so a GPU run only needs `--gpus all` at runtime
# (the host driver is injected by the nvidia-container-toolkit); no CUDA base image
# is required. Override BASE_IMAGE only if you need system CUDA (e.g. to build apex).
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# "cpu" (default) or "cu124" for NVIDIA GPUs. See pyproject.toml [tool.uv.sources].
ARG TORCH_EXTRA=cpu

# uv manages Python and the virtualenv, so this works on any base (slim or CUDA).
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Resolve dependencies from the lockfile first so this layer is cached until the
# lockfile changes, independent of source edits.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ${TORCH_EXTRA}

COPY . .

ENTRYPOINT ["python", "train.py"]
