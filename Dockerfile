# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# #
# # This source code is licensed under the BSD-style license found in the
# # LICENSE file in the root directory of this source tree.

# # Multi-stage build using openenv-base
# # This Dockerfile is flexible and works for both:
# # - In-repo environments (with local OpenEnv sources)
# # - Standalone environments (with openenv from PyPI/Git)
# # The build script (openenv build) handles context detection and sets appropriate build args.

# ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
# FROM ${BASE_IMAGE} AS builder

# WORKDIR /app

# # Ensure git is available (required for installing dependencies from VCS)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends git && \
#     rm -rf /var/lib/apt/lists/*

# # Build argument to control whether we're building standalone or in-repo
# ARG BUILD_MODE=in-repo
# ARG ENV_NAME=first_rl_demo

# # Copy environment code (always at root of build context)
# COPY . /app/env

# # For in-repo builds, openenv is already vendored in the build context
# # For standalone builds, openenv will be installed via pyproject.toml
# WORKDIR /app/env

# # Ensure uv is available (for local builds where base image lacks it)
# RUN if ! command -v uv >/dev/null 2>&1; then \
#         curl -LsSf https://astral.sh/uv/install.sh | sh && \
#         mv /root/.local/bin/uv /usr/local/bin/uv && \
#         mv /root/.local/bin/uvx /usr/local/bin/uvx; \
#     fi
    
# # Install dependencies using uv sync
# # If uv.lock exists, use it; otherwise resolve on the fly
# RUN --mount=type=cache,target=/root/.cache/uv \
#     if [ -f uv.lock ]; then \
#         uv sync --frozen --no-install-project --no-editable; \
#     else \
#         uv sync --no-install-project --no-editable; \
#     fi

# RUN --mount=type=cache,target=/root/.cache/uv \
#     if [ -f uv.lock ]; then \
#         uv sync --frozen --no-editable; \
#     else \
#         uv sync --no-editable; \
#     fi

# # Final runtime stage
# FROM ${BASE_IMAGE}

# WORKDIR /app

# # Copy the virtual environment from builder
# COPY --from=builder /app/env/.venv /app/.venv

# # Copy the environment code
# COPY --from=builder /app/env /app/env

# # Set PATH to use the virtual environment
# ENV PATH="/app/.venv/bin:$PATH"

# # Set PYTHONPATH so imports work correctly
# ENV PYTHONPATH="/app/env:$PYTHONPATH"

# ENV ENABLE_WEB_INTERFACE=true
# # Health check
# HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# # Run the FastAPI server
# # The module path is constructed to work with the /app/env structure
# CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]


# # Use the OpenEnv base image provided by Meta
# ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
# FROM ${BASE_IMAGE} AS builder

# WORKDIR /app

# # 1. Install system dependencies for Git and MuJoCo rendering
# # We need these to handle the graphics pipeline for the robot camera
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     git \
#     # libgl1-mesa-glx \
#     libgl1 \
#     libosmesa6 \
#     libglew-dev \
#     mesa-utils \
#     && rm -rf /var/lib/apt/lists/*

# # Copy your project files
# COPY . /app/env
# WORKDIR /app/env

# # Ensure uv is available for dependency management
# RUN if ! command -v uv >/dev/null 2>&1; then \
#         curl -LsSf https://astral.sh/uv/install.sh | sh && \
#         mv /root/.local/bin/uv /usr/local/bin/uv && \
#         mv /root/.local/bin/uvx /usr/local/bin/uvx; \
#     fi
    
# # Install dependencies using uv
# # This creates the .venv that contains torch, transformers, and mujoco
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --no-editable

# # --- Final Runtime Stage ---
# FROM ${BASE_IMAGE}

# # Re-install runtime graphics libraries (required in the final image)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends \
#     # libgl1-mesa-glx \
#     libgl1 \
#     libosmesa6 \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /app
# # Copy the prepared environment and code from the builder stage
# COPY --from=builder /app/env /app/env
# COPY --from=builder /app/env/.venv /app/.venv

# # Set up paths so 'python' points to your virtual environment
# ENV PATH="/app/.venv/bin:$PATH"
# ENV PYTHONPATH="/app/env:$PYTHONPATH"

# # 2. MuJoCo Headless Configuration
# # Since Hugging Face has no monitor, we force MuJoCo to render via CPU (OSMesa)
# ENV MUJOCO_GL="osmesa"
# ENV PYOPENGL_PLATFORM="osmesa"

# WORKDIR /app/env

# # # 3. The Training Trigger
# # # This tells Docker exactly which file to run the moment the Space starts
# # CMD ["python", "scripts/train_llm.py", \
# #      "--train-steps", "60", \
# #      "--group-size", "4", \
# #      "--max-new-tokens", "12", \
# #      "--temperature", "0.2", \
# #      "--top-p", "0.9", \
# #      "--log-every", "1", \
# #      "--save-every", "60"]



# ENV WANDB_API_KEY=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
# ENV WANDB_SILENT=true

# CMD ["python", "scripts/train_llm.py", \
#      "--model-name", "Qwen/Qwen2.5-3B-Instruct", \
#      "--train-steps", "1000", \
#      "--group-size", "4", \
#      "--max-new-tokens", "64", \
#      "--temperature", "0.7", \
#      "--use-wandb", \
#      "--wandb-project", "ptz-camera-alignment", \
#      "--push-checkpoints-to-hub", \
#      "--hub-repo-id", "JanaksinhVen/ptz-qwen-vla-checkpoints", \
#      "--hub-private"]



# # The Training Trigger
# # CMD ["python", "scripts/train_llm.py", \
# #     "--train-steps", "60", \
# #     "--group-size", "4", \
# #     "--max-new-tokens", "128", \
# #     "--temperature", "0.2", \
# #     "--top-p", "0.9", \
# #     "--log-every", "1", \
# #     "--save-every", "60", \
# #     "--use-wandb", \
# #     "--wandb-project", "ptz-camera-alignment"]




# Use the OpenEnv base image
FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

# 1. Install system dependencies for Git and MuJoCo rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libosmesa6 \
    libglew-dev \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv globally
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx

# Copy your project files
COPY . /app/env
WORKDIR /app/env

# 3. FORCE UV TO INSTALL TO THE SYSTEM PATH 
# This is the "Magic Fix" for ModuleNotFoundError in Docker
ENV UV_PROJECT_ENVIRONMENT="/usr/local"

# Sync dependencies (this will install transformers, torch, etc. to /usr/local)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable --locked

# 4. MuJoCo & Python Path Configuration
ENV MUJOCO_GL="osmesa"
ENV PYOPENGL_PLATFORM="osmesa"
ENV PYTHONPATH="/app/env"

# 5. Secrets and W&B (W&B key is here as requested, but HF_TOKEN should be a Secret)
ENV WANDB_API_KEY=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
ENV WANDB_SILENT=true

# 6. The Training Trigger
# CMD ["python", "scripts/train_llm.py", \
#      "--model-name", "Qwen/Qwen2.5-3B-Instruct", \
#      "--train-steps", "1000", \
#      "--group-size", "4", \
#      "--max-new-tokens", "128", \
#      "--temperature", "0.7", \
#      "--use-wandb", \
#      "--wandb-project", "ptz-camera-alignment", \
#      "--push-checkpoints-to-hub", \
#      "--hub-repo-id", "JanaksinhVen/ptz-qwen-vla-checkpoints", \
#      "--hub-private"]