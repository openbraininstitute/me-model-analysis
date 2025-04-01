# syntax=docker/dockerfile:1.9
ARG UV_VERSION=0.6
ARG PYTHON_VERSION=3.12
ARG PYTHON_BASE=${PYTHON_VERSION}-slim

# uv stage
FROM ghcr.io/astral-sh/uv:${UV_VERSION} as uv

# build stage
FROM python:$PYTHON_BASE AS builder
SHELL ["bash", "-e", "-x", "-o", "pipefail", "-c"]

RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    build-essential \
    ca-certificates
EOT

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_PYTHON=python${PYTHON_VERSION}

WORKDIR /code
ARG ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml <<EOT
if [ "${ENVIRONMENT}" = "prod" ]; then
  uv sync --locked --no-install-project --no-dev
elif [ "${ENVIRONMENT}" = "dev" ]; then
  uv sync --locked --no-install-project
else
  echo "Invalid ENVIRONMENT"; exit 1
fi
EOT

# run stage
FROM python:$PYTHON_BASE
SHELL ["bash", "-e", "-x", "-o", "pipefail", "-c"]

RUN <<EOT
apt-get update -qy
apt-get install -qyy \
    -o APT::Install-Recommends=false \
    -o APT::Install-Suggests=false \
    build-essential
EOT

RUN <<EOT
groupadd -r app
useradd -r -d /code -g app -N app
EOT

USER app
WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH"
ENV PYTHONPATH="/code:$PYTHONPATH"
COPY --chown=app:app --from=builder /code/.venv/ .venv/
COPY --chown=app:app docker-cmd.sh pyproject.toml ./
COPY --chown=app:app app/ app/
COPY --chown=app:app nexus/ nexus/

RUN python -m compileall .  # compile app files

ARG ENVIRONMENT
ENV ENVIRONMENT=${ENVIRONMENT}

RUN <<EOT
python -V
python -m site
python -c 'import app'
EOT

STOPSIGNAL SIGINT
CMD ["./docker-cmd.sh"]
