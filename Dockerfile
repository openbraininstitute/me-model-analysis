FROM python:3.12-slim AS builder
WORKDIR /app
ARG VERSION
COPY . .
RUN python setup.py sdist

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

ARG VERSION

RUN apt-get update \
  && apt-get install -q -y --no-install-recommends build-essential git \
  && pip install --no-cache-dir --upgrade setuptools pip

WORKDIR /opt/me-model-analysis

COPY packages packages
RUN pip install matplotlib==3.8.4 ./packages/icselector \
  ./packages/bluepyemodel \
  ./packages/bluepyemodelnexus \
  ./packages/nexusforge

COPY --from=builder /app/dist/me_model_analysis-${VERSION}.tar.gz ./
RUN pip install --no-cache-dir me_model_analysis-${VERSION}.tar.gz

# forge config files
COPY nexus .

COPY entrypoint.sh logging.yaml .

EXPOSE 8080

ENTRYPOINT ["./entrypoint.sh"]
