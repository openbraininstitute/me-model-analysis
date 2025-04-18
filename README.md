# me-model-analysis

## Description

Repo that contains the code to run me-model-analysis using on-demand service in AWS

## Local build and deployment

Requirements:

- [Docker compose](https://docs.docker.com/compose/) >= 2.24.4
- [uv](https://docs.astral.sh/uv/)

Valid `make` targets:

```
help                    Show this help
install                 Install dependencies into .venv
compile-deps            Create or update the lock file, without upgrading the version of the dependencies
upgrade-deps            Create or update the lock file, using the latest version of the dependencies
check-deps              Check that the dependencies in the existing lock file are valid
lint                    Run linters
build                   Build the docker images
run                     Run the application in docker
kill                    Take down the application and remove the volumes
clean                   Take down the application and remove the volumes and the images
show-config             Show the docker-compose configuration in the current environment
sh                      Run a shell in the app container
```

To build and start the Docker images locally, you can execute:

```bash
make run
```

## Remote deployment

To make a release, build and publish the Docker image to the registry, you need to start the `publish-production-image` workflow.

The format of the tag should be `YYYY.MM.DD.N`, where:

- `YYYY` is the full year (2024, 2025 ...)
- `M` is the month, zero-padded (01, 02 ... 11, 12)
- `D` is the date, zero-padded (01, 02 ... 29, ...)
- `N` is any incremental number, not zero-padded

## Documentation

The API documentation is available locally at <http://127.0.0.1:8100/docs> after the local deployment.

## Funding & Acknowledgment

The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government's ETH Board of the Swiss Federal Institutes of Technology.

Copyright © 2024 Blue Brain Project/EPFL
Copyright © 2025 Open Brain Institute
