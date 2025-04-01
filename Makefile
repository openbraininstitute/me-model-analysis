SHELL := /bin/bash

export ENVIRONMENT ?= dev
export APP_NAME := me-model-validation
export IMAGE_NAME ?= $(APP_NAME)
export IMAGE_TAG := latest-$(ENVIRONMENT)

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-23s\033[0m %s\n", $$1, $$2}'
install:  ## Install dependencies into .venv
	uv sync --no-install-project

compile-deps:  ## Create or update the lock file, without upgrading the version of the dependencies
	uv lock

upgrade-deps:  ## Create or update the lock file, using the latest version of the dependencies
	uv lock --upgrade

check-deps:  ## Check that the dependencies in the existing lock file are valid
	uv lock --locked

format:  # Run formatters
	uv run -m ruff format
	uv run -m ruff check --fix

lint:  ## Run linters
	uv run -m ruff format --check
	uv run -m ruff check
	uv run -m mypy app

build:  ## Build the Docker image
	docker compose build app

publish: build  ## Publish the Docker image to DockerHub
	docker compose push app

run: build  ## Run the application in Docker
	docker compose up --watch --remove-orphans

kill:  ## Take down the application and remove the volumes
	docker compose down --remove-orphans --volumes

clean: ## Take down the application and remove the volumes and the images
	docker compose down --remove-orphans --volumes --rmi all

show-config:  ## Show the docker-compose configuration in the current environment
	docker compose config

sh: build  ## Run a shell in the app container
	docker compose run --rm app bash
