.PHONY: help run_dev python_build sort_imports docker_build

VERSION?=$(shell cat ./VERSION)
export VERSION
IMAGE_NAME?=me-model-analysis

define HELPTEXT
Makefile usage
 Targets:
    python_build    Build python package.
    sort_imports    Sort imports in python modules.
    run_dev         Run development instance of the backend.
    docker_build    Build dev backend docker image.
endef
export HELPTEXT

help:
	@echo "$$HELPTEXT"

venv:
	python3 -m venv $@
	venv/bin/pip install pycodestyle pydocstyle pylint isort codespell setuptools
	venv/bin/pip install -e .

python_build: | venv
	@venv/bin/codespell me_model_analysis
	@venv/bin/pycodestyle me_model_analysis
	@venv/bin/pydocstyle me_model_analysis
	@venv/bin/isort --check-only --line-width 100 me_model_analysis
	@venv/bin/pylint me_model_analysis
	@venv/bin/python setup.py sdist

sort_imports: | venv
	@venv/bin/isort --line-width 100 me_model_analysis

docker_build:
	docker build -t $(IMAGE_NAME):dev \
		--build-arg http_proxy=http://bbpproxy.epfl.ch:80/ \
		--build-arg https_proxy=http://bbpproxy.epfl.ch:80/ \
		--build-arg VERSION=$(VERSION) \
		.

run_dev: docker_build
	docker run --rm -it \
		-e DEBUG=True \
		-e ALLOWED_ORIGIN=http://localhost:8080 \
		-v $$(pwd)/me_model_analysis:/usr/local/lib/python3.12/site-packages/me_model_analysis \
		-v $$(pwd)/models:/opt/me-model-analysis/models \
		-p 8000:8000 \
		$(IMAGE_NAME):dev
