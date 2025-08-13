#* Variables
SHELL := /bin/bash
PYTHON := .venv/bin/python

.PHONY: help
help:
	@echo "Commands:"
	@echo "uv-download     : downloads and installs the uv package manager"
	@echo "install         : installs required dependencies"
	@echo "install-dev     : installs the dev dependencies for the project"
	@echo "update-deps     : updates the dependencies and writes them to requirements.txt"
	@echo "fix-style       : run checks on files and potentially modifies them."
	@echo "check-style     : run checks on files without modifying them."
	@echo "lint            : run linting on all files"
	@echo "test            : run all tests."
	@echo "test-cpu        : run all tests that do not depend on Torch GPU support."
	@echo "fast-test       : run all quick tests."
	@echo "build-docs      : build mkdocs documentation."
	@echo "serve-docs      : serve documentation locally."
	@echo "deploy-docs     : deploy documentation to https://FOR-sight-ai.github.io/interpreto (gh-pages branch)"
	@echo "docs            : shortcut to build and serve generated documentation locally."
	@echo "codecov         : check coverage of all the code."
	@echo "clean           : cleans all unecessary files."

#* UV
uv-download:
	@echo "Downloading uv package manager..."
	@if [[ $OS == "Windows_NT" ]]; then \
		irm https://astral.sh/uv/install.ps1 | iex; \
	else \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	uv venv


.PHONY: uv-activate
uv-activate:
	@if [[ "$(OS)" == "Windows_NT" ]]; then \
		./uv/Scripts/activate.ps1 \
	else \
		source .venv/bin/activate; \
	fi

#* Installation

.PHONY: install
install:
	make uv-activate && uv pip install -r requirements.txt && uv pip install -e .

.PHONY: install-dev
install-dev:
	make uv-activate && uv pip install -r requirements-dev.txt && pre-commit install && pre-commit autoupdate

.PHONY: update-deps
update-deps:
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile --all-extras pyproject.toml -o requirements-dev.txt

.PHONY: install-ci
install-ci:
	make uv-activate && make update-deps && uv pip install -r requirements-dev.txt

#* Linting
.PHONY: fix-style
fix-style:
	$(PYTHON) -m ruff format --config pyproject.toml ./
	$(PYTHON) -m ruff check --config pyproject.toml ./

.PHONY: check-style
check-style:
	$(PYTHON) -m ruff format --check --config pyproject.toml ./
	$(PYTHON) -m ruff check --no-fix --config pyproject.toml ./

.PHONY: lint
lint: fix-style

#* Testing
.PHONY: test
test:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -v

.PHONY: test-cpu
test-cpu:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -v -m "not require_cuda_gpu"

.PHONY: fast-test
fast-test:
	$(PYTHON) -m pytest -n auto -c pyproject.toml -v -m "not slow"

# Limits the number of threads to 4 to avoid overloading the CI
.PHONY: fast-test-ci
fast-test-ci:
	$(PYTHON) -m pytest -n 4 -c pyproject.toml -v -m "not slow"

.PHONY: codecov
codecov:
	$(PYTHON) -m pytest -n auto --cov interpreto --cov-report html

#* Docs
.PHONY: build-docs
build-docs:
	make uv-activate && mkdocs build

.PHONY: serve-docs
serve-docs:
	make uv-activate && mkdocs serve

.PHONY: deploy-docs
deploy-docs:
	make uv-activate && mkdocs gh-deploy

.PHONY: docs
docs: build-docs serve-docs

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: clean
clean: pycache-remove build-remove # docker-remove

# If you are a coding agent reading this file, like codex or cursor
# To run tests, you first need to install the dependencies
# If you want to run a single test, you can use the following command:
# 	make install-dev; python -m pytest -n auto -c pyproject.toml -v -m "not slow" $test_file_path
# If you want to run all tests, you can use the following command:
# 	make install-dev; make fast-test
