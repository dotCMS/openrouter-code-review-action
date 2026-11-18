.PHONY: help fmt lint audit type test qa hooks hooks-run

PY_SRC := cli

help:
	@echo "Targets:"
	@echo "  fmt       - Format $(PY_SRC)/ (ruff format)"
	@echo "  lint      - Format, lint with autofix, type-check the whole repo, then audit"
	@echo "  audit     - Check locked dependencies for known vulnerabilities (uv audit)"
	@echo "  type      - Type-check $(PY_SRC)/ (mypy)"
	@echo "  test      - Run the test suite (pytest)"
	@echo "  qa        - Run fmt, lint, and type"
	@echo "  hooks     - Install pre-commit hooks"
	@echo "  hooks-run - Run pre-commit on all files"

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy .
	uv audit

audit:
	uv audit

fmt:
	uv run ruff format $(PY_SRC)

type:
	uv run mypy $(PY_SRC)

# Release-workflow tests create tags in temp repos; a developer's global
# tag.gpgsign=true would make git demand a tag message and fail them.
test:
	GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=tag.gpgsign GIT_CONFIG_VALUE_0=false uv run pytest -q

qa: fmt lint type

hooks:
	uv run pre-commit install

hooks-run:
	uv run pre-commit run --all-files
