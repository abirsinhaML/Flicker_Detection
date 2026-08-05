.PHONY: install test verify-s3 lint format check run help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies using uv"
	@echo "  make test       - Run the test suite (offline, no AWS needed)"
	@echo "  make verify-s3  - Verify live S3 access and scoring end to end"
	@echo "  make lint       - Run ruff linter"
	@echo "  make format     - Run ruff formatter"
	@echo "  make check      - Run mypy type checker"
	@echo "  make all        - Run format, lint, check, and test"

install:
	uv sync

verify-s3:
	./scripts/verify_s3.sh

test:
	uv run python -m unittest discover -s tests -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run mypy .

all: format lint check test
