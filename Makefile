.PHONY: install test lint format check run help

help:
	@echo "Available commands:"
	@echo "  make install  - Install dependencies using uv"
	@echo "  make test     - Run the test suite"
	@echo "  make lint     - Run ruff linter"
	@echo "  make format   - Run ruff formatter"
	@echo "  make check    - Run mypy type checker"
	@echo "  make all      - Run format, lint, check, and test"

install:
	uv sync

test:
	uv run python -m unittest discover -s tests -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run mypy .

all: format lint check test
