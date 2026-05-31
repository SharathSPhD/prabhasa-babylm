# PSALM developer entrypoints. Thin wrappers over uv; the closure contract's
# TECHNICAL layer is `make gate`.

.PHONY: help install gate lint format types test cov clean docker paper site

help:
	@echo "PSALM make targets:"
	@echo "  install   - uv sync with dev + stats extras"
	@echo "  gate      - TECHNICAL closure gate: lint + format-check + types + tests+coverage"
	@echo "  lint      - ruff check"
	@echo "  format    - ruff format"
	@echo "  types     - mypy"
	@echo "  test      - pytest"
	@echo "  cov       - pytest with coverage report"
	@echo "  docker    - build the GB10 container"
	@echo "  paper     - build the LaTeX manuscript"
	@echo "  site      - build the Astro site"

install:
	uv sync --extra dev --extra stats

gate: lint types
	uv run ruff format --check
	uv run pytest --cov=psalm

lint:
	uv run ruff check

format:
	uv run ruff format

types:
	uv run mypy

test:
	uv run pytest

cov:
	uv run pytest --cov=psalm --cov-report=term-missing

docker:
	docker build -t psalm:dev .

paper:
	cd paper && latexmk -pdf psalm.tex

site:
	cd site && npm install && npm run build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
