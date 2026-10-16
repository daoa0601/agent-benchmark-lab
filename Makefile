.PHONY: sync format lint typecheck test check

sync:
	uv sync --extra dev

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run basedpyright

test:
	uv run pytest --cov=agent_benchmark_lab --cov-report=term-missing

check: lint typecheck test
