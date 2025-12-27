clean:
	rm -rf dist
	rm -rf build
	rm -rf site
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

setup:
	poetry install

lint:
	poetry run black .
	poetry run ruff format .
	poetry run ruff check . --fix --show-fixes

test:
	poetry run pytest tests/

coverage:
	poetry run pytest --cov=shapmonitor --cov-report=html tests/

docs-serve:
	poetry run mkdocs serve

docs-build:
	poetry run mkdocs build --strict
