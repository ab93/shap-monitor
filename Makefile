setup:
	poetry install

lint:
	poetry run black .
	poetry run ruff format .
	poetry run ruff check . --fix --show-fixes

test:
	poetry run pytest tests/
