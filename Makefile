.PHONY: install run tests

setup-env:
	pip install --upgrade pip poetry
	poetry config virtualenvs.in-project true
	poetry config cache-dir /tmp/poetry-cache

install: setup-env
	poetry install --only main --no-interaction

install-dev: setup-env
	poetry install --no-interaction

ci-mlflow-download:
	poetry run mlflow artifacts download \
		--dst-path data/ \
		--artifact-uri models:/$(mlflow-model-name)/$(mlflow-model-stage-or-version)

	cat data/requirements.txt | xargs poetry add

linter:
	poetry run black . --check
	poetry run isort . --check --diff
	poetry run pflake8
	poetry run pydocstyle .

format:
	poetry run black .
	poetry run isort .

tests:
	poetry run pytest -v -s tests/

ci: linter tests

run:
	poetry shell && $(shell cat .env | xargs) newrelic-admin run-program mlserver start SERVINGAPIserving

