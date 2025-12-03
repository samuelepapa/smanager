.PHONY: install install-dev lint format test clean help

help:
	@echo "Available commands:"
	@echo "  make install      Install package"
	@echo "  make install-dev  Install package with dev dependencies"
	@echo "  make lint         Run all linters"
	@echo "  make format       Format code with black and isort"
	@echo "  make test         Run tests"
	@echo "  make clean        Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	black --check --diff .
	isort --check-only --diff .
	pylint smanager --rcfile=pyproject.toml

format:
	black .
	isort .

test:
	pytest tests/ -v --cov=smanager --cov-report=term-missing

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
