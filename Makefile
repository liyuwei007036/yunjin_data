.PHONY: help install dev test clean lint format run

help:
	@echo "Available commands:"
	@echo "  install     - Install package in development mode"
	@echo "  dev         - Install development dependencies"
	@echo "  test        - Run test suite"
	@echo "  clean       - Clean build artifacts and cache"
	@echo "  lint        - Run linter (ruff)"
	@echo "  format      - Format code (ruff)"
	@echo "  run-test    - Run scraper in test mode"
	@echo "  run-full    - Run scraper in full mode"
	@echo "  playwright  - Install Playwright browsers"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	playwright install chromium

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ *.egg-info .eggs/

lint:
	ruff check src/ tests/

format:
	ruff check src/ tests/ --fix

run-test:
	python -m digicol_scraper.scraper --mode test --limit 10

run-full:
	python -m digicol_scraper.scraper --mode full

playwright:
	playwright install chromium
