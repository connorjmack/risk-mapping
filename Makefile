# PC-RAI Makefile
# Common development commands

.PHONY: help install install-dev test test-cov test-fast lint format typecheck clean build docs

# Default target
help:
	@echo "PC-RAI Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install package in production mode"
	@echo "  make install-dev  Install package with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-cov     Run tests with coverage report"
	@echo "  make test-fast    Run tests excluding slow ones"
	@echo "  make test-MODULE  Run tests for specific module (e.g., make test-slope)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linter (ruff)"
	@echo "  make format       Format code (black)"
	@echo "  make typecheck    Run type checker (mypy)"
	@echo "  make check        Run all checks (lint + typecheck)"
	@echo ""
	@echo "Build:"
	@echo "  make build        Build distribution packages"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Utilities:"
	@echo "  make synthetic    Generate synthetic test data"
	@echo "  make demo         Run demo on synthetic data"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=pc_rai --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow"

# Pattern rule for testing specific modules
# Usage: make test-slope, make test-roughness, etc.
test-%:
	pytest tests/test_$*.py -v

# Run tests matching a keyword
# Usage: make test-k-classification
test-k-%:
	pytest tests/ -v -k "$*"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	ruff check pc_rai/ tests/

lint-fix:
	ruff check pc_rai/ tests/ --fix

format:
	black pc_rai/ tests/

format-check:
	black pc_rai/ tests/ --check

typecheck:
	mypy pc_rai/ --ignore-missing-imports

check: lint typecheck
	@echo "All checks passed!"

# =============================================================================
# Build
# =============================================================================

build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Utilities
# =============================================================================

# Generate synthetic test data
synthetic:
	python -c "from tests.conftest import *; print('Fixtures available')"
	@echo "Use pytest fixtures for synthetic data generation"

# Run demo pipeline on synthetic data
demo:
	@echo "Running demo on synthetic cliff data..."
	python -m pc_rai process tests/test_data/synthetic_cliff.las -o demo_output/ --methods both -v

# Check project structure
check-structure:
	@echo "Checking project structure..."
	@test -f pc_rai/__init__.py || echo "Missing: pc_rai/__init__.py"
	@test -f pc_rai/config.py || echo "Missing: pc_rai/config.py"
	@test -f pc_rai/cli.py || echo "Missing: pc_rai/cli.py"
	@test -d pc_rai/io || echo "Missing: pc_rai/io/"
	@test -d pc_rai/features || echo "Missing: pc_rai/features/"
	@test -d pc_rai/classification || echo "Missing: pc_rai/classification/"
	@test -d tests || echo "Missing: tests/"
	@echo "Structure check complete"

# Show current task from todo.md
todo:
	@echo "Current Progress:"
	@grep -A 5 "^## Project Status" pc_rai_todo.md 2>/dev/null || echo "Check pc_rai_todo.md"
	@echo ""
	@echo "Next incomplete task:"
	@grep -m 1 "^### Task.*" pc_rai_todo.md 2>/dev/null | head -1 || echo "Check pc_rai_todo.md"

# Count lines of code
loc:
	@echo "Lines of Code:"
	@find pc_rai -name "*.py" -exec cat {} + | wc -l | xargs echo "  pc_rai/:"
	@find tests -name "*.py" -exec cat {} + | wc -l | xargs echo "  tests/:"

# =============================================================================
# CI Simulation
# =============================================================================

ci: format-check lint test
	@echo "CI checks passed!"
