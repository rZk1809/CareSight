# CareSight Risk Engine Makefile

.PHONY: help setup install data train all clean test lint format check dashboard serve

# Default target
help:
	@echo "CareSight Risk Engine - Available targets:"
	@echo ""
	@echo "  setup     - Install dependencies and setup pre-commit hooks"
	@echo "  install   - Install Python dependencies only"
	@echo "  data      - Run data pipeline (cohort, labels, features, merge)"
	@echo "  train     - Run training pipeline (train, calibrate, evaluate)"
	@echo "  all       - Run complete pipeline (data + train)"
	@echo "  clean     - Clean generated files and cache"
	@echo "  test      - Run unit tests"
	@echo "  lint      - Run code linting"
	@echo "  format    - Format code with black"
	@echo "  check     - Run all quality checks (lint + test)"
	@echo "  dashboard - Start Streamlit dashboard"
	@echo "  serve     - Start FastAPI server"
	@echo ""

# Setup development environment
setup: install
	@echo "Setting up development environment..."
	pip install pre-commit
	pre-commit install
	@echo "Development environment setup complete!"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed!"

# Run data pipeline stages
data:
	@echo "Running data pipeline..."
	dvc repro build_cohort make_labels make_features merge_training_table
	@echo "Data pipeline complete!"

# Run training pipeline stages
train:
	@echo "Running training pipeline..."
	dvc repro train_lgbm calibrate evaluate
	@echo "Training pipeline complete!"

# Run complete pipeline
all:
	@echo "Running complete pipeline..."
	dvc repro
	@echo "Complete pipeline finished!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	@echo "Cleanup complete!"

# Run tests
test:
	@echo "Running unit tests..."
	pytest tests/ -v
	@echo "Tests complete!"

# Run linting
lint:
	@echo "Running code linting..."
	flake8 src/ tests/
	@echo "Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	@echo "Code formatting complete!"

# Run all quality checks
check: lint test
	@echo "All quality checks passed!"

# Show pipeline status
status:
	@echo "Pipeline status:"
	dvc status

# Show pipeline DAG
dag:
	@echo "Pipeline DAG:"
	dvc dag

# Reproduce specific stage
repro-%:
	@echo "Reproducing stage: $*"
	dvc repro $*

# Show metrics
metrics:
	@echo "Model metrics:"
	@python -c "import os, json; print(json.dumps(json.load(open('data/reports/metrics.json')), indent=2) if os.path.exists('data/reports/metrics.json') else 'No metrics file found. Run make all first.')"

# Initialize DVC (for first-time setup)
dvc-init:
	@echo "Initializing DVC..."
	dvc init
	@echo "DVC initialized!"

# Add data directory to DVC tracking
dvc-add-data:
	@echo "Adding data directory to DVC tracking..."
	dvc add data/
	@echo "Data directory added to DVC!"

# Start Streamlit dashboard
dashboard:
	@echo "🚀 Starting Streamlit dashboard..."
	@echo "Dashboard will be available at: http://localhost:8501"
	python -m streamlit run src/dashboards/streamlit_app.py --server.address=localhost

# Start FastAPI server
serve:
	@echo "🚀 Starting FastAPI server..."
	@echo "API will be available at: http://localhost:8000"
	@echo "API docs will be available at: http://localhost:8000/docs"
	python -m uvicorn src.serving.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
