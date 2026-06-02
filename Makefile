.PHONY: install data train-als train-two-tower train-ranker experiments drift test lint api ui docker-up all

PYTHON := C:/Users/fikri/AppData/Local/Programs/Python/Python311/python.exe

## Install full development dependencies
install:
	$(PYTHON) -m pip install -r requirements-dev.txt

## Download MovieLens 1M dataset
data:
	$(PYTHON) scripts/download_data.py

## Train ALS baseline
train-als:
	$(PYTHON) -m src.train_als

## Train Two-Tower neural retrieval model
train-two-tower:
	$(PYTHON) -m src.train_two_tower

## Train LightGBM ranker (requires Two-Tower embeddings)
train-ranker:
	$(PYTHON) -m src.train_ranker

## Run full 3-model comparison experiment
experiments:
	$(PYTHON) -m src.experiments

## Run all in sequence
all: data experiments

## Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing

## Lint
lint:
	$(PYTHON) -m ruff check src/ tests/ api/ app/ --ignore E501

## Start FastAPI server
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Start Gradio UI locally
ui:
	$(PYTHON) app/gradio_app.py

## Run data drift report (PSI-based, saves reports/drift_report.html)
drift:
	$(PYTHON) -m monitoring.drift_report

## Build and start Docker stack (API + MLflow)
docker-up:
	docker-compose up --build
