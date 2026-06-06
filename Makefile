# Shortcuts for common dev tasks
# Usage: make <target>

.PHONY: install test lint format ingest run clean

install:
	pip install -r requirements.txt

test:
	pytest tests/unit/ -v --tb=short

test-all:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

format:
	black src/ tests/ app.py ingest.py

ingest:
	python ingest.py

ingest-reset:
	python ingest.py --reset

run:
	python app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf logs/*.log
