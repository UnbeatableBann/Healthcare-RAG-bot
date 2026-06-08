.PHONY: run ingest evaluate test up down no-cache

run:
	uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	uv run python -m scripts.ingest

evaluate:
	uv run python -m scripts.evaluate data/evaluation/dataset.json

test:
	uv run pytest

login:
	docker login

up:
	docker compose up --build

down:
	docker compose down

no-cache:
	docker compose build --no-cache