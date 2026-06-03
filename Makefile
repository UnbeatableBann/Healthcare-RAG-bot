.PHONY: run ingest evaluate test docker-up docker-down

run:
	uv run uvicorn apps.api.main:app --reload --host 0.0.0.0 --port 8000

ingest:
	uv run python -m scripts.ingest

evaluate:
	uv run python -m scripts.evaluate data/evaluation/dataset.json

test:
	uv run pytest

docker-up:
	docker compose up --build

docker-down:
	docker compose down
