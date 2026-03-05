# CarbonSight

CLI-first greenest cloud advisor and launcher using WattTime grid emissions and SkyPilot.

## Install

From repo root (CarbonSight):

```bash
cd carbonsight
pip install -e .
```

For dev (E2E + vcr):

```bash
pip install -e ".[dev]"
```

## Configure

Copy `.env` from repo root or set:

- `WATTTIME_USERNAME` / `WATTTIME_PASSWORD` — WattTime API credentials
- `CARBON_PROVIDER=watttime`
- `LOG_LEVEL=INFO`

## Commands

- `carbonsight advise --yaml train.yaml [--explain] [--json]` — Ranked regions with carbon/cost and confidence
- `carbonsight run train.yaml [--dry-run] [--no-exec] [--skip-preflight]` — Advise + AWS quota preflight + patch YAML + optional SkyPilot launch
- `carbonsight mappings validate [--registry path]` — Drift checks (region-from-loc vs stored)
- `carbonsight backtest run [--days 180] [--n 1000] [--seed 42] [--json]` — Backtest harness (synthetic workloads, metrics)

## API

From `carbonsight` directory:

```bash
pip install -e ".[api]"
uvicorn carbonsight_api.main:app --reload
```

- `GET /health` — Health check
- `POST /v1/recommendations` — Body: `{ "gpu_type", "gpu_count", "duration_hours" }` → ranked list
- `POST /v1/runs` — Body: `{ "job_spec", "chosen_region", "estimated_co2_kg", "estimated_cost_usd" }` → `run_id`
- `GET /v1/runs/{id}` — Run status and estimates
- `GET /v1/regions` — Cloud regions with mapping confidence
- `POST /v1/mappings/revalidate` — Trigger drift job

Docker: `docker-compose -f infra/docker/docker-compose.yml up -d` (API + Postgres; schema in `infra/schemas.sql`).

## E2E tests

Replay (no credentials): `pytest tests/e2e/ -v`  
Record/refresh: `pytest tests/e2e/ -v --record-mode=once` (requires WattTime in `.env`)

## Design

See *CarbonSight Technical Design for Execution* and `IMPLEMENTATION_PLAN.md` in the repo.
