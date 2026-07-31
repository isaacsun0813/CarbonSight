# CarbonSight (Python package)

SkyNomad-style **multi-lever scheduler** (cost · carbon · time · spot availability) with a **central WattTime cache** so CLIs need no personal grid credentials.

## Install (`uv`)

```bash
cd carbonsight
uv sync --all-extras
```

Or editable classic: `uv pip install -e ".[dev,api]"`.

## Configure

| Mode | Env |
|------|-----|
| **Team CLI → API** | `CARBONSIGHT_API_URL=http://localhost:8001` |
| **Direct WattTime** | `WATTTIME_USERNAME` / `WATTTIME_PASSWORD` |
| **Offline / CI** | (none) → synthetic 17-region MOER |

Copy [`.env.example`](.env.example). Never commit secrets.

## Commands

```bash
# Multi-lever schedule (joint U_s)
uv run carbonsight schedule \
  --yaml tests/fixtures/train_minimal.yaml \
  --deadline-hours 45 --checkpoint-size-gb 100 --carbon-price 50 --json

# Classic advise (falls back to joint/synthetic without WattTime)
uv run carbonsight advise --yaml tests/fixtures/train_minimal.yaml --json

# Spot-aware backtest
uv run carbonsight backtest run --spot --n 200 --json
```

## API

```bash
uv run uvicorn carbonsight_api.main:app --host 0.0.0.0 --port 8001
```

| Endpoint | Notes |
|----------|--------|
| `GET /v1/carbon/forecast?region=CAISO_NORTH` | Cached MOER series |
| `GET /v1/regions` | Cloud/grid regions (non-empty) |
| `GET /v1/recommendations` | **17** ranked rows (synthetic OK) |

Docker: `infra/docker/docker-compose.yml` (API **:8001**, worker, Postgres `grid_signal_cache`).

## Domain map

- Core: `packages/core/carbonsight_core/`
- Spot / SkyNomad: `…/spot/`
- WattTime cache: `…/watttime/`
- Providers: `…/providers/`
- See repo root [`ARCHITECTURE.md`](../ARCHITECTURE.md) for formulas and code pointers.

## Tests

```bash
uv run pytest tests/unit -q
```
