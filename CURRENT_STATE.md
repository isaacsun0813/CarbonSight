# CarbonSight — current state (code-derived)

This document is meant to answer: **“What can CarbonSight do right now?”** based on the code in this repo.

If this doc and code disagree, treat the **code as truth** and update this doc alongside code changes.

## What it is

CarbonSight is a Python CLI (and optional API) that helps you **choose a lower-operational-carbon AWS region for GPU jobs** by combining:

- **WattTime marginal emissions** (MOER) for grid regions (forecast + historical)
- A **GPU/CPU/memory power model** with **PUE** sampling (Monte Carlo uncertainty)
- A **mapping registry** from cloud regions → lat/lon → WattTime grid regions (supports mixtures with weights)
- A simple **cost estimate** and a **“max cost premium”** filter (pick greenest within budget)

Core domain logic lives in `carbonsight/packages/core/carbonsight_core/` and is used by both CLI and API.

## Primary user flows

### 1) Rank greener AWS regions for a SkyPilot YAML (`carbonsight advise`)

- **Input**: a SkyPilot-style YAML containing (at minimum) `resources.accelerators` and `duration`
- **Output**: a ranked list of AWS regions (greenest-first), filtered to those within your cost tolerance
- **Optional**: emit machine-readable JSON (`--json`)

Key behavior from the code:

- **Ranking policy**: sort by lowest `expected_co2_kg_mean`, then apply a **cost ceiling** based on `--max-cost-premium` vs the cheapest region.
- **GPU utilization**: you can fix `JobSpec.gpu_utilization` via:
  - `--gpu-util 0..1` (highest precedence)
  - `--nvidia-smi` (samples local `nvidia-smi` utilization)
  - YAML `carbonsight.gpu_utilization`
- **Registry required**: the CLI loads a mapping registry JSON (defaults to a seeded file if present).
- **Credentials required for recommendations**: if WattTime credentials aren’t in env, the CLI prints an empty list (`--json`) or a message.

Relevant code:

- CLI implementation: `carbonsight/apps/cli/carbonsight_cli/commands/advise.py`
- Ranking orchestration: `carbonsight/packages/core/carbonsight_core/region_ranking.py`

### 2) Pick a region, patch YAML, and optionally launch (`carbonsight run`)

`carbonsight run` does:

- Run the same estimate + ranking pipeline
- Choose the best region within the cost premium
- Patch the YAML with `resources.cloud` and `resources.region`
- Optionally invoke SkyPilot:
  - `sky launch` (default)
  - `sky jobs launch` (`--managed`)

Key behavior from the code:

- **Dry run**: `--dry-run` prints patched YAML and exits
- **No exec**: `--no-exec` prints patched YAML and exits (after choosing region)
- **AWS quota preflight**: enabled by default; can be skipped with `--skip-preflight`
- **After a successful SkyPilot launch**: attempts to compute **actual CO₂** using **historical MOER** over the wall-clock run window and compares to the estimate.

Relevant code:

- CLI implementation: `carbonsight/apps/cli/carbonsight_cli/commands/run.py`
- Quota checker: `carbonsight/packages/core/carbonsight_core/preflight/quota.py`
- Actual-run carbon: `carbonsight/packages/core/carbonsight_core/estimator/carbon_model.py`

### 3) Generate a SkyPilot YAML from a Python script (`carbonsight train`)

`carbonsight train path/to/train.py` creates a temporary SkyPilot-style YAML (resources + duration + run command) and then:

- **Without `--launch`**: runs the same behavior as `advise`
- **With `--launch`**: runs the same behavior as `run`

Relevant code:

- CLI implementation: `carbonsight/apps/cli/carbonsight_cli/commands/train.py`

### 4) Mapping registry drift checks (`carbonsight mappings validate`)

- **Input**: mapping registry JSON (default seeded file, or `--registry`)
- **Behavior**: for each region + site (lat/lon), calls WattTime `region-from-loc` and compares the returned region code to what’s stored
- **Output**: prints `DRIFT: ...` lines and a count, or `OK: no drift detected`
- **Credentials**: if WattTime creds are missing, it exits successfully and prints that drift checks are skipped

Relevant code:

- CLI implementation: `carbonsight/apps/cli/carbonsight_cli/commands/mappings.py`

### 5) Backtest harness (`carbonsight backtest run`)

- **Purpose**: a reproducible synthetic experiment that reports:
  - carbon savings vs a baseline region
  - cost savings vs baseline
  - regret (chosen vs oracle) and rank accuracy
- **Note**: current implementation uses **synthetic MOER and synthetic pricing**, not WattTime/AWS data.

Relevant code:

- CLI entrypoint: `carbonsight/apps/cli/carbonsight_cli/commands/backtest.py`
- Core implementation: `carbonsight/packages/core/carbonsight_core/backtest/runner.py`

## Optional API (FastAPI)

The API mirrors the CLI “advise” flow and exposes minimal endpoints.

- **Start**:
  - `uvicorn carbonsight_api.main:app --reload`
- **Endpoints**:
  - `GET /health`
  - `POST /v1/recommendations`: ranks regions like `carbonsight advise` (requires WattTime env vars; returns `[]` without them)
  - `GET /v1/regions`: returns regions from the registry plus confidence + WattTime mixture weights
  - `POST /v1/mappings/revalidate`: currently a stub trigger message (no worker wired)
  - `POST /v1/runs` / `GET /v1/runs/{run_id}`: in-memory “runs store” (no Postgres wired yet)

Relevant code:

- App wiring: `carbonsight/apps/api/carbonsight_api/main.py`
- Routes: `carbonsight/apps/api/carbonsight_api/routes/`

## What it does *not* do yet (important limitations)

- **No persistent DB**: API runs are stored in memory only (`_runs_store`), not Postgres.
- **No mapping auto-revalidation**: API `/mappings/revalidate` doesn’t actually revalidate; CLI `mappings validate` does live checks.
- **Not compliance-grade accounting**: this is an operational estimate from marginal emissions + a simplified power model.

## “Where to look in code”

- **CLI entrypoint**: `carbonsight/apps/cli/carbonsight_cli/main.py`
- **Core orchestration**: `carbonsight/packages/core/carbonsight_core/region_ranking.py`
- **Estimator**: `carbonsight/packages/core/carbonsight_core/estimator/`
- **Mapping registry**: `carbonsight/packages/core/carbonsight_core/mapping/registry.py`
- **WattTime client**: `carbonsight/packages/core/carbonsight_core/watttime.py`

