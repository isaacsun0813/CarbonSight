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
- **Live AWS pricing (optional)**: `CARBONSIGHT_LIVE_AWS_PRICING=1` or `--live-pricing` uses EC2 spot history when `--spot`, and Pricing API on-demand when `--no-spot` (or `advise` without `--spot`). `--static-pricing` forces static tables. JSON may include `notes` like `cost:live_spot`, `cost:live_ondemand`, or `cost:static_*_fallback`.
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

- **Spot pricing**: `--spot` (default ON) sets `resources.use_spot: true` in the patched YAML. By default, cost uses static on-demand × 35% (`SPOT_PRICE_FRACTION`). With `CARBONSIGHT_LIVE_AWS_PRICING=1` or `--live-pricing`, spot cost uses EC2 `describe_spot_price_history`; on-demand (`--no-spot`) uses Pricing API `get_products`. Falls back to static tables on API errors. `--static-pricing` forces static tables.
- **Carbon-aware scheduling**: `--max-delay 6h` scans the forecast for the lowest-carbon start time within the delay window and prints a recommendation. Default `0h` (run immediately).
- **Checkpoint resilience**: `--checkpoint` (default ON) wraps the training command with automatic checkpoint saving and resume. Detects the ML framework (HuggingFace, Lightning, PyTorch) via AST analysis, mounts persistent SkyPilot Storage at `/ckpt`, and injects a shim that handles SIGTERM → SIGINT for graceful saves on preemption. Auto-enables `--managed` (SkyPilot managed spot) when spot + checkpoint are both on. Customizable with `--checkpoint-bucket` (explicit S3 URI) and `--checkpoint-interval` (steps, default 500). Disable with `--no-checkpoint`.
- **Run tracking**: each run is persisted to a SQLite ledger (`~/.carbonsight/runs.db` by default, overridable with `--db` or `CARBONSIGHT_DB`). Baseline is `us-east-1` on-demand/spot (matching `--spot`).
- **Dry run**: `--dry-run` prints patched YAML and exits
- **No exec**: `--no-exec` prints patched YAML and exits (after choosing region)
- **AWS preflight**: enabled by default (`--skip-preflight` to disable); intersects registry with account-enabled regions, checks GPU Service Quotas, and EC2 instance type offerings per region. Instance availability and enabled-regions checks extend `BaseAWSProvider` (`cloud/aws/base.py`); quota checks use raw boto3.
- **After a successful SkyPilot launch**: attempts to compute **actual CO₂** using **historical MOER** over the wall-clock run window and compares to the estimate.

Relevant code:

- CLI implementation: `carbonsight/apps/cli/carbonsight_cli/commands/run.py`
- Quota checker: `carbonsight/packages/core/carbonsight_core/preflight/quota.py`
- Instance availability: `carbonsight/packages/core/carbonsight_core/preflight/availability.py`
- Enabled regions: `carbonsight/packages/core/carbonsight_core/preflight/enabled_regions.py`
- Actual-run carbon: `carbonsight/packages/core/carbonsight_core/estimator/carbon_model.py`
- Scheduler: `carbonsight/packages/core/carbonsight_core/scheduler.py`
- Run ledger: `carbonsight/packages/core/carbonsight_core/tracking.py`

### 3) Generate a SkyPilot YAML from a Python script (`carbonsight train`)

`carbonsight train path/to/train.py` creates a temporary SkyPilot-style YAML (resources + duration + run command) and then:

- **Without `--launch`**: runs the same behavior as `advise`
- **With `--launch`**: runs the same behavior as `run`

Relevant code:

- CLI implementation: `carbonsight/apps/cli/carbonsight_cli/commands/train.py`

### 4) Mapping registry drift checks (`carbonsight mappings validate`)

- **Input**: mapping registry JSON (default seeded file, or `--registry`)
- **Behavior**: for each region, calls WattTime `region-from-loc` per site, builds mixture `wt_regions` (same logic as refresh), compares to stored values; reports **mapping confidence** per region
- **Output**: human `OK` / `DRIFT` lines with confidence, or `--json` with `status` (`ok`, `drift`, or `skipped`)
- **Exit codes**: `0` if no drift (or creds missing → skipped); `1` if drift detected
- **Credentials**: if WattTime creds are missing, exits `0` with `status: skipped`

Relevant code:

- Core: `carbonsight/packages/core/carbonsight_core/mapping/validate.py`
- CLI: `carbonsight/apps/cli/carbonsight_cli/commands/mappings.py`

### 4b) Refresh WattTime mappings (`carbonsight mappings refresh`)

- **Input**: mapping registry JSON (default seeded file, or `--registry`); requires WattTime credentials
- **Behavior**: calls `region-from-loc` per site, rebuilds `wt_regions` (including mixtures), sets `last_verified_at` and `s_recency` on success
- **Default**: dry-run (prints per-region diffs only)
- **`--write`**: persist JSON to `--registry` path or `--out` (refuses to overwrite bundled `seed_registry.json` without `--out`)

Relevant code:

- Core: `carbonsight/packages/core/carbonsight_core/mapping/refresh.py`
- CLI: `carbonsight/apps/cli/carbonsight_cli/commands/mappings.py`

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
  - `POST /v1/recommendations`: ranks regions from the server's shared forecast cache/Postgres store; returns `503` when carbon data is unavailable
  - `GET /v1/carbon/forecast`: serves cached WattTime forecast points for one grid region
  - `GET /v1/carbon/regions`: lists grid regions known to the shared carbon store
  - `GET /v1/regions`: returns regions from the registry plus confidence + WattTime mixture weights
  - `POST /v1/mappings/revalidate`: runs mapping drift validation (same core as CLI `mappings validate`); returns JSON with `status` (`ok` / `drift`); `503` without WattTime creds
  - `POST /v1/runs` / `GET /v1/runs/{run_id}`: in-memory “runs store” (no Postgres wired yet)

Relevant code:

- App wiring: `carbonsight/apps/api/carbonsight_api/main.py`
- Routes: `carbonsight/apps/api/carbonsight_api/routes/`

### 6) View run history (`carbonsight history`)

Lists all recorded runs from the SQLite ledger: ID, date, region, GPU, duration, spot flag, estimated CO₂ and cost.

### 7) Cumulative savings report (`carbonsight report`)

Prints total estimated CO₂/cost vs baseline (us-east-1), with absolute and percentage savings.

## What it does *not* do yet (important limitations)

- **No persistent DB for API**: API runs are stored in memory only (`_runs_store`), not Postgres. CLI runs are persisted in SQLite.
- **No mapping auto-refresh via API**: `POST /v1/mappings/revalidate` validates drift only; use CLI `mappings refresh --write` to update the registry JSON.
- **Not compliance-grade accounting**: this is an operational estimate from marginal emissions + a simplified power model.
- **Scheduling is advisory**: `--max-delay` recommends a start time but does not actually wait before launching.
- **Central historical backfill is not exposed**: `run` can launch from central forecast data, but post-run actual CO₂ requires direct WattTime credentials and is skipped otherwise.

## “Where to look in code”

- **CLI entrypoint**: `carbonsight/apps/cli/carbonsight_cli/main.py`
- **Core orchestration**: `carbonsight/packages/core/carbonsight_core/region_ranking.py`
- **Estimator**: `carbonsight/packages/core/carbonsight_core/estimator/`
- **Live AWS pricing**: `carbonsight/packages/core/carbonsight_core/estimator/aws_estimation/` (`SpotPriceProvider`, `OnDemandPriceProvider`; re-exported from `aws_estimation/__init__.py`)
- **AWS GPU instance catalog**: `carbonsight/packages/core/carbonsight_core/cloud/aws/gpu_catalog.py`
- **AWS provider base**: `carbonsight/packages/core/carbonsight_core/cloud/aws/base.py` (`BaseAWSProvider` — shared session/client for spot, on-demand, preflight)
- **Cloud protocols**: `carbonsight/packages/core/carbonsight_core/cloud/base.py`
- **Scheduler**: `carbonsight/packages/core/carbonsight_core/scheduler.py`
- **Run ledger**: `carbonsight/packages/core/carbonsight_core/tracking.py`
- **Checkpoint core**: `carbonsight/packages/core/carbonsight_core/checkpoint.py`
- **Checkpoint shim**: `carbonsight/packages/core/carbonsight_core/checkpoint_shim.py`
- **Mapping registry**: `carbonsight/packages/core/carbonsight_core/mapping/registry.py`
- **Mapping validate**: `carbonsight/packages/core/carbonsight_core/mapping/validate.py`
- **Mapping refresh**: `carbonsight/packages/core/carbonsight_core/mapping/refresh.py`
- **WattTime client**: `carbonsight/packages/core/carbonsight_core/watttime.py`
