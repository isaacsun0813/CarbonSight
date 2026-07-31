# CarbonSight Architecture

Readable map of the SkyNomad + carbon-aware stack. Paths are relative to the repo root.

---

## 1. Product shape

```
JobSpec (GPU, deadline, carbon $/t)
        │
        ▼
┌───────────────────┐     ┌────────────────────┐
│ SpotPriceProvider │     │ CarbonIntensity    │
│ (static / boto3)  │     │ Provider (API /    │
└─────────┬─────────┘     │  WattTime / synth) │
          │               └─────────┬──────────┘
          ▼                         ▼
   ODCandidate[]  ←  LifetimeStats + AvailabilityTracker + ProgressState
          │
          ▼
   rank by U_s  →  schedule / recommendations / advise
          │
          ▼
   SkyNomadPolicy  (PROBE / RUN / MIGRATE / WAIT)
```

---

## 2. SkyNomad modules (`carbonsight_core/spot/`)

Paper: [SkyNomad arXiv:2601.06520](https://arxiv.org/pdf/2601.06520).

| Section | Module | Responsibility |
|---------|--------|----------------|
| 4.3 Availability | [`availability.py`](carbonsight/packages/core/carbonsight_core/spot/availability.py) | `SpotObservation`, `VirtualInstance`, `AvailabilityTracker`, at-risk set, eviction counts |
| 4.4 Lifetime | [`lifetime.py`](carbonsight/packages/core/carbonsight_core/spot/lifetime.py) | Kaplan–Meier style `compute_at_risk`, hazard, cumulative hazard, survival, `expected_remaining` (Lbar) |
| 4.5 Progress | [`progress.py`](carbonsight/packages/core/carbonsight_core/spot/progress.py) | `ProgressState`: \(P,p,T,t\), `theta()`, `urgency()` |
| 4.6 Unified | [`unified_model.py`](carbonsight/packages/core/carbonsight_core/spot/unified_model.py) | `ODCandidate`, \(U_s = V\eta - C_{total} - E/\bar L\) |
| 4.7 Policy | [`policy.py`](carbonsight/packages/core/carbonsight_core/spot/policy.py) | `Action` enum, `SkyNomadPolicy`, probe every 2h + Δ migrate |
| Orchestration | [`scheduler_service.py`](carbonsight/packages/core/carbonsight_core/spot/scheduler_service.py) | Build candidates, `schedule_job`, 17-row synthetic pad |

Carbon is an **extra lever** inside \(C_{total}\):  
`spot_cost + (carbon_kg/1000) * carbon_price * carbon_weight`, with  
`carbon_kg = MWh * MOER_lb * 0.45359237`.

---

## 3. Central WattTime cache (P0)

### 3.1 Package [`watttime/`](carbonsight/packages/core/carbonsight_core/watttime/)

| File | Role |
|------|------|
| [`client.py`](carbonsight/packages/core/carbonsight_core/watttime/client.py) | Login, token refresh, `get_forecast`; **synthetic fallback** without creds |
| [`cache.py`](carbonsight/packages/core/carbonsight_core/watttime/cache.py) | `ForecastCache`, TTL **900s**, `get_moer(region, …, lbar_hours)` time-weighted average; mixture blend; fill-forward past last point |
| [`__init__.py`](carbonsight/packages/core/carbonsight_core/watttime/__init__.py) | Re-exports |

### 3.2 Providers [`providers/carbon.py`](carbonsight/packages/core/carbonsight_core/providers/carbon.py)

- `ApiCarbonProvider` — `GET {CARBONSIGHT_API_URL}/v1/carbon/forecast?region=`
- `WattTimeCarbonProvider` — direct client + cache
- `SyntheticCarbonProvider` — 17 deterministic regions
- `get_carbon_provider()` preference: **API URL > WattTime env > synthetic**

### 3.3 API routes

| Route | File |
|-------|------|
| `GET /v1/carbon/forecast` | [`routes/carbon.py`](carbonsight/apps/api/carbonsight_api/routes/carbon.py) |
| `GET /v1/regions` | [`routes/regions.py`](carbonsight/apps/api/carbonsight_api/routes/regions.py) — never empty |
| `GET/POST /v1/recommendations` | [`routes/recommendations.py`](carbonsight/apps/api/carbonsight_api/routes/recommendations.py) — **17 rows** synthetic-capable |
| Lifespan warm | [`main.py`](carbonsight/apps/api/carbonsight_api/main.py) |

### 3.4 Worker + DB

- Worker: [`apps/worker/fetch_watttime.py`](carbonsight/apps/worker/fetch_watttime.py) populates cache + optional `grid_signal_cache`
- Schema: [`infra/schemas.sql`](carbonsight/infra/schemas.sql) (`grid_signal_cache` + retention delete in worker)
- Compose: [`infra/docker/docker-compose.yml`](carbonsight/infra/docker/docker-compose.yml) — API on **8001**, worker service

### 3.5 Time-weighted MOER

Integrate MOER over \([t, t+\bar L]\). If the window extends beyond the forecast, **hold the last value**. Mixture mappings blend series by weight (`mixture_weighted_moer`).

---

## 4. Spot price isolation

Parallel work may add boto3 EC2 spot history. This stack **must not** edit:

- `_GPU_BASE_PRICE_USD_PER_HR`
- `_REGION_MULTIPLIER`
- `SPOT_PRICE_FRACTION`

in [`estimator/pricing.py`](carbonsight/packages/core/carbonsight_core/estimator/pricing.py).

Instead:

| Type | Location |
|------|----------|
| `SpotPriceProvider` Protocol | [`providers/spot.py`](carbonsight/packages/core/carbonsight_core/providers/spot.py) |
| `StaticSpotPriceProvider` | same — **canonical**; snapshots dicts at init |
| `Boto3SpotPriceProvider` | stub + cache + static fallback |
| Re-export only | `spot/unified_model.py` imports Static from providers (no duplicate class) |

---

## 5. CLI

| Command | Module |
|---------|--------|
| `carbonsight schedule --yaml … --deadline-hours … --json` | [`commands/schedule.py`](carbonsight/apps/cli/carbonsight_cli/commands/schedule.py) |
| `carbonsight advise` | [`commands/advise.py`](carbonsight/apps/cli/carbonsight_cli/commands/advise.py) — uses `get_carbon_provider` / joint path without creds |
| `carbonsight backtest run --spot` | [`commands/backtest.py`](carbonsight/apps/cli/carbonsight_cli/commands/backtest.py) + [`spot_runner.py`](carbonsight/packages/core/carbonsight_core/backtest/spot_runner.py) |

Config knobs: [`config.py`](carbonsight/packages/core/carbonsight_core/config.py) — `CARBONSIGHT_API_URL`, `CACHE_TTL_SECONDS=900`, `CARBON_PRICE_USD_PER_TON`.

Job extensions: [`models.py`](carbonsight/packages/core/carbonsight_core/models.py) — `deadline_hours`, `checkpoint_size_gb`, `cold_start_minutes`, `carbon_price_usd_per_ton`.

---

## 6. Package manager

- **`uv`** at `carbonsight/` (`uv sync`, `uv run`, `.venv/bin/pytest`)
- Build: **hatchling** in [`pyproject.toml`](carbonsight/pyproject.toml)
- Commit `uv.lock` when generated

---

## 7. Testing without secrets

```bash
cd carbonsight
uv sync --all-extras
uv run python /tmp/validate_cache.py
.venv/bin/pytest tests/unit/test_watttime_cache.py \
  tests/unit/test_carbon_provider.py \
  tests/unit/test_central_cache_stub.py -v
```

Carbon lever unit test: `test_watttime_provider_green_vs_dirty_lever` uses **\$20 000/t** so MOER spread flips rank (facility MWh is intentionally tiny).

---

## 8. Honesty / risks

- Operational electricity carbon only (MOER × power model).
- WattTime ToS: central proxy may need a paid/org tier for redistribution; demos use **synthetic** when unpaid.
- Synthetic Lbar + small MWh ⇒ need high \$/t to flip ranks in tests.
