# CarbonSight Architecture

Readable map of the SkyNomad + carbon-aware stack. Paths are relative to the repo root. Please checkout repo root or this will be confusing :)

---

## 1. Product shape

You have a GPU training job and a deadline. AWS will sell you that GPU in ~21 regions,
either as **spot** (cheap, can be killed at any moment) or **on-demand** (expensive, safe),
and every region sits on a grid with a different carbon intensity that changes hour to hour.

That's ~42 ways to run the same job, each trading **dollars**, **carbon**, and
**interruption risk** differently — and the right answer changes while the job is running,
as grids get dirtier, spot capacity evaporates, and the deadline gets closer.

CarbonSight is **not an advisor**. You hand it a job; it scores every placement with a single
number, $U_s$, **launches on the winner, supervises the run, and moves the job when the
winner changes.** You watch status; you don't get asked to pick.

```
  submit ──► rank ──► launch ──► supervise ──► migrate/idle ──► done
                ▲                     │
                └──── re-rank ◄───────┘   every probe interval (~2h)
```

### 1.1 What goes in — [`JobSpec`](carbonsight/packages/core/carbonsight_core/models.py)

| Group | Fields | Why the scheduler cares |
|-------|--------|--------------------------|
| **The work** | `gpu_type`, `gpu_count`, `duration_hours`, `cpu_count`, `mem_gib`, `gpu_utilization` | Feeds the power model → watts → MWh/hr → kg CO₂/hr. `gpu_utilization` (or `--nvidia-smi`) replaces sampled draw with a measured one. |
| **The clock** | `deadline_hours` (T), `progress_hours_done` (p), `start_time_utc` | Sets deadline pressure θ = (P−p)/(T−t). Falling behind raises the value of an hour, which makes the ranker accept pricier, safer placements. |
| **Interruption cost** | `checkpoint_size_gb`, `cold_start_minutes` (d), `current_region` (r₀) | A restart costs egress on the checkpoint plus dead time restoring it. `current_region` pays no migration, so it gets a stickiness bonus. |
| **Your preference** | `carbon_price_usd_per_ton`, `carbon_weight` | Converts kg CO₂ into dollars so carbon competes with price in one unit. `0` ignores carbon; higher values buy greener grids. |
| **Data movement** | `data_in_gb`, `data_out_gb`, `constraints` | Transfer cost and hard filters (region allow-lists, compliance). |

### 1.2 The pipeline

```
JobSpec ── what to run · when it's due · how far along · what you'll pay for carbon
   │
   ├─► SpotPriceProvider ──────► $/hr for spot and on-demand, per region
   │     (static | boto3)
   │
   ├─► CarbonIntensityProvider ► kg CO2/hr, per region, averaged over [t, t+Lbar]
   │     (API | WattTime | synthetic)
   │
   ├─► AvailabilityTracker ────► LifetimeStats ──► Lbar  how long spot survives here
   │     (probe history)          (Nelson-Aalen)
   │
   └─► ProgressState ──────────► V  what one hour of progress is worth right now
         (p, P, t, T)
                    │
                    ▼
        CandidateState[]   one per (region x {spot, on_demand}), plus one idle
                    │
                    ▼
              rank by U_s          <- see 1.3
                    │
      ┌─────────────┼─────────────┐
      ▼             ▼             ▼
  schedule    recommendations   advise
      │
      ▼
  SkyNomadPolicy ──► launch | stay | idle | terminate
```

### 1.3 What "rank by $U_s$" means

$U_s$ is a **single dollars-per-hour score** answering one question per candidate:
*is running here for the next hour worth what it costs?* Every term is \$/hr, so they subtract cleanly.

$$
U_s = \underbrace{V \cdot \eta}_{\text{value earned}} - \underbrace{C_{\mathrm{total}}}_{\text{price + carbon}} - \underbrace{E / \bar L}_{\text{move cost, spread out}}
$$

| Term | Reads as | Detail |
|------|----------|--------|
| $V$ | what an hour of progress is worth | $C_{od}\cdot\theta/\tilde\theta$ — anchored so that **on schedule → $V$ = cheapest on-demand rate**. Fall behind and $V$ climbs, so pricier/riskier options start winning. |
| $\eta$ | fraction of the instance actually spent working | $(\bar L - d)/\bar L$. A 3-hour spot slot with a 20-min restore only gives you ~89% useful time. On-demand has $\bar L = \infty$, so $\eta = 1$. |
| $C_{\mathrm{total}}$ | the real hourly bill | spot \$/hr **+** carbon priced in: `kg/hr ÷ 1000 × carbon_price × carbon_weight`. |
| $E / \bar L$ | migration, amortized | $E$ = egress \$/GB × checkpoint GB, paid once. Divided by $\bar L$ because a slot you'll hold 12h absorbs it far better than one you'll hold 3h. Zero for `current_region`. |

**Highest $U_s$ wins.** Idle scores exactly `0`, so a negative $U_s$ everywhere means *nothing is worth
paying for right now* and the scheduler waits.

Two rules short-circuit the ranking before it runs:

- **Thrifty** — `p >= P`: the job is done. Release the instance.
- **Safety net** — `T - t < P - p + 2d`: too little slack left to risk another spot restart.
  Skip ranking and take the cheapest on-demand region that can still finish in time.

**Worked example** — real output, not hand-constructed. Reproduce with:

```python
from datetime import UTC, datetime
from carbonsight_core.models import JobSpec
from carbonsight_core.spot.scheduler_service import ranked_as_json, schedule_job

job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0, deadline_hours=45.0,
              checkpoint_size_gb=100.0, cold_start_minutes=6.0, current_region="us-east-1")
res = schedule_job(job, now=datetime(2026, 7, 31, 12, 0, tzinfo=UTC))
```

(`carbonsight schedule --deadline-hours 45 --checkpoint-size-gb 100 --current-region us-east-1
--json` is the same thing at the current wall clock; `now` is pinned here only so the table is
stable.) No credentials, so synthetic MOER and static prices. 43 rows — 21 regions ×
{spot, on-demand} plus idle. $V = 4.104$: on schedule at $t=0$, so $V$ lands exactly on the
cheapest on-demand total. Top four, plus the `us-east-*` pair:

| Rank | Region | Mode | $\bar L$ | $\eta$ | \$/hr | kg/hr | $E/\bar L$ | $U_s$ |
|---|---|---|---|---|---|---|---|---|
| 1 | ap-northeast-1 | spot | 14.70h | 0.980 | 1.72 | 0.289 | \$0.136 | **2.148** |
| 2 | ap-south-1 | spot | 8.65h | 0.965 | 1.58 | 0.090 | \$0.231 | 2.148 |
| 3 | eu-west-1 | spot | 9.66h | 0.969 | 1.65 | 0.086 | \$0.207 | 2.115 |
| 4 | us-east-1 | spot | 2.13h | 0.859 | 1.43 | 0.084 | \$0.000 | 2.086 |
| 8 | us-east-2 | spot | 3.65h | 0.918 | 1.43 | 0.164 | \$0.548 | 1.776 |

Three things this shows that a carbon-only ranking cannot:

- **Egress outweighs a real effectiveness advantage.** `us-east-1` and `us-east-2` have
  *identical* \$1.435/hr spot. `us-east-2` actually earns **more** value per hour — its longer
  3.65h lifetime lifts $V\eta$ by \$0.242 — but the \$0.548/hr amortized egress on a 100 GB
  checkpoint it would have to move swamps that, for a net \$0.310 deficit. Incumbency is a
  priced term, not a tiebreak.
- **Lifetime beats headline price.** `us-east-1` is the cheapest region on the board and still
  loses to `ap-northeast-1` at \$1.72/hr: a 2.13h expected lifetime against a 0.28h cold start
  throws away 14% of every slot ($\eta = 0.859$), where `ap-northeast-1` throws away 2%.
- **Carbon is the weakest lever at \$50/t, but it is live.** Ranks 1 and 2 sit within 0.0004 of
  each other even though `ap-south-1` is 3.2× cleaner (0.090 vs 0.289 kg/hr) — worth only
  \$0.010/hr here, so which of the two leads flips with the time of day. Raise
  `--carbon-price` to \$200/t and `ap-south-1` takes first outright; by \$500/t the top three
  are `ap-south-1`, `eu-west-1`, `us-east-1` and `ap-northeast-1` has dropped out.

The best on-demand row scores exactly `0.000`, tying with idle. That is the $V$ anchoring
working as designed: for a job on schedule, paying on-demand rates is precisely break-even.

### 1.4 Decide → execute → supervise → report

The ranker is one stage of a closed loop, not the product. The loop is:

| Stage | Does what | Today |
|-------|-----------|-------|
| **Rank** | Score every region×mode by $U_s$ | ✅ `schedule`, `scheduler_service` |
| **Decide** | Thrifty / safety-net / launch / stay / migrate | ✅ `SkyNomadPolicy` emits actions |
| **Execute** | Patch YAML, `sky launch` / `sky jobs launch`, quota preflight | ⚠️ **only via `run`** — one-shot, carbon-only pick, never re-evaluated |
| **Supervise** | Re-rank on a cadence, act on the new decision | ❌ **not built** — nothing consumes `SkyNomadPolicy`'s actions |
| **Migrate** | Checkpoint → egress → relaunch elsewhere → resume | ⚠️ checkpoint/restore primitives exist; the region-to-region move does not |
| **Report** | Live job status while running | ❌ **not built** — `history`/`dashboard`/`report` read *finished* runs only |

The two gaps that matter:

**`schedule` doesn't launch.** It ranks and prints. `carbonsight run` is the only path that
actually calls SkyPilot, and it decides once, on the old carbon-only ranking, and never
revisits that choice. Closing this means `schedule` becomes the launcher and inherits `run`'s
executor (`launch_skypilot_with_patched_yaml`, quota preflight, checkpoint wrapping).

**The ledger can't answer "what's my job doing?"** [`tracking.py`](carbonsight/packages/core/carbonsight_core/tracking.py)'s
`runs` table is written once, after launch, for cost/carbon accounting. It has no `status`,
no current region over time, no progress $p$, no migration count. A live status view needs
those columns plus a supervisor writing to them on every probe — at which point
`carbonsight status` can show: where the job is now, hours in vs. deadline, $p/P$, how many
times it moved, cumulative \$ and kg, and what the next probe will consider.

---

## 2. SkyNomad modules (`carbonsight_core/spot/`)

Paper: [SkyNomad arXiv:2601.06520](https://arxiv.org/pdf/2601.06520).

| Section | Module | Responsibility |
|---------|--------|----------------|
| 4.3 Availability | [`availability.py`](carbonsight/packages/core/carbonsight_core/spot/availability.py) | `SpotObservation`, `VirtualInstance`, `AvailabilityTracker`: probe trace → 0→1/1→0 transitions → lifetimes with censoring |
| 4.4 Lifetime | [`lifetime.py`](carbonsight/packages/core/carbonsight_core/spot/lifetime.py) | Nelson–Aalen `compute_at_risk`, $h=e/n$, $H=\sum h$, $S=e^{-H}$, `expected_remaining` ($\bar L$), $\gamma^*$ volatility adjustment |
| 4.5 Progress | [`progress.py`](carbonsight/packages/core/carbonsight_core/spot/progress.py) | `ProgressState(p,P,t,T)`: `deadline_pressure`, `avg_progress`, `future_progress_value`, `is_thrifty`, `is_safety_net` |
| 4.6 Unified | [`unified_model.py`](carbonsight/packages/core/carbonsight_core/spot/unified_model.py) | `CandidateState`, `MigrationCostEstimator`, $U_s = V\eta - C_{total} - E/\bar L$ (all \$/hr) |
| 4.7 Policy | [`policy.py`](carbonsight/packages/core/carbonsight_core/spot/policy.py) | `PolicyState`, `Action`, `SkyNomadPolicy`: thrifty → safety net → probe → rank → Δ |
| Orchestration | [`scheduler_service.py`](carbonsight/packages/core/carbonsight_core/spot/scheduler_service.py) | `build_candidates`, `schedule_job` over every region the registry maps (21 AWS today) |

Carbon is an **extra lever** inside $C_{total}$:
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
- `SyntheticCarbonProvider` — 17 deterministic grid regions, each with its own MOER curve
- `get_carbon_provider()` preference: **API URL > WattTime env > synthetic**

### 3.3 API routes

| Route | File |
|-------|------|
| `GET /v1/carbon/forecast` | [`routes/carbon.py`](carbonsight/apps/api/carbonsight_api/routes/carbon.py) |
| `GET /v1/regions` | [`routes/regions.py`](carbonsight/apps/api/carbonsight_api/routes/regions.py) — registry contents only |
| `GET/POST /v1/recommendations` | [`routes/recommendations.py`](carbonsight/apps/api/carbonsight_api/routes/recommendations.py) — one row per mapped region, greenest first |
| Lifespan warm | [`main.py`](carbonsight/apps/api/carbonsight_api/main.py) |

### 3.4 Worker + DB

- Worker: [`apps/worker/fetch_watttime.py`](carbonsight/apps/worker/fetch_watttime.py) populates cache + optional `grid_signal_cache`
- Schema: [`infra/schemas.sql`](carbonsight/infra/schemas.sql) (`grid_signal_cache` + retention delete in worker)
- Compose: [`infra/docker/docker-compose.yml`](carbonsight/infra/docker/docker-compose.yml) — API on **8001**, worker service

### 3.5 Time-weighted MOER

Integrate MOER over $[t, t+\bar L]$. If the window extends beyond the forecast, **hold the last value**. Mixture mappings blend series by weight (`mixture_weighted_moer`).

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
