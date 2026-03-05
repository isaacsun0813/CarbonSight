# CarbonSight MVP – Implementation Plan

This plan follows the [CarbonSight Technical Design for Execution](https://docs.watttime.org/) and breaks the MVP into ordered phases with clear deliverables and acceptance criteria.

---

## Overview

**Goal:** Ship a CLI-first tool that advises the greenest AWS region for an ML job (using WattTime MOER + mapping registry), runs AWS quota preflight, and launches via SkyPilot.

**Stack:** Python 3.11+, Typer/Rich/Pydantic (CLI), FastAPI (minimal API), Postgres, WattTime v3 API, SkyPilot, AWS (Service Quotas + EC2/spot).

**Repo shape:** Monorepo with `apps/cli`, `apps/api`, `packages/core`, `infra`, `tests`.

---

## Testing Strategy: E2E Only + Live Replayer

We do **not** use unit tests. We verify the system only with **end-to-end tests**: real CLI and API flows. External HTTP (WattTime, and optionally AWS) uses a **live replayer**: record real API traffic once, then replay it so E2E runs in CI without credentials while still using real response shapes.

**Live replayer (record/replay)**

- **Tool:** [vcr.py](https://vcrpy.readthedocs.io/) with [pytest-recording](https://pytest-recording.readthedocs.io/) (or vcr.py directly). Records HTTP to YAML cassettes; replays them when the same requests are made.
- **Record (live run on our end):** Run E2E with real credentials (WattTime in `.env`; AWS if testing quota). First run hits real APIs; vcr writes request/response to e.g. `tests/e2e/cassettes/`. Filter sensitive headers (e.g. `Authorization`) in vcr config so cassettes are safe to commit.
- **Replay (default in CI):** Run E2E without credentials. vcr uses `record_mode=none` or `once`: replay from cassettes only, no network. CI and anyone without credentials get deterministic, real-shaped behavior.
- **Refresh cassettes:** When the API or our usage changes, re-record: run E2E with credentials and `--record-mode=once` or `--record-mode=new_episodes`. Do this locally or in a secure CI job; commit updated cassettes.

**Can we run the live replayer on our end?** Yes. Locally: set `.env` with WattTime (and optionally AWS), run `pytest tests/e2e/ -v --record-mode=once` to record or refresh. In CI: run `pytest tests/e2e/ -v` (replay only, no secrets). Optional: a scheduled job with secrets that re-records and commits cassettes.

**Where E2E tests live**

- `tests/e2e/` — all E2E tests: CLI flows (`advise`, `run --dry-run`, `mappings validate`, `backtest run`), API flows (POST recommendations, GET runs, etc.). Each test runs the real application; HTTP is recorded/replayed via vcr.
- `tests/e2e/cassettes/` — vcr cassette files (YAML). One per test or per logical flow. Redact secrets via vcr `filter_headers` / `filter_post_data`.
- `tests/fixtures/` — YAML job specs, minimal SkyPilot YAML, seed data. Consumed by E2E only.

**How to run**

- **Replay (no credentials):** `pytest tests/e2e/ -v` — uses existing cassettes; safe for CI.
- **Record or refresh (with credentials):** `pytest tests/e2e/ -v --record-mode=once` or `--record-mode=new_episodes` — requires WattTime (and AWS if those calls are recorded) in env.
- **Lint/types only:** `ruff check .`; `mypy packages/core apps/cli apps/api`.

In each phase below, **Testing** describes only **E2E scenarios** and how they use the replayer.

---

## Phase 0: Repo and Foundations (Week 0)

**Goal:** Monorepo layout, tooling, and shared types so all later work has a single place to land.

| Task | Details |
|------|--------|
| **0.1** Create monorepo layout | `carbonsight/` root with `pyproject.toml`, `apps/cli/`, `apps/api/`, `packages/core/`, `infra/`, `tests/`. Use a single `pyproject.toml` with optional deps or workspace-style layout (e.g. `pip install -e packages/core`, `-e apps/cli`). |
| **0.2** Tooling | Ruff + mypy, pre-commit or CI lint. Python 3.11+ in CI. |
| **0.3** Core types | In `packages/core/carbonsight_core/models.py`: `JobSpec` (gpu_type, gpu_count, duration_hours, cpu_count, mem_gib, data_in/out_gb, start_time_utc, constraints), `EstimateResult` (cloud, cloud_region, watttime_regions, expected_cost_usd, expected_co2_kg_mean/p10/p90, mapping_confidence, notes). Use Pydantic. |
| **0.4** Config and env | Load from env (WattTime credentials, CARBON_PROVIDER, LOG_LEVEL, etc.). No secrets in config files. |

**Testing (Phase 0)** — E2E only. No tests yet; “done” = lint and mypy pass, and a single E2E smoke that imports the app and runs `carbonsight --help` (no HTTP). Optional: one E2E that builds a `JobSpec` from a fixture YAML and asserts it parses; no WattTime. **How to run:** `ruff check .`; `mypy packages/core`; `pytest tests/e2e/ -v` (smoke only). **Done when:** Layout exists, types load, smoke E2E passes.

---

## Phase 1: WattTime Client v3 (Weeks 1–2)

**Goal:** Correct, robust WattTime v3 client: auth, token cache, and all MVP endpoints with correct units and error handling.

| Task | Details |
|------|--------|
| **1.1** Auth and token cache | POST `/login` (basic auth). Cache bearer token; refresh on 401. Respect rate limits (3,000/5min, 100/5min for login). |
| **1.2** Discovery | GET `/v3/my-access`: list regions, signal types, models. Use for validation and for knowing what's available. |
| **1.3** Region-from-loc | GET `/v3/region-from-loc?signal_type=co2_moer&latitude=...&longitude=...`. Core for mapping. |
| **1.4** MOER signals | Implement clients for: `/v3/forecast`, `/v3/historical` (where plan allows), and `/v3/signal-index` as fallback. Parse and expose `units` (require `lbs_co2_per_mwh` for kg CO₂). |
| **1.5** Error and rate-limit handling | 401 → re-login and retry once. 429 → backoff. Timeouts and retries (e.g. 15s timeout, 3 retries). |

**Testing (Phase 1)** — E2E with replayer. (1) **Record:** Run E2E that invokes the real WattTime client (login → my-access → region-from-loc for one coord → forecast or signal-index for one region). Use real credentials; vcr records to a cassette. Assert response has `units=lbs_co2_per_mwh` and that the client returns usable data. (2) **Replay:** Run the same E2E without credentials; vcr replays the cassette. Assert same behavior (no network). (3) **401 refresh:** E2E that triggers 401 then retry (either record a live 401+retry sequence or use a pre-made cassette that contains it); assert second request succeeds. **How to run:** Record: `pytest tests/e2e/test_watttime_flow.py -v --record-mode=once` (with `.env`). Replay: `pytest tests/e2e/test_watttime_flow.py -v`. **Done when:** E2E passes in replay mode; cassette contains real WattTime response shapes and units; no kg CO₂ from unsupported units.

---

## Phase 2: Mapping Registry + Drift (Weeks 1–2, parallel or after Phase 1)

**Goal:** Three-layer mapping (cloud region → site coords → WattTime region), confidence scoring, and automated drift checks.

| Task | Details |
|------|--------|
| **2.1** Data model | Tables/schemas: `cloud_region` (provider, region_code, display_name, country, source_url), `cloud_site` (site_id, provider, region_code, lat, lon, source_type, last_verified_at), `grid_mapping` (site_id, signal_type, wt_region, validated_at, confidence_subscores). Support mixture: one cloud region → multiple (wt_region, weight). |
| **2.2** Registry loader | Load from JSON/DB: cloud regions → site candidates → WattTime regions. Resolve mixture weights. Expose `resolve(cloud, region_code) -> MappingResult` (wt_regions list, confidence). |
| **2.3** Confidence scoring | S_source, S_geo, S_wt_stability, S_recency in [0,1]. Overall confidence = 0.35*S_source + 0.25*S_geo + 0.25*S_wt_stability + 0.15*S_recency. Label: High ≥0.85, Medium 0.60–0.84, Low &lt;0.60. Policy: High default-in, Medium with warning, Low exclude unless opt-in. |
| **2.4** Coordinate → WattTime | For each site coordinate, call `/v3/region-from-loc`. Build/update mapping; support "all same region" vs "mixture" (store weights). |
| **2.5** Drift and revalidation | Job/script: (1) WattTime region existence via `/v3/my-access`. (2) Optional: `/v3/maps` last_updated; if changed, trigger full revalidation. (3) For each site coord, call `/v3/region-from-loc`; if result differs from stored, mark stale, lower confidence, optionally open PR or write diff. |
| **2.6** Seed data | Curated seed: at least AWS eu-north-1, us-east-1, GCP europe-north1, Azure norwayeast (per design examples). Store as JSON or seed DB. |

**Testing (Phase 2)** — E2E with replayer. (1) **E2E: resolve and confidence:** Run `carbonsight advise --yaml tests/fixtures/train_minimal.yaml --json` (or a small script that calls registry.resolve + WattTime). HTTP to WattTime is replayed from cassettes (recorded in Phase 1 or a dedicated mapping cassette). Assert output includes at least one region, with mapping_confidence and watttime_regions; for a region with mixture (if any), assert weights sum to 1. (2) **E2E: mappings validate:** Run `carbonsight mappings validate` with replayed WattTime; assert exit 0 and output contains diff/confidence or "ok". If cassette includes a "drift" scenario (one coord returns different region), assert confidence drop or diff. **How to run:** `pytest tests/e2e/test_advise_with_registry.py tests/e2e/test_mappings_validate.py -v` (replay). Record: same with `--record-mode=once` and credentials. **Done when:** E2E advise shows resolved mappings and confidence; mappings validate runs and reflects drift when cassette encodes it; mixture mappings work in E2E.

---

## Phase 3: Carbon Estimator (Weeks 2–4)

**Goal:** Power model + PUE + MOER integration + Monte Carlo uncertainty; outputs point estimate and band (e.g. p10/p90).

| Task | Details |
|------|--------|
| **3.1** Power model | Additive: P_IT = P_base + P_gpu + P_cpu + P_mem + P_net. GPU: utilization-based (e.g. P_gpu_idle + u*(P_cap - P_idle)); defaults for T4, A100, H100. CPU: sockets × (idle + u*(TDP - idle)). Mem: mem_gib × W_per_GiB. P_base 150–300W per node. Configurable defaults. |
| **3.2** PUE | Default 1.20; overrides by cloud/region. For uncertainty: PUE ~ Normal(μ=1.20, σ=0.05) truncated [1.05, 1.60]. |
| **3.3** Time-window and MOER | Nowcast (latest MOER), forecast-integrated (over job window), or historical backfill. E_facility_MWh = (P_IT_W/1000)*duration_hours*PUE/1000. CO2_lb = Σ_t (E_facility_MWh_t × MOER_lb_per_MWh(t)). CO2_kg = CO2_lb × 0.45359237. |
| **3.4** Mixture mapping in estimator | For (wt_region, weight), fetch MOER per region; weighted sum for MOER at each time step. |
| **3.5** Monte Carlo uncertainty | N=1000 samples: sample u_gpu, PUE, P_base, and if mixture sample region by weights. Return mean, p10, p90 (or p05/p95). |
| **3.6** Unit gate | Refuse to compute kg CO₂ unless WattTime returns `units=lbs_co2_per_mwh` (or another explicitly supported unit); fail closed with clear error. |

**Testing (Phase 3)** — E2E with replayer. (1) **E2E: advise returns CO₂ and band:** Run `carbonsight advise --yaml tests/fixtures/train_minimal.yaml --json`. Replay WattTime cassettes. Assert each recommendation has expected_co2_kg_mean, expected_co2_kg_p10, expected_co2_kg_p90, and p10 &lt; mean &lt; p90 (or narrow band when inputs are fixed). (2) **E2E: unit gate:** Use a cassette that contains a response with wrong or missing `units` (or a small E2E that injects that response once); assert the CLI or API returns a clear error and does not output kg. **How to run:** `pytest tests/e2e/test_advise_carbon_band.py tests/e2e/test_estimator_units_gate.py -v`. **Done when:** E2E advise shows CO₂ mean and band; unit-gate E2E fails closed on bad units.

---

## Phase 4: CLI MVP (Weeks 3–4)

**Goal:** `advise`, `run`, `mappings validate`, `backtest run` with Rich tables and JSON mode.

| Task | Details |
|------|--------|
| **4.1** Parse job from YAML + args | Parse SkyPilot-style YAML; extract or override GPU type/count, duration, CPU, memory. Build `JobSpec`. |
| **4.2** `carbonsight advise --yaml train.yaml [--explain] [--json]` | For each candidate region: resolve mapping, get MOER, run estimator, get cost (next phase). Rank by score (carbon + cost weights). Output: ranked table (region, expected CO2 kg, cost, confidence, notes); with `--explain` show top drivers; with `--json` machine-readable. |
| **4.3** Cost model | Use SkyPilot pricing (or cached prices) + optional AWS spot. Integrate into scoring: score = w_carbon * normalized(co2) + w_cost * normalized(cost). |
| **4.4** `carbonsight run train.yaml [--dry-run] [--no-exec]` | Run advise → preflight (Phase 5) → patch YAML with chosen region → if not dry-run/no-exec, call SkyPilot to launch. Emit patched YAML when dry-run. |
| **4.5** `carbonsight mappings validate` | Run drift checks (region-from-loc vs stored); print diff and confidence drops. |
| **4.6** `carbonsight backtest run --days 180 --n 1000` | Generate 1000 synthetic workloads (GPU type/count, duration, start time). For each: decision-time view (forecast/nowcast + spot at t0), select region by score, compare to oracle (best realized). Output: savings vs baseline, regret distribution, rank accuracy (top-1/top-3). |

**Testing (Phase 4)** — E2E with replayer. (1) **E2E advise:** `carbonsight advise --yaml tests/fixtures/train_minimal.yaml --json` → exit 0, JSON parseable, ranked regions with cloud_region, expected_co2_kg_mean, mapping_confidence. With `--explain`, stdout contains explain/driver text. (2) **E2E run --dry-run:** `carbonsight run tests/fixtures/train_minimal.yaml --dry-run` → exit 0; stdout or out-file has patched YAML with `resources.region` set; no SkyPilot launch (subprocess not invoked). (3) **E2E mappings validate:** `carbonsight mappings validate` → exit 0; output has diff/confidence/ok. (4) **E2E backtest:** `carbonsight backtest run --days 1 --n 10` (small) with replayed WattTime (and optional AWS spot cassette); exit 0; output or artifact has carbon savings, regret, rank accuracy; same seed → same metrics. **How to run:** `pytest tests/e2e/ -v` (replay). Record: `pytest tests/e2e/ -v --record-mode=once` with credentials. **Done when:** All four E2E flows pass in replay; advise and run --dry-run produce expected output; backtest is reproducible.

---

## Phase 5: AWS Quota Preflight (Weeks 4–5)

**Goal:** Before launching, check GPU (and relevant) quotas per region; fail over to next region if insufficient.

| Task | Details |
|------|--------|
| **5.1** Service Quotas client | Use AWS Service Quotas API (ListServices, GetServiceQuota, etc.) for EC2/GPU-related quotas per region. |
| **5.2** Policy | If GPU quota is 0 or below requested GPUs, mark region "blocked by quota" and exclude from ranking (or try next in run flow). |
| **5.3** Caching | Cache quota per (account, region, quota_code) for ~15 minutes to avoid excessive API calls. |
| **5.4** Logging | Log reason for failover so UX can show "why" (e.g. "us-east-1 skipped: insufficient GPU quota"). Rate-limit multi-region enumeration. |

**Testing (Phase 5)** — E2E with replayer. (1) **Record AWS (optional):** E2E that runs `carbonsight run ...` with preflight enabled and real AWS; vcr records Service Quotas (and any EC2) HTTP if using a client that vcr can intercept; or record to a dedicated AWS cassette format. (2) **Replay:** E2E run with preflight; replay cassette so "quota insufficient" for one region is replayed; assert next region is chosen and log/reason mentions quota. (3) **E2E advise without preflight:** `carbonsight advise ...` with preflight disabled; assert no AWS calls (replay only WattTime). **How to run:** `pytest tests/e2e/test_run_preflight.py -v` (replay). **Done when:** E2E run with replayed "quota blocked" chooses next region and logs reason; advise without preflight does not require AWS.

---

## Phase 6: SkyPilot Execution (Weeks 5–6)

**Goal:** Patch SkyPilot YAML with chosen region and launch (or emit patched YAML only).

| Task | Details |
|------|--------|
| **6.1** YAML patching | Given chosen cloud + region, patch `resources.region` (and optionally `use_spot`) in the user's YAML. Preserve rest of spec. |
| **6.2** Launch | Invoke SkyPilot (e.g. `sky jobs launch` or equivalent) with patched YAML. Capture job id / run id. |
| **6.3** Run recording | Store run in Postgres: run_id, submitted_at, job_spec_json, chosen_region, estimated_co2_kg, estimated_cost_usd. Optional: link to SkyPilot job id. |

**Testing (Phase 6)** — E2E. (1) **E2E run --dry-run / --no-exec:** Already covered in Phase 4; assert patched YAML and no subprocess to `sky`. (2) **E2E run (real launch, optional):** In a dedicated env with SkyPilot and cloud creds, run `carbonsight run ...` without --dry-run; assert job submitted and run record exists in DB. This can be a manual or scheduled E2E; no vcr for SkyPilot (subprocess). (3) **E2E error handling:** Invalid YAML path or invalid region → clear error message in stdout. **How to run:** `pytest tests/e2e/test_run_dry_run.py tests/e2e/test_run_errors.py -v`. **Done when:** Dry-run E2E passes; optional real-launch E2E passes in secure env; error E2E passes.

---

## Phase 7: Minimal API + Postgres (Weeks 5–6, can overlap with 6)

**Goal:** FastAPI service and DB for recommendations, runs, regions, and mapping revalidation.

| Task | Details |
|------|--------|
| **7.1** Postgres schema | Create tables: cloud_region, cloud_site, grid_mapping, run, grid_signal_cache (optional). Add `infra/migrations` (e.g. raw SQL or Alembic). |
| **7.2** API endpoints | POST /v1/recommendations (body: job spec or YAML ref) → ranked list. POST /v1/runs → record run, return run_id. GET /v1/runs/{id} → status + estimates. GET /v1/regions → cloud regions + mapping confidence. POST /v1/mappings/revalidate → trigger drift job (async or sync). |
| **7.3** Docker | docker-compose: api (FastAPI + Uvicorn), db (Postgres). Optional worker for revalidation. CLI can run on host with env/secrets. |

**Testing (Phase 7)** — E2E. (1) **E2E API with replayer:** Start API (and Postgres via Docker or test container); E2E test: POST /v1/recommendations with job spec → 200, ranked list (WattTime replayed via vcr). POST /v1/runs → 201, run_id. GET /v1/runs/{id} → 200, status and estimates. GET /v1/regions → 200, regions with confidence. POST /v1/mappings/revalidate → 202 or 200. (2) **E2E DB persistence:** After POST /v1/runs, assert row in run table; GET returns it. (3) **Docker smoke (optional):** `docker-compose up`; E2E hits API and DB; assert recommendations and runs work. **How to run:** `pytest tests/e2e/test_api_recommendations.py tests/e2e/test_api_runs.py -v` (replay WattTime). **Done when:** E2E API flows pass; DB stores and returns runs; revalidate trigger works.

---

## Phase 8: Backtesting Harness (Weeks 6–8)

**Goal:** Reproducible backtest with 1000 workloads, WattTime historical + AWS spot data, and metrics (savings, regret, rank accuracy).

| Task | Details |
|------|--------|
| **8.1** Data ingest | Scripts to build: watttime_moer.parquet (wt_region, point_time, value, units, data_point_period_seconds, model_date), aws_spot_prices.parquet (region, instance_type, timestamp, price_usd_per_hour). workloads_seed.json for reproducibility. |
| **8.2** Simulation loop | For each of 1000 workloads: decision-time view (MOER forecast/nowcast + spot at t0), select region by score, compute oracle (best realized over [t0,t1]), compute regret and savings vs baseline. |
| **8.3** Metrics and report | Primary: carbon savings %, cost savings %, regret distribution, rank accuracy (top-1, top-3). Secondary: confidence calibration, sensitivity to weights. Output artifacts (CSV/JSON + optional plots). |

**Testing (Phase 8)** — E2E. (1) **E2E backtest (small):** `carbonsight backtest run --days 1 --n 20` (or equivalent) with fixture data or replayed WattTime/AWS cassettes; exit 0; output has carbon savings, cost savings, regret, rank accuracy. (2) **Reproducibility:** Same seed twice → same metrics. (3) **Regret ceiling (optional):** E2E with larger n/days and threshold assertion; mark slow, run nightly. **How to run:** `pytest tests/e2e/test_backtest.py -v`. **Done when:** E2E backtest produces metrics and is reproducible; regret ceiling E2E exists and can gate releases.

---

## Phase 9: Polish and CI (Ongoing / Week 7–8)

| Task | Details |
|------|--------|
| **9.1** Critical acceptance as E2E | (1) Mapping determinism: E2E that runs advise twice with same coords/cassette; same ranking. (2) Unit correctness: E2E unit-gate (Phase 3). (3) Boundary drift: E2E mappings validate with drift cassette. (4) Backtest regret ceiling: E2E Phase 8. |
| **9.2** CI | Lint (ruff, mypy). E2E only: `pytest tests/e2e/ -v` (replay). Optional: nightly or weekly re-record job with secrets to refresh cassettes. |
| **9.3** Docs | README: install, .env, advise/run/mappings validate/backtest examples. Mapping registry format. |

**Testing (Phase 9)** — All critical checks are E2E. (1) **E2E mapping determinism:** Same cassette, two runs of advise; assert consistent ranking. (2) **E2E unit gate:** Already in Phase 3. (3) **E2E boundary drift:** mappings validate with drift cassette. (4) **E2E regret ceiling:** backtest E2E with threshold. CI: `ruff check . && mypy . && pytest tests/e2e/ -v`. Optional: script or job to run `pytest tests/e2e/ -v --record-mode=once` with secrets and commit cassettes. **How to run:** `pytest tests/e2e/ -v`; full CI workflow. **Done when:** All E2E pass in CI (replay); optional re-record job documented; README up to date.

---

## Dependency Order (Summary)

```
Phase 0 (foundations)
    ↓
Phase 1 (WattTime client) ──┐
    ↓                       │
Phase 2 (mapping registry) ─┼──→ Phase 3 (estimator)
    ↓                       │         ↓
    └───────────────────────┘    Phase 4 (CLI: advise + run flow)
                                        ↓
                                  Phase 5 (AWS preflight)
                                        ↓
                                  Phase 6 (SkyPilot execution)
                                        ↓
Phase 7 (API + Postgres) ←─────────────┘ (can start after 2/3)
Phase 8 (backtest) ← after 3 + data ingest
Phase 9 (polish/CI) ← ongoing
```

---

## Suggested First Steps (This Repo)

1. Create `carbonsight/` monorepo layout and `pyproject.toml` with core, cli, api extras.
2. Add `packages/core/carbonsight_core/models.py` with `JobSpec` and `EstimateResult`.
3. Implement WattTime client in `packages/core` or a dedicated `packages/watttime` (login, token cache, `/v3/region-from-loc`, `/v3/my-access`, and one of forecast/historical/signal-index).
4. Add a minimal mapping registry (in-memory or JSON-backed) with 2–3 AWS regions and seed coordinates, then wire `region-from-loc` to fill WattTime region codes.
5. Implement power model + PUE + single-region MOER integration and one Monte Carlo pass to get mean + p10/p90.
6. Add `carbonsight advise` that: loads YAML → builds JobSpec → for each region gets mapping + MOER → runs estimator → ranks and prints table.

From there, add `run` (with preflight and SkyPilot patch), then Postgres + API, then backtest and CI.

---

## Quick reference: E2E and replayer

| Mode | Command | When |
|------|---------|------|
| Replay (CI, no secrets) | `pytest tests/e2e/ -v` | Every PR; default. |
| Record / refresh cassettes | `pytest tests/e2e/ -v --record-mode=once` | Local with `.env`; or secure CI job. |
| New episodes only | `pytest tests/e2e/ -v --record-mode=new_episodes` | Add new requests without re-recording existing. |
| Lint and types | `ruff check . && mypy packages/core apps/cli apps/api` | Every PR. |

**Full gate before merge:** `ruff check . && mypy . && pytest tests/e2e/ -v`

---

## References

- WattTime v3 API: https://docs.watttime.org/
- vcr.py: https://vcrpy.readthedocs.io/
- pytest-recording: https://pytest-recording.readthedocs.io/
- SkyPilot YAML and CLI: SkyPilot docs (YAML spec, `sky jobs launch`, etc.)
- Design doc: "CarbonSight Technical Design for Execution" (full truth table, estimator formulas, backtest design, security).
