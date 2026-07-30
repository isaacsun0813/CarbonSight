# CarbonSight — how it works (condensed v3)

**Goal:** pick lower-carbon AND cheaper AWS spot region under deadline. Combines WattTime MOER (lb/MWh), power model, spot $/GPU/hr, spot lifetime, migration cost. Evolved from 1D green ranking to SkyNomad-style `U = V·η - C_total - E/L̄`.

---

## Layout

```
carbonsight/
  packages/core/carbonsight_core/
    watttime/         client + central cache
    providers/        carbon + spot abstractions (single-credential design)
    estimator/        power + carbon math + pricing (pricing owned by other branch)
    spot/             SkyNomad: availability, lifetime, progress, utility, policy
    mapping/          seed_registry.json → wt_regions mixture
    scheduler.py      v1 lowest-carbon start
  apps/cli/commands/  advise (1D), schedule (2D joint), run, backtest, mappings
  apps/api/routes/    recommendations, regions, carbon (central proxy)
  apps/worker/        fetch_watttime.py single writer
  infra/              schemas.sql grid_signal_cache, docker-compose db+api+worker
```

Rule: domain logic in `carbonsight_core`, CLI/API call it.

---

## WattTime single-credential central cache (P0)

**Why one cred is enough:** forecast is same for all users. Old: N users × R regions API calls → 429. New: 30 WT regions × 1/15min = 120/h << 3000/5min limit.

**Flow:**
```
WattTime API ← ONE cred (server env WATTTIME_USERNAME)
  ↑  POST /v2/login token 25min [client.py:36], 401 re-login once 429 exp backoff [client.py:75]
  worker fetch_watttime.py dedupes ~30 regions [fetch_watttime.py:44], get_forecast horizon 72h [fetch_watttime.py:140], upsert ON CONFLICT grid_signal_cache [fetch_watttime.py:62]
  ↓
Postgres grid_signal_cache (schema in infra/schemas.sql) + ForecastCache in-memory TTL 15min [cache.py:71]
  ↓
API GET /v1/carbon/forecast?wt_region= [carbon.py:180] tries cache → DB [carbon.py:61] → live → empty not 500; /forecast/window [carbon.py:240] time-weighted [t,t+L̄]
  ↓
CLI CARBONSIGHT_API_URL=http://api:8001 → ApiCarbonProvider [carbon.py:113] httpx GET central, local weighting, NO WattTime env; else WattTimeCarbonProvider [carbon.py:74] direct+cache; else Synthetic 400 lb [carbon.py:193]; factory prefers API_URL > WTT creds > synthetic [carbon.py:250]
  POST /v1/recommendations now returns 17 rows without creds via synthetic fallback (TestClient proves) [recommendations.py:49-76]
```

Config: `config.py:21` adds `CARBONSIGHT_API_URL`, `CACHE_TTL_MIN`, `CARBON_PRICE_USD_PER_TON`.

Validation without creds:
```
WATTTIME_USERNAME= pytest test_central_cache_stub.py -v → 5 passed
TestClient /v1/regions →21, /v1/recommendations without creds →17
```

---

## Carbon + power math

```
kg/hr = (P_IT/1000 * PUE/1000 MWh/hr) × MOER(lb/MWh) × 0.45359237
  P_IT = P_base 200W + P_gpu (count*(idle+u*(cap-idle))) + P_cpu + P_mem + P_net
  GPU defaults T4 25-70W, A100 80-400W, H100 150-700W [power_model.py:13], power_it_w [power_model.py:58]
  PUE Normal(1.20,0.05) truncated [1.05,1.60] [power_model.py:69], sample_power_params [power_model.py:85]
  _time_weighted_moer(points,start,end) each point covers [t_i,next) clipped to window [carbon_model.py:24 + cache.py:40]
  Facility MWh/hr deterministic seed 0 for kg/hr calc [_facility_mwh_per_hour in carbon.py]

carbon $/hr = kg/hr/1000 × price_per_ton × weight [carbon.py:74-113 + unified_model.py:202]
```

Monte Carlo old: 1000 samples of u_gpu 0.6-0.9, PUE → mean/p10/p90 `estimate_region:93`.

Windowed new: `ForecastCache.get_time_weighted_moer(mixture,start,end)` weighted sum over mixture [cache.py:101] for `[now,now+L̄]` not just `data[0].value` old [carbon_model.py:81].

---

## Spot price isolation (for your boto3 wrapper branch)

`estimator/pricing.py` owned by other work: base dict [pricing.py:20] + multiplier [pricing.py:31] + SPOT_PRICE_FRACTION=0.35 [pricing.py:56] + `estimate_cost_usd:59`.

Isolation via Protocol, no edits:

- `providers/spot.py:26` `SpotPriceProvider` Protocol `get_spot_price_usd_per_gpu_hr`, `get_od_price`, `get_egress_per_gb`
- `providers/spot.py:63` `StaticSpotPriceProvider` read-only copy of pricing.py dicts (no mutation) — od=base*mult, spot=od*0.35
- `providers/spot.py:142` `Boto3SpotPriceProvider` stub: fetch EC2 DescribeSpotPriceHistory + file cache `~/.carbonsight/spot_prices.json` TTL, docstring for your wrapper to implement same protocol
- `spot/unified_model.py:44` also defines Protocol for backward compat, but now imports from providers

---

## Spot scheduling (SkyNomad Sec 4.3-4.7)

| Piece | File:line | Formula / behavior |
|-------|-----------|--------------------|
| **Probing** | `spot/availability.py:19` SpotObservation, `48` VirtualInstance, `105` AvailabilityTracker | outcome 0/1, `0→1` start, `1→0` = preemption, ends in 1 → censored, `is_available(region,at)` last observation, `to_dict` |
| **Lifetime** | `spot/lifetime.py:41` LifetimeStats, `80` n(l)=Σ≥l(e+c), `110` h=e/n, `150` H=Σh, `180` S=exp(-H), `210` L̄(a)=1/S(a) Σ_{>a}S, `260` γ_W=e_W/Σh(a), `290` γ*, `310` S̃=S^{γ*} | heavy-tailed + volatile periods, 20+29 tests |
| **Progress V(t)** | `spot/progress.py:44` ProgressState(p,P,t,T), `92` θ=(P-p)/(T-t), `80` θ̃=p/t fallback P/T, `110` V=C_od·θ/θ̃ anchoring V=C_od when balanced, `140` thrifty p≥P, `149` safety_net T-t<P-p+2d, `193` ODCandidate, `213` total cost C·(P-p+d)+E+carbon$/1000*price*weight, `253` select cheapest OD | deadline pressure monotonic, scale-invariant |
| **Utility** | `spot/unified_model.py:114` MigrationCostEstimator E=e·ckpt (0.02 $/GB default), `167` CandidateState, `202` carbon_cost, `217` C_total=spot$+carbon$, `237` η=max(0,L̄-d)/L̄ (∞→1), `258` U=V·η - C_total - E/L̄, `330` rank_candidates descending U | idle U=0, OD U=V-C, spot penalized d and E/L̄, anti-flap Δ=0.05 in policy |
| **Policy loop** | `spot/policy.py:20` PolicyState, `60` Action Launch/Idle/Stay/Terminate, `80` SkyNomadPolicy decide: thrifty→Idle, safety_net→cheapest OD, rank by U, launch if U_s>U_current+Δ | Algo1 paper |

---

## Joint 2D schedule command (live)

`apps/cli/commands/schedule.py:30` `_GENERALLY_AVAILABLE` synthetic L̄ map (us-east-1 12h etc else 3.5h volatile), `40` `run_schedule()`:

1. parse YAML `advise.py:60`, P=duration, T=deadline_hours default P*1.5, d=cold_start_minutes/60 + ckptGB*0.002
2. Registry + Config factory carbon_provider (Api/WattTime/Synthetic), spot_provider Static, migration_estimator
3. now→now+L̄ carbon kg/hr via `carbon_provider.get_kg_per_hr(wt_regions,now,now+L̄)` using cache time-weighted, migration_cost if region!=r0
4. CandidateState spot+OD per region + idle, C_od_total_min = min OD C_total, V=ProgressState(0,P,0,T).future_progress_value(C_od_total_min)
5. thrifty/safety_net → `rank_candidates` → Rich table or --json fields `cloud_region,mode,mean_lifetime_hr,eta,price_per_hr,carbon_kg_per_hr,carbon_cost,total_cost,migration,utility_U,V`

Validated without creds:

```
.venv/bin/carbonsight schedule --yaml tests/fixtures/train_minimal.yaml --deadline-hours 45 --checkpoint-size-gb 100 --carbon-price 50 --json
→ us-east-1 spot L̄12 η0.97 $1.43 kg0.144 U2.56 V4.10 top action Launch

--deadline-hours 1 → safety_net_on_demand us-east-1 (T-t < P-p+2d)

--carbon-price 0 → US cheapest wins, --carbon-price 20000 → green wins (need ~13700 $/ton to overcome $1 spot diff: facility 0.0006 MWh ×270 lb delta×0.453=0.073 kg/hr)
```

Registered in `main.py:18` `app.command("schedule")`.

---

## API / worker prod pieces

- `apps/api/main.py:12` registry_path parents[3], `8` includes carbon router
- `infra/docker/docker-compose.yml` db postgres:15-alpine healthcheck + schemas.sql init, api port 8000 env DATABASE_URL=postgresql://...@db:5432 + WTT creds, worker same build `command python -m apps.worker.fetch_watttime --loop --interval-min 15` restart unless-stopped

Backtest v2 planned: `backtest/spot_runner.py` synthetic traces 13 zones patterns generally/frequent/unavailable like paper Fig 2, 30h/45h deadline ckpt 100GB baselines UP single, UP(S) cheapest-failover, metrics savings, deadline_met%, migrations, egress%.

---

## Tests (how we validate vibe-coding without real creds)

```
uv sync --extra dev --extra api
uv run pytest tests/unit/test_watttime_cache.py tests/unit/test_carbon_provider.py tests/unit/test_central_cache_stub.py -v
  9+5 passed: time-weighted 00:30-01:30→150, TTL hit fetch_count 1, mixture 140, factory, kg/hr, green vs dirty lever

WATTTIME_USERNAME= pytest test_central_cache_stub.py -v  # no creds, stubbed diurnal 150 night 420 day, API 21 regions →17 recs synthetic

.venv/bin/carbonsight advise --yaml tests/fixtures/... --json → [] without creds (BYOK)
CARBONSIGHT_API_URL=http://... advise --json → 17 rows (central cache, no WTT creds) → everyone can schedule

.venv/bin/pytest tests/unit -v → 191 passed, e2e advise replay →4 passed via vcr cassettes
```

If doc and code disagree, code wins.

