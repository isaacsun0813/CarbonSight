# CarbonSight — current state (condensed v3)

**What it is:** CLI + API to pick lower-carbon AND cheaper AWS spot region under deadline. Combines WattTime MOER (central cache, ONE cred), power model, spot price Protocol (static fallback, dynamic boto3 wrapper isolated), spot lifetime, migration cost, deadline value.

Core brain: `carbonsight/packages/core/carbonsight_core/` used by CLI and API.

## Flows live now

| Flow | What it does | Key code |
|------|--------------|----------|
| **1 advise (1D)** | greenest-first within max-cost-premium | `commands/advise.py:109 run_advise` parses YAML `60`, Registry `82`, Config check BYOK vs API_URL `145` ( [] without creds, 17 rows with API_URL), `region_ranking.py:17` |
| **2 run** | pick greenest + patch YAML + sky launch + post-run historical | `commands/run.py`, `scheduler.py:25` lowest-carbon start, `checkpoint.py` + shim |
| **3 train** | script -> temp YAML -> advise/run | `commands/train.py` |
| **4 mappings validate** | region_from_loc drift | `commands/mappings.py`, `watttime/client.py:88` |
| **5 backtest run** | synthetic MOER/price, savings, regret, rank accuracy | `commands/backtest.py`, `backtest/runner.py:18` |
| **6 schedule (NEW joint 2D)** | `schedule --yaml --deadline-hours 45 --checkpoint-size-gb 100 --carbon-price 50 --json` -> region x mode ranking by `U=V·eta - C_total - E/Lbar` | `commands/schedule.py:40` builds candidates: synthetic Lbar map `30`, providers `200`, V anchoring `300`, rank `340`, safety_net `320` / thrifty; `spot/availability.py:105`, `lifetime.py:41`, `progress.py:44`, `unified_model.py:114-330` |
| **7 central cache (NEW P0)** | ONE WattTime cred serves all | `watttime/client.py:26` token 25min 401/429, `watttime/cache.py:71` ForecastCache TTL 15min + mixture `101`, `providers/carbon.py:34 Protocol /74 WattTime /113 Api (GET /v1/carbon/forecast) /193 Synthetic /250 factory`, `routes/carbon.py:180` GET forecast central proxy + `240` window, `routes/recommendations.py:49` fallback to synthetic 17 rows without creds, `worker/fetch_watttime.py:44` dedup 30 regions + upsert `62` + loop `200`, `docker-compose.yml` db+api+worker |

## API

`uvicorn carbonsight_api.main:app --reload` or `docker-compose -f infra/docker/docker-compose.yml up -d`

- `GET /health`
- `POST /v1/recommendations` — now 17 rows without creds via synthetic (was []), real when worker populated
- `GET /v1/regions` — 21 regions, confidence + mixture
- `GET /v1/carbon/forecast?wt_region=&horizon_hours=72` — central proxy, source cache/live/empty
- `GET /v1/carbon/forecast/window?start=&end=` — time-weighted MOER for [t,t+Lbar]
- `POST /v1/mappings/revalidate`, `POST /v1/runs` / `GET /v1/runs/{id}` (in-memory)

## Validation without WattTime creds

```
pytest test_central_cache_stub.py -v # 5 passed stubbed diurnal 150 night 420 day, API 21->17 recs, green vs dirty flip at 20000 $/ton
pytest test_watttime_cache.py test_carbon_provider.py -v # 9 passed
pytest tests/unit -v # 191 passed
advise --json # [] BYOK mode
CARBONSIGHT_API_URL=http://... advise --json # 17 rows via central cache
schedule --deadline-hours 45 --checkpoint-size-gb 100 --json # top us-east-1 spot U2.56
--deadline-hours 1 -> safety_net_on_demand
backtest spot --n 5 --days 2 --json # cost mean, deadline_met 100%
```

## Not yet

- `spot/policy.py` full Algo1 loop while p<P with probe every 2h + Delta 0.05 (schedule does single decision now)
- `models.py` extensions deadline_hours etc still CLI-only not JobSpec
- `providers/spot.py` exists 374 lines but unified_model still defines own Static fallback
- `backtest/spot_runner.py` implemented but needs real WT traces
- `tracking.py` no spot_observations table yet
- UV workspace still setuptools + PYTHONPATH hack, not hatchling + uv workspace
- Advise CLI not yet using ApiCarbonProvider for windowed queries

## File map

| Change | File |
|--------|------|
| WattTime client | `watttime/client.py:26` |
| ForecastCache central | `watttime/cache.py:71` |
| Carbon providers one-cred | `providers/carbon.py:250` factory |
| Spot price isolation | `providers/spot.py:26` Protocol, Unified:55 Static |
| Availability Sec 4.3 | `spot/availability.py:105` |
| Lifetime Sec 4.4 | `spot/lifetime.py:80-310` |
| Progress V(t) Sec 4.5 | `spot/progress.py:44` |
| Utility U Sec 4.6 | `spot/unified_model.py:258` |
| Joint schedule live | `commands/schedule.py:40` |
| Central API + worker | `routes/carbon.py:180`, `worker/fetch_watttime.py:44`, `docker-compose.yml` worker |

