# CarbonSight (Python package)

CLI and library to **rank AWS regions by operational CO₂ + $ + deadline**, using WattTime MOER + power model + spot lifetime, now with **SkyNomad-inspired joint rank** and **central single-credential cache**.

## Install

```bash
cd carbonsight
uv sync --extra dev --extra api  # or pip install -e ".[dev,api]"
```

## Configure

**BYOK mode** (old, per-user WattTime):
```bash
export WATTTIME_USERNAME=...  WATTTIME_PASSWORD=...
```

**Central cache mode** (NEW, one cred for all):
```bash
export CARBONSIGHT_API_URL=http://127.0.0.1:8000
# server holds ONE WattTime cred, worker populates grid_signal_cache
# client needs NO WattTime env
```

## Flows

### advise (1D greenest-first)

```bash
carbonsight advise --yaml ../examples/skypilot/train.yaml --json
# greenest within --max-cost-premium 20%
```

### schedule (NEW joint 2D region×mode, live without creds)

```bash
carbonsight schedule --yaml tests/fixtures/train_minimal.yaml \
  --deadline-hours 45 --checkpoint-size-gb 100 --carbon-price 50

# Rank by U = V·η - C_total - E/Lbar
# C_total = spot $/hr + carbon kg/hr/1000*price, η=(Lbar-d)/Lbar, V=C_od·θ/θ̃
# Thrifty p>=P → idle, safety-net T-t<P-p+2d → cheapest OD
```

<details><summary>Example U ranking JSON</summary>

```json
{
  "cloud_region": "us-east-1",
  "mode": "spot",
  "mean_lifetime_hr": 12.0,
  "effectiveness_eta": 0.97,
  "price_per_hr_usd": 1.43,
  "carbon_kg_per_hr": 0.144,
  "total_cost_per_hr_usd": 1.44,
  "utility_U": 2.56,
  "V": 4.10
}
```
</details>

```bash
# Tight deadline → safety net
carbonsight schedule --yaml ... --deadline-hours 1 --json
# {"action":"safety_net_on_demand","chosen_region":"us-east-1"}
```

### run / train (existing)

```bash
carbonsight run path/to/train.yaml --dry-run
carbonsight train path/to/train.py --launch --dry-run
```

### backtest

```bash
carbonsight backtest run --json
carbonsight backtest spot --n 20 --days 14 --deadline-ratio 1.5 --checkpoint-gb 100 --json
# SkyNomad cost mean vs UP single vs UP multi, deadline_met%, migrations, egress%
```

## API (central ONE cred)

```bash
docker-compose -f infra/docker/docker-compose.yml up -d
# db + api:8000 + worker loop 15m (ONE WattTime cred)

curl http://127.0.0.1:8000/v1/regions | jq length # 21
curl "http://127.0.0.1:8000/v1/carbon/forecast?wt_region=PJM_DC" | jq .source
curl -X POST http://127.0.0.1:8000/v1/recommendations -d '{"gpu_type":"A100"}' | jq length # 17 even without creds (synthetic)
```

## How single credential works

- `watttime/cache.py:71` ForecastCache TTL 15min + time-weighted window `[t,t+Lbar]` mixture
- `providers/carbon.py:250` factory prefers `CARBONSIGHT_API_URL` (Api no creds) > `WATTIME_USERNAME` (ONE cred server) > synthetic 400
- Worker `fetch_watttime.py:44` dedupes 30 WT regions, horizon 72h, upserts `grid_signal_cache` ON CONFLICT
- `routes/carbon.py:180` central proxy reads cache → DB → live → empty not 500

Validate without creds: `WATTTIME_USERNAME= pytest tests/unit/test_central_cache_stub.py -v` → 5 passed

## File map (short pointers)

| What | File |
|------|------|
| Client token 25min 401/429 | `watttime/client.py:26` |
| ForecastCache | `watttime/cache.py:71` |
| Carbon one-cred provider | `providers/carbon.py:250` |
| Spot price isolated | `providers/spot.py:26` Protocol, `unified_model.py:55` Static |
| Availability Sec 4.3 | `spot/availability.py:105` |
| Lifetime Sec 4.4 | `spot/lifetime.py:80` h=e/n, S=exp(-H), Lbar |
| Progress V Sec 4.5 | `spot/progress.py:44` V=C_od·θ/θ̃ |
| Utility U Sec 4.6 | `spot/unified_model.py:258` |

More: `../ARCHITECTURE.md` (line pointers), `../CURRENT_STATE.md`

