# CarbonSight

**Multi-lever GPU training scheduler**: pick *where* and *when* to run so you minimize **cost + carbon + risk** under a **deadline**.

Built on [SkyNomad](https://arxiv.org/pdf/2601.06520) (multi-region spot scheduling) plus a **carbon lever** from [WattTime](https://www.watttime.org/) marginal operating emissions rates (MOER).

This is an **operational estimate**, not certified LCA or offsets.

---

## Why ONE WattTime credential (central cache)

Personal WattTime keys do not scale for a team CLI. CarbonSight uses a **server-side credential + forecast cache**:

```
CLI / clients                     API (one WattTime login)
───────────────                   ─────────────────────────
CARBONSIGHT_API_URL  ──GET──►  /v1/carbon/forecast
                               ForecastCache (TTL 15 min)
                               WattTimeClient  ──or── synthetic fallback
```

| Who | Needs |
|-----|--------|
| **API server** | `WATTTIME_USERNAME` / `WATTTIME_PASSWORD` (optional; synthetic if absent) |
| **CLI users** | only `CARBONSIGHT_API_URL=http://localhost:8001` |

Factory order in [`get_carbon_provider()`](carbonsight/packages/core/carbonsight_core/providers/carbon.py):

1. `CARBONSIGHT_API_URL` → `ApiCarbonProvider`
2. `WATTTIME_*` → `WattTimeCarbonProvider`
3. else → `SyntheticCarbonProvider` (deterministic 17 regions)

---

## Joint ranking \(U_s\)

For each candidate region/instance:

\[
U_s = V \cdot \eta - C_{\mathrm{total}} - \frac{E}{\bar L}
\]

| Symbol | Meaning | Code |
|--------|---------|------|
| \(\theta = (P-p)/(T-t)\) | Urgency (progress rate needed) | [`progress.py`](carbonsight/packages/core/carbonsight_core/spot/progress.py) |
| \(V = C_{od}\cdot\theta/\tilde\theta\) | Value of finishing | [`unified_model.py`](carbonsight/packages/core/carbonsight_core/spot/unified_model.py) |
| \(\eta\) | Instance efficiency | `ODCandidate.efficiency` |
| \(C_{\mathrm{total}}\) | Spot $ + carbon $ (MOER × MWh × \$/t) | same |
| \(E\) | Eviction penalty × (1 − survival) | same |
| \(\bar L\) | Expected remaining lifetime | [`lifetime.py`](carbonsight/packages/core/carbonsight_core/spot/lifetime.py) |

Rank by **\(U_s\) descending**. Greener grids raise \(U_s\) when `carbon_price_usd_per_ton` is material.

---

## Quick start (`uv`)

```bash
cd carbonsight
uv sync --all-extras
export CARBONSIGHT_API_URL=   # empty → synthetic
uv run carbonsight schedule \
  --yaml tests/fixtures/train_minimal.yaml \
  --deadline-hours 45 --checkpoint-size-gb 100 --carbon-price 50 --json
```

API (synthetic without creds):

```bash
uv run uvicorn carbonsight_api.main:app --port 8001
curl -s localhost:8001/v1/regions | head
curl -s "localhost:8001/v1/carbon/forecast?region=CAISO_NORTH" | head
curl -s localhost:8001/v1/recommendations | jq length   # → 17
```

---

## Layout

| Path | Role |
|------|------|
| [`carbonsight/packages/core/carbonsight_core/`](carbonsight/packages/core/carbonsight_core/) | Domain brain |
| [`spot/`](carbonsight/packages/core/carbonsight_core/spot/) | SkyNomad availability, lifetime, progress, policy, \(U_s\) |
| [`watttime/`](carbonsight/packages/core/carbonsight_core/watttime/) | Client + `ForecastCache` (time-weighted MOER) |
| [`providers/`](carbonsight/packages/core/carbonsight_core/providers/) | Carbon + spot Protocols (pricing isolated) |
| [`apps/cli`](carbonsight/apps/cli/) | `schedule`, `advise`, `backtest`, … |
| [`apps/api`](carbonsight/apps/api/) | Central cache proxy |
| [`apps/worker`](carbonsight/apps/worker/) | Periodic WattTime → cache/DB |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Deep dive + code pointers |

**Do not edit** base price dicts in [`estimator/pricing.py`](carbonsight/packages/core/carbonsight_core/estimator/pricing.py). Spot prices go through [`StaticSpotPriceProvider`](carbonsight/packages/core/carbonsight_core/providers/spot.py) (read-only snapshot).

---

## License

MIT — see [`LICENSE`](LICENSE).
