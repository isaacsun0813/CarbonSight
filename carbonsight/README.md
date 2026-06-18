# CarbonSight (Python package)

CLI and library to **rank AWS regions by estimated operational CO₂** for a GPU job (WattTime marginal MOER + power model), with optional **SkyPilot** launch.

**“More efficient” here means lower estimated electricity-related emissions for the same job spec**, not faster wall-clock training. You still choose cost vs. carbon using `--max-cost-premium`.

## Who this is for

- Teams running **ML training on AWS** who want a **data-informed default** for **which region** to use.
- Anyone using **SkyPilot** who can point at a **YAML** describing accelerators and duration.

## Install

```bash
cd carbonsight
pip install -e .
```

Development (tests + linters):

```bash
pip install -e ".[dev]"
```

## Configure (your WattTime account)

Use **your own** [WattTime](https://www.watttime.org/) API credentials—copy [`.env.example`](.env.example) to `.env` or export:

```bash
export WATTTIME_USERNAME=...
export WATTTIME_PASSWORD=...
```

## Minimal flow

### Option A — YAML only

1. Write or reuse a **SkyPilot-style YAML** (`resources`, `duration`, `run:`). Example: [`../examples/skypilot/train.yaml`](../examples/skypilot/train.yaml).
2. **Advise** — ranked regions (greenest first, optional cost ceiling):

   ```bash
   carbonsight advise --yaml path/to/train.yaml --json
   ```

3. **Run** (optional) — pick greenest affordable region, patch YAML, call SkyPilot:

   ```bash
   carbonsight run path/to/train.yaml --dry-run   # see patched YAML
   carbonsight run path/to/train.yaml --yes       # needs `sky` CLI + cloud creds
   ```

### Option B — `train` (no YAML hand-authoring)

From your **project root** (so paths match what SkyPilot uploads):

```bash
carbonsight train path/to/train.py --json
carbonsight train path/to/train.py --launch --dry-run   # show patched YAML
carbonsight train path/to/train.py --launch --yes       # launch with SkyPilot
```

Defaults: `A100:1`, `1h`, 8 CPUs, 32 GiB. Override with `--accelerators`, `--duration`, `--cpus`, `--memory`, `--name`.

Optional: `--gpu-util 0.72` or `--nvidia-smi` to anchor GPU power; optional YAML block `carbonsight.gpu_utilization` when using `advise`/`run` with a file.

## Commands

| Command | Purpose |
|---------|---------|
| `carbonsight train SCRIPT.py [--json] [--launch ...]` | Build a SkyPilot task from `python SCRIPT.py`, then same as advise or run |
| `carbonsight advise --yaml FILE [--json] [--max-cost-premium P]` | Rank regions by CO₂ (and cost), filter expensive outliers |
| `carbonsight run FILE [--dry-run] [--no-exec] [--skip-preflight]` | Advise + optional AWS quota check + patch YAML + `sky launch` / `sky jobs launch` |
| `carbonsight mappings validate` | Compare registry coords to WattTime `region-from-loc` (needs credentials) |
| `carbonsight backtest run [--json]` | Synthetic MOER/price policy experiment |

## API (optional)

```bash
pip install -e ".[api]"
uvicorn carbonsight_api.main:app --reload
```

- `GET /health`
- `POST /v1/recommendations` — same inputs as a `JobSpec` (see OpenAPI at `/docs`)

## Tests

```bash
pytest tests/unit tests/e2e -q
```

Integration tests that call live WattTime **skip** without credentials. CI runs on every PR (see `.github/workflows/ci.yml`).

## Design docs

- Repo root [`ARCHITECTURE.md`](../ARCHITECTURE.md) — MOER, Monte Carlo, limitations  
- [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md) — phased scope  

## License

MIT — see [`../LICENSE`](../LICENSE).
