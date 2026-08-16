# CarbonSight

**Pick a lower-carbon AWS region for GPU training** before you launch—using live **marginal grid emissions** (MOER) from [WattTime](https://www.watttime.org/) and a rough power model for your job.

This is **not** certified carbon accounting or offsets. It is an **operational estimate** to compare regions and reason about **electricity-related** CO₂ when you train in the cloud.

## Why this exists

Cloud ML training uses electricity; grids differ by hour and region. CarbonSight helps you **see greener options** (and cost tradeoffs) and, if you use [SkyPilot](https://skypilot.readthedocs.io/), **patch your job YAML and launch** in the region you choose.

## Repository layout

| Path | Purpose |
|------|---------|
| [`carbonsight/`](carbonsight/) | Python package (CLI, core library, optional API) |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the pipeline works (MOER, power model, Monte Carlo) |
| [`CURRENT_STATE.md`](CURRENT_STATE.md) | What the code can do right now (code-derived) |
| [`AGENTS.md`](AGENTS.md) | Contributor / AI agent guidelines |
| [`examples/`](examples/) | Minimal SkyPilot + training stub |
| [`LICENSE`](LICENSE) | MIT |

## Quick start

1. **WattTime account** — Sign up for API access ([WattTime](https://www.watttime.org/)); you will use **your own** credentials (bring-your-own-key).

2. **Install** (from `carbonsight/`):

   ```bash
   cd carbonsight
   pip install -e .
   ```

3. **Configure** — export `WATTTIME_USERNAME` and `WATTTIME_PASSWORD` (a [WattTime](https://www.watttime.org/) account of your own), or set `CARBONSIGHT_API_URL` to a CarbonSight API that holds the credential centrally. With neither set, CarbonSight refuses to rank rather than inventing numbers.

4. **Advise** — Rank regions for a SkyPilot-style YAML:

   ```bash
   carbonsight advise --yaml examples/skypilot/train.yaml --json
   ```

   Or point at a **Python file** (CarbonSight builds a temporary SkyPilot task):

   ```bash
   cd carbonsight
   carbonsight train ../examples/skypilot/train_stub.py --json
   ```

5. **Optional: launch** — With SkyPilot installed, `carbonsight run` or `carbonsight train script.py --launch` can patch `cloud`/`region` and call `sky launch` (see `carbonsight/README.md`).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome; keep domain logic in `carbonsight_core` and one ranking path for CLI + API.

## License

MIT — see [`LICENSE`](LICENSE).
