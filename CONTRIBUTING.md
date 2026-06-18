# Contributing to CarbonSight

Thanks for helping. The goal is a **small, correct** pipeline: registry → WattTime MOER → power model → ranked regions.

## Before you change code

- Read [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`AGENTS.md`](AGENTS.md).
- Put **domain logic** in `carbonsight/packages/core/carbonsight_core/`.
- Keep **one** ranking path: `AwsRegionRankingService` + `JobCarbonEstimator` (CLI and API should not fork the loop).

## Development setup

```bash
cd carbonsight
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Checks

From `carbonsight/`:

```bash
ruff check packages/core apps/cli apps/api
pytest tests/unit tests/e2e -q
```

Live WattTime integration tests skip without `WATTTIME_USERNAME` / `WATTTIME_PASSWORD` in the environment.

## Pull requests

- Keep diffs **focused** on the issue; avoid drive-by refactors.
- Update docs if you change CLI flags, env vars, or public JSON shapes.
- Do not commit secrets (`.env` is gitignored).

## Code of conduct

Be respectful and assume good intent. For serious conduct issues, contact the maintainers via the repo’s issue tracker or email listed on the profile.
