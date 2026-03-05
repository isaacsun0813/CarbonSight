# advise.py: Full Guide (Understanding, Best Practices, Speed)

## 1. What This Command Does (One Sentence)

**`carbonsight advise`** reads a SkyPilot job YAML, estimates CO₂ per cloud region via WattTime, ranks regions (greenest first), and prints a table or JSON.

---

## 2. Data Flow (Mental Model)

```
YAML file → JobSpec (GPU, duration, etc.)
                ↓
Registry (region list + WattTime mappings)
                ↓
For each AWS region: estimate_option(job, region, WattTime) → EstimateResult
                ↓
Sort by CO₂ mean → output table or JSON
```

- **JobSpec**: what you're running (GPU type/count, duration). Comes from the YAML.
- **Registry**: list of cloud regions and their WattTime grid mappings. Loaded from JSON.
- **estimate_option**: core function that calls WattTime API and returns one `EstimateResult` per region.
- **EstimateResult**: one row (region, CO₂ mean/p10/p90, confidence).

---

## 3. Section-by-Section (Code + Why)

### 3.1 Imports

- **`pathlib.Path`**: Use instead of string paths. Gives you `.exists()`, `.read_text()`, `/` for joining, and works across OSes. **Best practice**: prefer `Path` over `os.path` and raw strings.
- **`yaml.safe_load`**: Never use `yaml.load` without a Loader; `safe_load` avoids arbitrary code execution from untrusted YAML. **Best practice**: always `safe_load` for untrusted or external YAML.
- **Typer**: CLI framework; options are declared with types and help text. **Best practice**: type hints on CLI options give validation and better help.

### 3.2 `_job_spec_from_yaml(path)`

**Purpose**: Turn a SkyPilot YAML into a single `JobSpec` object.

- **`data = yaml.safe_load(path.read_text()) or {}`**  
  Read file once, parse YAML, default to `{}` so missing/empty file doesn't crash. **Best practice**: defensively default so callers don't need to handle None.

- **`resources.get("accelerators") or ""`**  
  Use `.get()` to avoid KeyError; `or ""` so we always have a string for parsing. **Best practice**: use `.get()` for optional dict keys; normalize to a single type (here, str).

- **Accelerator parsing**  
  Two formats: `"A100:1"` (type:count) and `"1x A100"` (count type). The code handles both; defaults (1 GPU, A100) when missing or unparseable. **Best practice**: document supported formats in a docstring; have safe defaults so bad input doesn't crash.

- **Duration parsing**  
  `duration` is read once in the refactor to avoid repeated `data.get("duration", ...)`. **Best practice**: read a value once into a variable instead of repeating the same get/expression; handle "1h" vs "30m" (hours vs minutes) in one place.

- **Return `JobSpec(...)`**  
  Pydantic model: validates types and constraints. **Best practice**: use a single canonical type (JobSpec) so the rest of the app doesn't depend on raw dicts.

### 3.3 `_default_registry_path()`

- **`Path(__file__).resolve().parents[4]`**  
  Goes from this file up to repo root (4 levels: commands → carbonsight_cli → cli → apps → carbonsight). **Best practice**: resolve paths from `__file__` so they work regardless of current working directory; comment why `parents[4]` so future readers understand.

### 3.4 `advise()` – main command

- **Fail fast on missing input**  
  Check `yaml_path.exists()` and registry path before doing work. **Best practice**: validate inputs at the top; exit with a clear message and non-zero code (`typer.Exit(1)`).

- **Registry fallback**  
  If default path doesn't exist, try CWD-relative candidates. Using `next((c for c in candidates if c.exists()), None)` is a clear, one-expression way to pick the first existing path. **Best practice**: one obvious way to express “first match”; avoid manual loop + break when a comprehension or `next()` fits.

- **Credentials check**  
  If WattTime env vars are missing, return early: `[]` for `--json`, else a human message. **Best practice**: don’t make API calls when you know they’ll fail; keep JSON output consistent (empty list) for scripts.

- **Loop over regions**  
  Only consider entries that have `wt_regions` and `provider == "aws"`. **Best practice**: filter once per entry; keep the loop simple; handle errors per region so one bad region doesn’t kill the whole run.

- **Confidence score**  
  The formula `0.35*s_source + 0.25*s_geo + ...` is the same as in the registry. **Best practice**: DRY – import or reuse one implementation so the formula lives in a single place.

- **Exceptions**  
  Catching `WattTimeError` then `Exception` lets you treat API errors explicitly; both are handled the same here. **Best practice**: catch specific exceptions when you care about the type; use `err=True` so warnings go to stderr and don’t pollute JSON on stdout.

- **Sort**  
  `results.sort(key=lambda r: r.expected_co2_kg_mean)` sorts in place (no new list). **Best practice**: in-place sort is memory-efficient; key is a single attribute so lambda is fine (or `operator.attrgetter("expected_co2_kg_mean")` if you prefer).

- **Output**  
  JSON: one `json.dumps` over a list of `model_dump(mode="json")`. Table: build once, print once. **Best practice**: lazy-import Rich only when not `--json` to keep JSON path fast and avoid pulling in Rich for scripted use; alternatively, top-level import is simpler and more conventional.

---

## 4. Speed and Cleanliness (What We Optimized)

| Area | Change | Why |
|------|--------|-----|
| Duration | Read `data.get("duration")` once into `duration_raw` | Fewer dict lookups and string operations. |
| Registry fallback | `next((c for c in candidates if c.exists()), None)` | One expression, no loop variable, stops at first hit. |
| Confidence | Reuse `_confidence` from registry (or local helper) | Single source of truth; shorter loop body. |
| Accelerators | Optional: extract `_parse_accelerators(acc)` | Clearer and testable in isolation; same speed. |
| Rich | Top-level import | Cleaner (PEP 8); negligible cost unless you often use `--json`. |
| List build | Keep `results.append(res)` in loop | No need to preallocate; list growth is amortized O(1). |

---

## 5. Best Practices Summary

1. **Types**: Use `Path`, type hints, and Pydantic so the compiler and runtime help you.
2. **Defensive defaults**: `or {}`, `or ""`, default GPU/duration so bad/missing input doesn’t crash.
3. **Fail fast**: Validate paths and credentials up front; exit with clear messages.
4. **DRY**: Reuse the confidence formula; parse duration once.
5. **Single responsibility**: `_job_spec_from_yaml` only parses; `advise()` orchestrates and I/O.
6. **Stderr for chatter**: Use `err=True` for warnings so stdout stays clean for JSON.
7. **Safe YAML**: Always `yaml.safe_load` for untrusted or external YAML.

---

## 6. How to Run and Test

```bash
# From repo root (with .env set: WATTTIME_USERNAME, WATTTIME_PASSWORD)
carbonsight advise --yaml carbonsight/tests/fixtures/train_minimal.yaml

# JSON (for scripts)
carbonsight advise --yaml carbonsight/tests/fixtures/train_minimal.yaml --json

# With explanation line
carbonsight advise --yaml carbonsight/tests/fixtures/train_minimal.yaml --explain
```

You now understand the flow, the “why” behind the patterns, and what was tuned for clarity and speed.
