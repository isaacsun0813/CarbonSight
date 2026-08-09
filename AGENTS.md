# AGENTS.md — instructions for AI agents and contributors

Read this before making non-trivial changes. The goal is a **small, correct pipeline** that stays easy to change.

---

## 1. Read first (source of truth)

| Order | What | Why |
|--------|------|-----|
| 1 | [`ARCHITECTURE.md`](ARCHITECTURE.md) | What the product does, data flow, terminology (MOER, PUE), known limitations. |
| 2 | [`CONTRIBUTING.md`](CONTRIBUTING.md) | Setup, tests, PR expectations. |
| 3 | [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Phased scope and testing intent (when present). |
| 4 | `carbonsight/packages/core/carbonsight_core/` | Domain logic must live here. |

If docs and code disagree, **fix the doc or the code in the same PR**—do not leave them diverged.

---

## 2. Architecture rules (maintainability)

**Single brain.** Carbon math, WattTime usage, registry resolution, and region ranking policy belong in **`carbonsight_core`**. The CLI and API should **call** that layer, not reimplement loops or formulas.

**One orchestration path for rankings.** Use **`AwsRegionRankingService`** (and `JobCarbonEstimator` inside it) so CLI and API stay aligned. Do not duplicate “for each region…” logic in multiple places.

**Two axes, two packages.** `cloud/` answers *what does this compute cost and can I get it*; `carbon/` answers *how dirty is the power*. Anything provider-specific lives under its provider — all AWS code is in **`cloud/aws/`**, so `cloud/gcp/` would be a drop-in implementing the same Protocols in `cloud/base.py`. `watttime/` is a plain API client with no policy; source selection lives above it in `carbon/providers.py`. Don't scatter one provider across several trees.

**Never fabricate data to paper over a failure.** Synthetic MOER curves are hashes of the region name — arbitrary against real grids (Sweden scores dirtier than India). They exist so the tool demos with no credentials. A source that is *configured but broken* must **raise**, not silently substitute them: a fabricated ranking is indistinguishable from a real one and can be inverted. Same rule anywhere else a fallback is tempting — if you can't tell the user the number is made up, don't give them the number. See `ARCHITECTURE.md` § “Synthetic MOER is a demo, never a fallback”.

**One implementation per piece of physics.** There was briefly a second, subtly different time-weighted MOER function; it under-reported actual emissions by up to 28% and inflated the headline savings. Before adding a helper, grep for one that already exists.

**Boundaries.**

- **`carbonsight_core`**: no Typer/Rich/FastAPI imports in domain modules. Config/env loading via existing `config` patterns is OK.
- **CLI**: parsing, UX, subprocess (e.g. SkyPilot), optional host probes (e.g. `nvidia-smi`).
- **API**: HTTP shapes, request validation, wiring to core.

**Changes should be minimal.** Touch only what the task needs. No drive-by refactors, no “cleanup” in unrelated files, no new markdown files unless the user asked for documentation.

**Tests.** Prefer extending existing tests in `carbonsight/tests/`. New behavior should have a test when it’s easy to pin (pure math, parsing, ranking invariants). Follow the repo’s pytest layout and `PYTHONPATH` conventions documented in `ARCHITECTURE.md`.

**Honesty in outputs.** This tool estimates **operational electricity carbon** from marginal grid signals—it is not certified LCA, offsets, or compliance-grade without more work. Do not imply certification in copy or defaults.

---

## 3. Style and conventions

- Match **naming, typing, and structure** of the file you edit.
- Prefer **clear data** (`JobSpec`, `EstimateResult`) and **small functions** over deep inheritance.
- **Secrets**: environment variables only; never commit credentials or raw tokens in cassettes/tests.

---

## 4. What to avoid

- Large “clean OOP” rewrites that don’t change behavior.
- Second implementations of MOER integration or Monte Carlo outside `carbon_model` / `power_model`.
- Tight coupling from `carbonsight_core` to AWS/SkyPilot—keep those at app boundaries.

---

## 5. Path to the first paying user (product, not code)

Paying users usually buy **outcomes and trust**, not repository elegance.

**Ideal first buyer profile.** A team that already runs **GPU training on AWS** (or will), cares about **sustainability reporting or brand**, and has **budget for tools or infra**—e.g. ML platform, FinOps, or a startup with a public climate story.

**First wedge (keep scope narrow).**

1. **CLI + YAML** they can run in minutes (`advise`, then optional `run` with SkyPilot)—proof of value without your hosted infra.
2. **One honest page**: what you estimate (marginal grid MOER + power model), what you don’t (network, embodied hardware, compliance), and how to interpret bands.
3. **Credibility**: link methodology to WattTime; show example output; invite technical questions.

**What to charge for (examples).**

- **Support / setup** for their registry and CI (“greenest region” in a pipeline).
- **Team license** or **annual** for priority support and updates.
- Later: **hosted API** or **org dashboard** if you want recurring SaaS—only after someone pays for the core story.

**GTM that fits this codebase.**

- Publish the repo, a **short README** demo, and a **contact** (email or form) for teams wanting help integrating.
- Talk to **ML infra and platform** folks (Slack communities, SkyPilot/AWS ML circles), not generic “ESG software” buyers first—they move faster and understand the limits.

Agents should **not** invent pricing promises or compliance claims in code or user-facing strings unless product owners explicitly add them.

---

## 6. Checklist before you finish a change

- [ ] Core logic stays in `carbonsight_core` with boundaries respected.
- [ ] No duplicated ranking/carbon paths between CLI and API.
- [ ] Tests or a clear reason why not (e.g. docs-only).
- [ ] Docs updated if behavior or flags changed.
- [ ] No secrets; no unrelated files in the diff.
