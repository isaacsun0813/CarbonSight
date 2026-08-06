# UP_PLAN.md

Replacing CarbonSight's home-grown spot scheduler with Uniform Progress (Wu et al., NSDI '24, "Can't Be Late"), and re-scoping carbon to region choice.

Paper: `nsdi24-wu-zhanghao.pdf`, text at `/tmp/nsdi.txt`. Code at `feat/skynomad-merge` HEAD `8e96bc9`. Line citations into the paper text are `nsdi.txt:N`.

---

## 1. Corrections to the brief

### Confirmed correct

`cp(t) = C(0) − C(t)` and `ep(t) = t·C(0)/R(0)` (eq. 4, 5, §5.2.1). The three base rules verbatim (§3.3). Greedy = base rules only (§3.4). UP = base + Uniform Progress + Taking Risks + Hysteresis, threshold `cp(t) ≥ ep(t+2d)` (§5.2.2). Objective `min Σ[s(t)+v(t)k]` (eq. 6). Polarization + homogeneous clusters + the `a(t)=N ⟹ available` mapping (§5.5). 27–84% savings (Table 5). The §7.1 quote is verbatim (`nsdi.txt:1256-1273`). "A delay `d` is charged at the new instance type's price" and "`C(t)` does not decrease for a duration of `d` while `R(t)` continues to decrement" are both literal §3.1.

### C1 — The 63/84 numbers are right; the metric is not what it sounds like

Table 3 defines **"Spot Util." as `spot_hours(policy) / spot_hours(Omniscient)`**, not `spot_hours / total_hours`: "'Spot Util.' indicates the fraction of compute time on spot leveraged by a policy vs. the Omniscient policy." Check: 17.2/27.4 = 62.8% ≈ 63%; 22.9/27.4 = 83.6% ≈ 84%. In absolute terms Greedy runs **36%** of its compute on spot and UP **48%**. A harness computing the absolute ratio will report "failed to reproduce" a result it actually reproduced.

Table 3 is also one configuration, not a headline: job fraction 0.8 (so `C(0)=48h`, `R(0)=60h`), `d=0.2h`, 2-week availability traces, 8 (instance type, zone) pairs × 300 random start points (§6.1). The hours **exclude changeover delays** (§6.2) and sum to exactly `C(0)=48`.

### C2 — In Greedy, on-demand is absorbing

Worth stating because the backtest depends on it: Greedy has no OD→spot edge and no OD→idle edge. The only entry to on-demand is the Safety Net, which says "stay on it until the end" (§3.3). That single fact is Greedy's entire loss — §5.1 on Figure 7: greedy "can no longer afford frequent switches… rendering available spots close to deadline unusable."

### C3 — The Safety Net guard belongs on the *edge into spot*, not on the idle state

§3.3 states Safety Net only for the idle state. But UP's Taking Risks rule enters spot from **on-demand**, and the paper only says "we also apply Safety Net Rule on top" (§5.2.2) without restating the guard there. The correct condition for entering spot from *any* state is `R(t) ≥ C(t) + 2d`. This is precisely the invariant §3.3 uses to prove Exploitation safe: "the Safety Net Rule guarantees that `R(t) ≥ C(t) + 2d` holds at the time `t` when the job is moved to the current spot instance."

Derivation, since we need it for the boundary tests. Let slack `S(t) = R(t) − C(t)`. `S` is constant while making progress and falls at rate 1 while idle or in changeover. Entering spot costs `d`, so `S → S − d`. If the spot is preempted the instant it starts producing, we are idle at `S − d`; the Safety Net then fires (needs `S − d < 2d`), costs another `d`, and leaves `S − 2d ≥ 0`. So `S ≥ 2d` at spot entry is exactly sufficient.

**Corollary — the "stay to the end" latch is derivable, not needed.** After a Safety-Net entry, `S ≈ 2d` at the decision and `≈ d` once the changeover completes, and `S` is then constant. `d < 2d`, so the spot-entry guard blocks OD→spot permanently. Recommend deriving it and exposing `safety_net_engaged` as a read-only field for logging/UX only — a latch that can disagree with the derived state is a bug source once `C̄(0)` is revised at runtime.

### C4 — Rule precedence is under-specified in the prose; Figure 10 resolves it

When idle with `cp < ep` **and** spot available, rules 1 and 2 of §5.2.2 both fire. Figure 10(b) labels the Idle→On-Demand edge with the *spot-unavailable* predicate, so **Taking Risks wins**. And Safety Net must dominate Taking Risks — §3.3 explains why: "After `R(t) < C(t)+2d` becomes true, it is no longer safe to move from idle to spot."

### C5 — Likely error in the brief: UP has **no on-demand → idle edge**

The brief's "Hysteresis — once on on-demand, stay until `cp(t) ≥ ep(t+2d)`" reads as "…and then you may leave." I believe the only legal departure from on-demand is **to spot**. Two independent lines of evidence:

**Textual.** Figure 10's edge labels lost their math glyphs in extraction but the surviving fragments are diagnostic (`nsdi.txt:730-741`):

```
730	                   , spot slice ends, spot        <- (a) Idle->OD [formula]; OD->Spot
731	spot                                              <- (a) Idle->Spot or Spot->Idle
734	slice ends, spot                                  <- (a) OD->Idle
735	(a) Time Sliced
736	spot                                              <- (b) Idle->Spot
737	spot IdleSpot On-Demand                           <- (b) Spot->Idle + node labels
739	               , spot                             <- (b) Idle->OD,  cp(t) < ep(t)
740	                     , spot                       <- (b) OD->Spot, cp(t) >= ep(t+2d)
741	(b) Uniform Progress
```

Panel (a) Time Sliced carries **two** "slice ends" labels — one for OD→Spot and one for OD→Idle. Panel (b) has exactly four edges and no OD-departure label other than the `cp ≥ ep(t+2d), spot` one. The two formula labels in (b) have different indents (15 vs 21 columns), consistent with `cp(t) < ep(t)` and `cp(t) ≥ ep(t+2d)`.

**Arithmetic.** Suppose an OD→idle edge existed at the same threshold. Take the paper's own configuration `C(0)=48, R(0)=60, ρ=0.8, d=0.2` and a trace where spot is never available. Enter OD at `t₀` with `cp₀ = 0.8t₀`; pay 0.2 delay; leave when `cp(t) ≥ 0.8(t+0.4)`, i.e. `0.8t₀ + (t − t₀ − 0.2) ≥ 0.8t + 0.32` ⟹ `t ≥ t₀ + 2.6`. Then idle until `cp < ep` again: `0.8t₀ + 2.4 < 0.8t` ⟹ 0.4h later. **The policy cycles 2.6h on-demand / 0.4h idle forever, burning 0.2h of changeover per 2.4h of progress — 52.0k total vs Greedy's 48.2k, 8% worse**, in exactly the low-availability regime where Figure 12 shows UP beating Greedy (10 → 7 cost-difference-to-Omniscient). Hysteresis was introduced to stop thrash; a leave-to-idle edge reintroduces it in a form hysteresis cannot damp.

**Recommendation:** default to no OD→idle edge; put it behind `UniformProgressPolicy(allow_idle_from_on_demand=False)`; make the harness report both so the ambiguity is measured, not asserted.

### C6 — Missing from the brief and load-bearing: §A.8, loose deadlines

The paper's policies target `C(0)/R(0) > 0.6` (§6.1). For looser deadlines the prescription (§6.1, §A.8) is: run spot-opportunistically — spot when up, idle otherwise, **never on-demand** — until `C(t₀)/R(t₀) = 0.7`, then start the policy.

Without it, raw UP is badly wrong on CarbonSight's own defaults. `packages/core/carbonsight_core/models.py:29` defaults `deadline_hours=48.0`; `apps/api/carbonsight_api/routes/recommendations.py:27,36` defaults `duration_hours=1.0` alongside it — **job fraction 0.021**. Under raw UP with no OD→idle edge: at the first tick `cp=0 < ep`, so it launches on-demand; hysteresis is satisfied ~0.09h later; it then sits on on-demand at $4.10/hr waiting for spot for up to 48 hours to run a 1-hour job. `apps/cli/carbonsight_cli/commands/schedule.py:90` defaults `T = 1.5·P` (fraction 0.67), just inside the paper's range. The wrapper is not optional.

### C7 — Missing: §5.6 relaxations, which are what make this safe in production

- **Computation time.** `δ = C(0) − C̄(0)`. Guarantee degrades to `R(0)+δ`, held by: "all policies stay on the current instance and switch to on-demand, after the job does not finish but has already made `C̄(0)` progress, i.e. `C̄(t) ≤ 0`" (from spot, after preemption — footnote 5). This means **Thrifty must key on the job's own completion signal, not on `C̄(t) ≤ 0`**, and `C̄(t) ≤ 0 ∧ ¬finished` is a separate "pin to on-demand" rule.
- **Changeover delay.** With mean `d` and max `d̂`, the guarantee is `R(0) + 2(d̂ − d)` (proved §A.4); to hold the original deadline, run against `R(0) − 2(d̂ − d)`.
- Combined: `R(0) + δ + 2(d̂ − d)`.

### C8 — Missing: §5.4, and it is the reason `lifetime.py` cannot be salvaged

§5.4's Next Spot Lifetime Oracle is the paper's only lifetime-aware variant, and `o(t)` is an explicitly hypothetical **provider-supplied** value ("This assumption is reasonable as providers can determine when to reclaim a spot instance"). §2.2 — "we make no assumptions on spot availability patterns… We discuss existing prediction-based approaches in §8 and leave this direction to future work" — and §8 — "we design our policy to be robust against potential changes in spot eviction strategies" — make the authors' position explicit. A Nelson-Aalen `L̄` is not a stand-in for `o(t)`; it is the class of thing the paper argues against. Its thresholds, for the record: from idle take spot only if `o(t) > kd/(k−1)`; from on-demand only if `o(t) > 2kd/(k−1)`.

### C9 — UP reads no price at all

`k` appears in no UP predicate. It appears only in the §5.4 oracle thresholds and in cost *reporting*. This is a testable property: the new policy module must not import any pricing module. Relevant because `packages/core/carbonsight_core/estimator/pricing.py:56` pins `SPOT_PRICE_FRACTION = 0.35` globally (`k ≈ 2.857` everywhere), while the paper's own workloads span `k` = 2.2 to 11.1 (Table 4). That is a reporting inaccuracy, not a policy one.

### C10 — Omniscient is an ILP

Eq. (6)–(10), solved per trace with CBC/PuLP (refs [14], [31]); §A.7.2/§A.7.3 give homogeneous and heterogeneous multi-instance variants; §A.3 gives Partial Lookahead. The "100%" in Table 3 is an ILP solve, not a policy.

---

## 2. Module design

### 2.1 Layout

```
packages/core/carbonsight_core/uniform_progress/
    __init__.py
    states.py        InstanceState, Observation, Decision
    progress.py      JobProgress — cp/ep/C/R/d and every predicate
    policies.py      Policy protocol; GreedyPolicy; UniformProgressPolicy;
                     LooseDeadlineWrapper; (optional) TimeSlicedPolicy,
                     UniformProgressWithOracle
    cluster.py       Polarization Rule adapter for N > 1
    feasibility.py   classify(C0, R0, d) before anything runs
    controller.py    §7.1 heartbeat loop, d measurement, Provisioner protocol
    placement.py     carbon-aware region choice *within* a tier

packages/core/carbonsight_core/backtest/up/
    __init__.py
    trace.py         SpotTrace, semi-Markov generator, skypilot-org adapter
    simulator.py     discrete-time sim with real changeover delay + preemption
    omniscient.py    exact DP upper bound (ILP-equivalent)
    experiment.py    §6.1 sampling protocol + Table-3 metrics
```

The sim lives under `backtest/` to match the existing `backtest/runner.py` grouping; the policy lives outside it because it is production code. Both are in `carbonsight_core`, satisfying AGENTS.md's "single brain" rule.

### 2.2 Public interface — §7.1 verbatim

```python
class InstanceState(StrEnum):
    IDLE = "idle"; SPOT = "spot"; ON_DEMAND = "on_demand"

@dataclass(frozen=True, slots=True)
class Observation:
    current_state: InstanceState
    is_spot_available: bool
    job_finished: bool = False        # §5.6: Thrifty keys on this, not on C̄(t) <= 0

@dataclass(frozen=True, slots=True)
class Decision:
    state: InstanceState
    rule: Rule                        # thrifty | safety_net | exploitation |
                                      # taking_risks | uniform_progress |
                                      # hysteresis | wait | overrun_pin
    changeover: bool                  # state != current_state and state != IDLE

class Policy(Protocol):
    def decide(self, obs: Observation, progress: JobProgress) -> Decision: ...
```

Pure function of `(current_state, is_spot_available)` plus the progress state. No clock, no providers, no I/O, no randomness, no config beyond `d`. Zone and instance type never appear — they are the controller's fixed inputs (§7.1).

### 2.3 `JobProgress`

```python
@dataclass(frozen=True, slots=True)
class JobProgress:
    total_compute_hours: float      # C(0), the estimate C̄(0)
    total_deadline_hours: float     # R(0)
    progress_hours: float           # cp(t)
    elapsed_hours: float            # t
    changeover_delay_hours: float   # d
```

Derived, all one-liners:

| name | formula | paper |
|---|---|---|
| `remaining_compute` | `C(t) = C(0) − cp` | §3.1 |
| `remaining_time` | `R(t) = R(0) − t` | §3.1 |
| `slack` | `S(t) = R(t) − C(t)` | derived, C3 |
| `expected_progress` | `ep(t) = t·C(0)/R(0)` | eq. 5 |
| `expected_progress_at(u)` | `u·C(0)/R(0)` | eq. 5 |
| `behind_schedule` | `cp < ep(t)` | §5.2.2(1) |
| `hysteresis_satisfied` | `cp ≥ ep(t + 2d)` | §5.2.2(3) |
| `can_enter_spot` | `S(t) ≥ 2d` | §3.3, C3 |
| `estimate_exhausted` | `C(t) ≤ 0` | §5.6 |

`packages/core/carbonsight_core/spot/progress.py:112` (`is_thrifty`) and `:116` (`is_safety_net`) are already exactly right — `T − t < P − p + 2d` is `R(t) < C(t) + 2d`. They carry over unchanged. Everything else in that file (`deadline_pressure` `:73`, `future_progress_value` `:86`, `avg_progress`, `target_rate`) dies.

### 2.4 Transition tables

Both tables are evaluated **top to bottom, first match wins**, after one normalisation:

> **Preemption normalisation.** If `current_state == SPOT and not is_spot_available`, the instance no longer exists; rewrite `current_state := IDLE` before matching. Termination is free and instantaneous (§3.1: "Switching from an instance to idle… does not incur a delay").

#### Greedy (§3.4)

| # | current | spot avail | extra guard | → | rule | cite |
|---|---|---|---|---|---|---|
| 1 | any | any | `job_finished` | idle | thrifty | §3.3 |
| 2 | on_demand | any | — | on_demand | safety_net (absorbing) | §3.3, C2 |
| 3 | spot | true | — | spot | exploitation | §3.3 |
| 4 | any | any | `C(t) ≤ 0 ∧ ¬finished` | on_demand | overrun_pin | §5.6, **R1-1** |
| 5 | idle | any | `S(t) < 2d` | on_demand | safety_net | §3.3 |
| 6 | idle | true | — | spot | greedy-1 | §3.4 |
| 7 | idle | false | — | idle | wait | §3.4 |

**R1-1:** the overrun pin sits *below* Exploitation, not above it. §5.6 footnote 5: *"If the job was on a spot instance, it should switch to on-demand **after the spot instance is preempted** (Exploitation Rule)."* Placing it above forces a paid changeover off a working spot instance — the move Exploitation exists to forbid. Because `SPOT ∧ ¬available` is normalised to `IDLE` first, this ordering yields footnote 5's behaviour for free.

#### Uniform Progress (§5.2.2 + §3.3)

| # | current | spot avail | extra guard | → | rule | cite |
|---|---|---|---|---|---|---|
| # | current | spot avail | extra guard | → | rule | cite |
|---|---|---|---|---|---|---|
| 1 | any | any | `job_finished` | idle | thrifty | §3.3 |
| 2 | spot | true | — | spot | exploitation | §3.3 |
| 3 | any | any | `C(t) ≤ 0 ∧ ¬finished` | on_demand | overrun_pin | §5.6, **R1-1** |
| 4 | on_demand | true | `cp ≥ ep(t+2d) ∧ S(t) ≥ 2d` | spot | hysteresis + taking_risks | §5.2.2(2,3), C3 |
| 5 | on_demand | any | — | on_demand | hysteresis / safety_net | §5.2.2(3), C5 |
| 6 | idle | any | `S(t) < 2d` | on_demand | safety_net | §3.3, C4 |
| 7 | idle | true | — | spot | taking_risks | §5.2.2(2) |
| 8 | idle | false | `cp < ep(t)` | on_demand | uniform_progress | §5.2.2(1) |
| 9 | idle | false | `cp ≥ ep(t)` | idle | wait | — |

Row 5 is C5: **no OD→idle edge** — now CONFIRMED by rendering Figure 10(b), see §10.

Row 4's `S(t) ≥ 2d` conjunct is **provably redundant, not load-bearing** (§10, M-2). Hysteresis already implies it: with `f = C(0)/R(0)`, `t = cp + S(0) − S` and `f/(1−f) = C(0)/S(0)`, the condition `cp ≥ f·(t+2d)` rearranges to `cp ≥ C(0) − (C(0)/S(0))·(S − 2d)`, so `S < 2d` would demand `cp > C(0)` — unreachable while the job is unfinished. Verified over 179,155 hysteresis-true states: zero had `S < 2d`. **Keep the conjunct** as a cheap assertion documenting the invariant, but do not write a regression test for it — the state it guards cannot be reached.

Deltas vs Greedy, for the record: row 4 exists (Greedy's row 3 is unconditional), and row 8 exists (Greedy waits instead). Nothing else differs.

#### `LooseDeadlineWrapper` (§A.8, C6)

```
if C(t)/R(t) < switch_fraction (default 0.7):
    # Safety Net is a BASE rule (sec 3.3) and binds every policy, wrapper included.
    if not finished and S(t) < 2d and current_state != SPOT:
        return ON_DEMAND, "safety_net"
    return SPOT if is_spot_available else IDLE      # otherwise never on-demand
else:
    return inner.decide(...)
```

Once it hands off, it does not hand back. Applies to Greedy and UP alike. Default on for any job with `C(0)/R(0) ≤ 0.6`.

**The Safety Net check inside the wrapper is mandatory, not defensive.** Without it the wrapper misses the deadline on **9.0% of correctly-gated runs** (542/5,999, fractions as loose as 0.145) — see §10, M-1. The cause is a units mismatch: the handoff triggers on a *ratio* (`C/R ≥ 0.7`, i.e. `S = 0.3R`) while the guarantee needs an *absolute* reserve (`S ≥ 2d`). Those coincide only when `R ≥ 6.67d`; below that the wrapper idles the job straight past the point of no return and hands the inner policy an already-doomed state. §3.3 is explicit that the three base rules bind *"all policies without future knowledge"* — the wrapper is one. With the check, misses go to **0/5,999**.

### 2.5 `cp` vs `t` — the accounting that is easy to botch

Per tick of length `Δ`, in this order:

1. `t += Δ` — **always**, in every state including idle and changeover.
2. If a changeover is in flight: `delay_remaining -= Δ`; **`cp` does not move**; charge `price(target_tier) · Δ`.
3. Else if `state ∈ {spot, on_demand}`: `cp += Δ`; charge `price(state) · Δ`.
4. Else (idle): charge nothing.

Consequences that must be pinned by tests:
- `S(t) = R(t) − C(t)` is **constant** while productive, and falls at rate 1 while idle *or in changeover*. This is the single invariant the whole deadline guarantee rests on.
- A changeover started at `t` and preempted at `t + w`, `w < d`, costs `spot_price · w` and yields **zero** progress. That is where "a preemption costs work" actually lives — there is no separate penalty term in the paper's model, and inventing one would diverge from it.
- Re-entering the same tier in the same region after a preemption pays a **full** `d`: §3.1 says the delay occurs "whenever a job switches to a new spot or on-demand instance."
- Staying in a state pays no `d`.

The current backtest does none of this: `packages/core/carbonsight_core/backtest/spot_runner.py:440` does `done += 1.0` on every hour the policy runs, `COLD_START_HR = 0.1` at `:77` is used only inside the safety-net predicate and inside `effectiveness()`, and no delay is ever charged. That is the defect, and it is why "lifetime" and "effectiveness" were decorative.

### 2.6 Where `d` comes from

`d` = provision + setup + progress lost since the last checkpoint (§3.1). Paper's measured values (Table 4): ML 4+5+9 min ≈ 0.3h; Bioinfo 2+1+8 min ≈ 0.2h; Analytics 4+1+7 min ≈ 0.2h. Experimental default 0.2h (§6.1).

CarbonSight already has a configured estimator at `packages/core/carbonsight_core/spot/scheduler_service.py:159-161`: `d = cold_start_minutes/60 + checkpoint_size_gb · 0.002`. That matches §5.6's "the average changeover delay is given." Keep the function, move it to `uniform_progress/progress.py`, and:

- **Measure it.** The controller timestamps each `decision → first progress heartbeat` interval and maintains `d̄` (mean) and `d̂` (max). Report both.
- **Use §5.6 honestly.** Run the policy against an effective deadline `R(0) − 2(d̂ − d̄)` if the caller wants the original deadline held, or report the relaxed guarantee `R(0) + 2(d̂ − d̄)`. Do not silently use `d̄` and claim the original deadline.
- **Feed back.** Persist measured `d̄` per (region, instance type, checkpoint size) so the next run's configured `d` is real. `packages/core/carbonsight_core/tracking.py`'s SQLite ledger is the obvious home.
- **Align the default.** Today `models.py:39` says 5.0 min, `schedule.py:223` says 6.0, `scheduler_service.py:52` says 0.1h, `spot_runner.py:77` says 0.1h. Pick one; the paper's 0.2h is a better prior for a GPU instance with a checkpoint restore.
- **Charge at the new tier's price** (§3.1) — the simulator and the cost reporter must both do this.

### 2.7 Polarization (§5.5)

`cluster.py` is an adapter, not a policy. The policy is untouched.

- `is_spot_available := (spot_capacity_available(zone, type) ≥ N)`. Paper says `a(t) = N`; `≥` is the right implementation since you only need N.
- Action space is `{0 instances, N spot, N on-demand}`. Homogeneous only.
- A partial preemption (`a(t) < N` while running spot) is a full preemption: tear down to 0, re-decide.
- Cost per tick = `N · price(tier) · Δ`.
- A changeover is incurred on any reconfiguration **except** one that ends with no instances (§5.5).
- Validated only to N=16 (§6.6, §A.7.4). Refuse or warn above that.

### 2.8 Boundary conditions

| condition | behaviour | source |
|---|---|---|
| `t = 0` | `ep(0)=0=cp(0)`, so `cp ≥ ep` — UP does **not** go on-demand at `t=0`. With spot down it goes on-demand at the *first tick after* `t=0`. Pin this. | eq. 5 |
| `C(0) = 0` | Thrifty at `t=0`; never launches. `JobSpec.duration_hours` is `gt=0` (`models.py:14`) so it can't arrive via the API, but the policy must not divide by it. | §3.3 |
| `R(0) ≤ 0` | Reject at `feasibility.classify`, before the policy runs. | — |
| `R(t) ≤ 0 ∧ C(t) > 0` | Cannot occur if the guarantee holds; if it does (bad `C̄(0)`, straggler `d̂`), rows 2/6 pin on-demand and the controller raises `deadline_missed`. Never idle with work left and no time. Never miss silently. | §5.6, §A.4 |
| `R(0) < C(0) + d` | **Infeasible.** Reject with a reason; do not return a decision. | derived |
| `C(0)+d ≤ R(0) < C(0)+2d` | **On-demand only.** Safety Net fires at `t=0`; the policy has no room to gamble. Say so rather than pretending to schedule. | §3.3 |
| `R(0) ≥ C(0) + 2d` | Policy applies. | §3.3 |
| `C(0)/R(0) ≤ 0.6` | `LooseDeadlineWrapper`. | §6.1, §A.8 |
| changeover in flight | Suppress re-decision; only preemption detection runs. The §7.1 three-state interface has no "provisioning" state — report `current_state = target tier` while provisioning. **Not covered by the paper**; flagged as ours. | — |
| `Δ > d` | The tick cannot resolve the delay. Require `Δ ≤ d` in the simulator, or account for the fractional remainder. Paper's probe interval is 10 min and `d` ≥ 0.02h = 1.2 min, so `Δ ≤ d` is not automatic — Figure 13 sweeps `d = 0.02h`. Handle fractional delays explicitly. | §6.1, §6.5 |

---

## 3. What is deleted, what is kept, and why

### Delete

| target | lines | why |
|---|---|---|
| `spot/lifetime.py` | 248 | Nelson-Aalen `L̄`, `S(l)`, `γ*`. UP is "parameter-free and requires no assumptions on spot availability" (abstract). `L̄` appears in no UP predicate. C8: the paper's only lifetime-aware variant uses a *provider oracle*, and §2.2/§8 argue against prediction. |
| `spot/availability.py` | 250 | `AvailabilityTracker` / `VirtualInstance` / `synthetic_probe_trace` exist to feed Nelson-Aalen. In production `is_spot_available` comes from `create_capacity_reservation` (§7.1), not a probe log. **Lift `is_available()`'s replay semantics (`:179-187`) into `backtest/up/trace.py`.** Do not lift `synthetic_probe_trace` — i.i.d. Bernoulli is the wrong generator (§4.1 below). |
| `spot/unified_model.py` — `effectiveness`, `utility`, `candidate_utility`, `rank_candidates`, `CandidateState` | 159 | `η = (L̄−d)/L̄` needs `L̄`. `U = V·η − C_total − E/L̄` conflates tier and region into one scalar; UP separates them. `V` itself (`progress.py:86`) has no counterpart in the paper. |
| `spot/policy.py` `SkyNomadPolicy` | 286 | Replaced. The `delta` anti-flapping parameter (`:126`) is exactly the kind of tuned constant UP eliminates. |
| `spot/scheduler_service.py` — `build_candidates`, `schedule_job`, `mean_lifetime_hr`, `synthetic_lifetime_stats`, `ranked_as_json` | ~300 of 456 | The candidate construction is the region×mode cross product that `U_s` ranked. |
| `backtest/spot_runner.py` | 651 | Discredited (see §4). |
| `tests/unit/test_spot_lifetime.py`, `test_spot_availability.py`, `test_spot_policy.py`, `test_spot_backtest.py` | 121 tests | Follow their subjects. |

### Keep, with reasons

| target | why |
|---|---|
| `region_ranking.py` (143) | `AwsRegionRankingService.rank_greenest_first_then_cost_ceiling` (`:81-99`) is *already* "filter to affordable, then greenest-first" — the exact lexicographic shape the carbon tiebreak needs (§5). It is the carbon axis, orthogonal to UP. Untouched. |
| `spot/progress.py` — `is_thrifty` (`:112`), `is_safety_net` (`:116`) | Literally the paper's Thrifty and Safety Net predicates. Move to `uniform_progress/progress.py`; keep the tests (`test_model_pins.py:250-271`, `TestSafetyNetThreshold`). |
| `spot/progress.py` — `carbon_usd_per_hr` (`:32`), `ODCandidate`, `compute_safety_net_total_cost` (`:131`) | Choosing the cheapest/cleanest on-demand region for a Safety-Net launch is a real decision UP still has to make; it is just no longer a *utility*. Move to `placement.py`. Keep `TestSafetyNetCost` (`test_model_pins.py:273-328`). |
| `MigrationCostEstimator` — **split** | Delete the `E/L̄` amortisation (needs `L̄`). **Keep `E = e·ckpt_gb`** and re-purpose it: (a) a cross-region checkpoint transfer *lengthens* `d_r`, which the feasibility filter must see; (b) it is a reported cost of a region change. The paper never changes region, so it has nothing to say here — this is CarbonSight's extension, and deleting the dollar estimate would leave the carbon layer unable to price its own moves. |
| `scheduler_service.py` — `registry_pairs` (`:115`), `default_registry` (`:101`), `cold_start_hours` (`:159`), `warm_forecast_cache` (`:424`) | Registry enumeration and the `d` estimator have nothing to do with `U_s`. Re-home into `uniform_progress/` and `placement.py`. |
| `scheduler.py` `pick_lowest_carbon_start` (65) | The "when to start" lever, entirely outside UP's horizon (before `t=0`). This is the only surviving *temporal* carbon lever — see §5.4. |
| `providers/carbon.py`, `watttime/`, `estimator/*`, `mapping/registry.py`, `preflight/quota.py`, `tracking.py`, `checkpoint*.py`, `telemetry/`, `backtest/runner.py` | Untouched. None of them know about the spot policy. |
| `providers/spot.py` | Prices are needed for cost *reporting* and for the region cost band — never by the policy (C9). |

---

## 4. The four surfaces

### 4.1 `advise` — barely touched, and it gets simpler

`apps/cli/carbonsight_cli/commands/advise.py` has two branches. The WattTime branch (`:174-192`) already uses `AwsRegionRankingService` and is **completely unaffected**. Only the central-cache branch (`:160-172`) calls `schedule_job` — and it does so only to reach `result.estimates`, i.e. `candidates_to_estimates` (`scheduler_service.py:330`).

**Fix by simplification, not deletion:** make the central-cache branch build `EstimateResult` rows from the carbon path directly (registry × `ApiCarbonProvider` × `estimate_cost_usd`), so both branches share one shape and `advise` has **no dependency on the spot scheduler at all**. `advise` is a carbon-only surface and always should have been. The `utility_score` / `survival` / `lbar_hours` notes it currently emits (`scheduler_service.py:369-372`) go away; the CO2/cost/confidence columns the table actually renders (`advise.py:206-215`) are unchanged.

### 4.2 `schedule` — replaced by `up`

`carbonsight schedule` prints a 43-row region×mode `U` table. That table has no counterpart in the paper. Replace it with:

```
carbonsight up --yaml job.yaml \
    --compute-hours 48 --deadline-hours 60 \
    --progress-hours 12 --elapsed-hours 20 \
    --current-state spot --spot-available/--no-spot-available \
    [--policy up|greedy] [--instances 1] [--json]
```

Output is one decision, not a ranking:

```
state       spot -> on_demand
rule        uniform_progress   (§5.2.2 rule 1)
cp / ep     12.00 / 16.00 h    behind by 4.00 h
C(t) R(t)   36.00 / 40.00 h    slack 4.00 h, need 2d = 0.40 h
d           0.20 h (configured)  |  changeover: yes, charged at on-demand
region      eu-north-1  (cleanest of 3 affordable on-demand regions, 41 gCO2/kWh)
```

Keep `schedule` as a deprecated alias for one release that prints a pointer, or delete outright — recommend delete, since the semantics are not a superset.

Note the flag rename: `--cold-start-minutes` → `--changeover-delay-minutes`, because §3.1's `d` includes lost progress, not just cold start. The current name has been quietly under-counting.

### 4.3 `/v1/recommendations` — v2, clean break

`apps/api/carbonsight_api/routes/recommendations.py` currently returns `{action, decision, value_v, deadline_passed, regions[]}` where `regions[]` carries `utility_score`, `survival`, `lbar_hours`. All three die.

Recommend a clean break at `0.3.0` (there is no stated external contract, and half the payload becomes meaningless):

```jsonc
POST /v1/recommendations
{ "gpu_type": "A100", "gpu_count": 1,
  "compute_hours": 48.0, "deadline_hours": 60.0,
  "progress_hours": 12.0, "elapsed_hours": 20.0,
  "changeover_delay_hours": 0.2,
  "current_state": "spot", "is_spot_available": false,
  "instances": 1, "policy": "uniform_progress",
  "max_cost_premium": 0.20 }

200
{ "feasibility": "policy_applies",         // | on_demand_only | infeasible | loose_deadline
  "decision": { "state": "on_demand", "rule": "uniform_progress", "changeover": true },
  "progress": { "cp": 12.0, "ep": 16.0, "C_t": 36.0, "R_t": 40.0, "slack": 4.0,
                "safety_net_threshold": 0.4, "safety_net_engaged": false },
  "placement": { "region": "eu-north-1", "reason": "cleanest of 3 within +20% of cheapest",
                 "carbon_kg_per_hr": 0.041, "price_usd_per_hr": 4.10,
                 "changeover_delay_hours": 0.2, "egress_usd": 0.0 },
  "guarantee": { "deadline_hours": 60.0, "relaxed_by_delay_variance_hours": 0.0 },
  "regions": [ /* unchanged EstimateResult rows, carbon-ranked, for context */ ] }
```

Notable simplifications: `value_v` and the JSON-infinity dance (`recommendations.py:94-98`, `schedule.py:38-56`) disappear entirely — nothing in UP is ever infinite. Keep `RecommendationRequest`'s pydantic bounds; add `compute_hours` as the explicit `C(0)` (today it is `duration_hours`, which is overloaded with the carbon estimator's meaning). Add 422 for `infeasible`.

Also fix the default: `deadline_hours=48.0` against `duration_hours=1.0` is job fraction 0.021 (C6). Either make deadline required, or default it to `1.5 × duration_hours`.

### 4.4 `backtest` — replaced wholesale

`carbonsight backtest run --spot` → `carbonsight backtest up`. The non-spot path (`backtest/runner.py`, driven from `commands/backtest.py`) is untouched, along with `tests/e2e/test_backtest.py::test_backtest_run_small` and `::test_backtest_reproducible`. New surface in §5.

### 4.5 Test accounting

437 collected today (`pytest --collect-only -q`).

| bucket | count | action |
|---|---|---|
| `test_spot_lifetime.py` | 34 | delete |
| `test_spot_availability.py` | 30 | delete; ~4 replay-semantics tests migrate to `sim/trace` |
| `test_spot_policy.py` | 21 | replace with transition-table tests |
| `test_spot_backtest.py` | 36 | replace with harness tests |
| `test_model_pins.py` | 39 | `TestFutureProgressValue` (~14) and `TestUtilityTerms` (~5) and `TestArchitectureWorkedExample` (~4) delete; `TestSafetyNetThreshold` (~4) and `TestSafetyNetCost` (~7) **keep**; `TestCarbonWeight` (~8) partially retarget |
| `test_central_cache_stub.py` | 33 | ~13 retarget (`:146-303`: `schedule_job`, `mean_lifetime`, `build_candidates`, `theta`); ~20 untouched |
| `test_api_routes.py` | 17 | ~4 retarget (`:105-127`, `utility_score`) |
| `test_carbon_provider.py` | 20 | 2 retarget (`:165-170`, `rank_candidates`) |
| `test_spot_scheduler_tracking.py` | 22 | **untouched** — pricing/ledger/`pick_lowest_carbon_start` only |
| `tests/e2e/test_backtest.py` | 5 | 3 replaced (spot path), 2 kept |
| everything else | ~180 | untouched |

Deleted outright ≈ 121; retargeted ≈ 47; untouched ≈ 269. New tests ≈ 110–140. Net ≈ 400–430.

---

## 5. Where carbon goes

UP decides the **tier**. Region is the axis the paper leaves open — every experiment is a single fixed zone, and §7.1 makes zone and instance type *inputs*.

### 5.1 The composition rule

```
tier = policy.decide(observation, progress)     # paper, unmodified, region-blind
if tier is not IDLE:
    region = choose_region(tier, progress, t)
```

`choose_region` is strictly lexicographic and cannot reorder the tier:

1. **Availability.** spot → `{r : is_spot_available(r)}` from the capacity-reservation probe (§7.1). on-demand → all mapped regions, filtered by `preflight/quota.py`.
2. **Feasibility, per region.** Keep `r` only if `R(t) ≥ C(t) + 2·d_r`, where `d_r = d_base + transfer_time(ckpt_gb, current_region → r)`. This is the one place carbon could touch the guarantee, and this filter closes it: a cleaner-but-farther region with a bigger `d_r` is simply dropped when slack is thin. It is a per-region *strengthening* of §3.3's single-`d` predicate.
3. **Stickiness.** If the current region survives (1) and (2) and the cost band in (4), keep it. **A region change is only ever made on a changeover the policy is already paying for** — never as a standalone migration. This replaces the old `delta` anti-flapping parameter (`spot/policy.py:126`) with something that needs no tuning.
4. **Cost band.** Keep `{r : price_tier(r) ≤ (1+ε)·min price_tier}`, ε = `--max-cost-premium`, default 0.20, reusing `region_ranking.py:81-99`.
5. **Carbon.** `argmin` forecast kgCO2/hr over the next `H = min(C(t), forecast_horizon)` hours. Deterministic tiebreak on region code.

Global assertion before any launch executes: `t + d_r + C(t) ≤ R(0)`. If no region passes (2), fall back to the current or nearest region regardless of carbon and log it.

### 5.2 Why a tiebreak and not a term in a cost function

At $50/ton, for a 1×A100 job across the shipped region set, the entire dirtiest-to-cleanest spread is **0.287 kgCO₂/hr = $0.0143/hr**. That is 0.35% of an A100 on-demand hour ($4.10) and 1.0% of a spot hour ($1.435). The repo already proves this: `tests/unit/test_spot_backtest.py:406-429` derives the breakeven at **$174/ton** against a $0.05/hr delta and then asserts `spread/1000 · 50 < delta` — i.e. it asserts, in a passing test, that the shipped carbon lever is inert.

An additive term at $50/ton would be dominated by rounding in the price table. Lexicographic ordering *after* the affordability filter is the only placement where carbon can actually move a decision without lying about its magnitude.

### 5.3 Can preferring a cleaner region increase `d` or risk the deadline?

**Yes, via the checkpoint.** A cross-region move must egress and restore the checkpoint, so `d_r > d_same`. Three defences, all above: per-region `d_r` in the feasibility filter (2); no standalone migrations (3); the pre-launch finish assertion. In addition, the Safety Net launch is the one case where the stakes are highest — it must pick `argmin` over *feasible* regions only, and if `R(t) < C(t) + d_r` for every remote region, it launches locally.

A conservative alternative worth considering if the above proves fiddly: **pin the region at launch.** Carbon then chooses only the initial placement and the Safety-Net on-demand region, `d` is a constant, and the paper's analysis holds verbatim. Recommend shipping the pinned variant first (Phase 3a) and the migrating variant behind a flag (Phase 3b).

### 5.4 Temporal carbon shifting is out of scope, deliberately

Waiting for a greener hour consumes deadline slack, which is exactly the resource the guarantee is denominated in. **CarbonSight must not add a "wait for greener" rule to UP.** The one legal temporal lever is choosing *when to start*, before `t=0` — `scheduler.py::pick_lowest_carbon_start`, which is preserved.

This is a real loss versus the current product story and should be stated in the README rather than buried. Note that `backtest/spot_runner.py:31-34` already carries this as a *limitation* ("this harness cannot exercise temporal carbon shifting"); under UP it becomes a *design statement*.

---

## 6. Validation harness

### 6.1 Why the current one is discredited

- `_run_up_single` (`spot_runner.py:206-227`) and `_run_up_multi` (`:270-304`) run **every hour unconditionally** — they are forbidden to wait, so they buy on-demand at $4.10 whenever spot is down. The file's own docstring (`:240-249`) admits this made "SkyNomad's stall privilege look like skill."
- **No changeover delay anywhere.** `done += 1.0` at `:226, :266, :303, :440`; `COLD_START_HR = 0.1` (`:77`) is used only in the safety-net predicate and in `effectiveness()`. A preemption therefore costs exactly nothing.
- Traces are **i.i.d. Bernoulli per hour** (`:160-167`), so there are no multi-hour or multi-day unavailability periods (§2.2: "periods of unavailability can last for hours or even days") and no non-stationarity (Figure 3: "availability can jump from 100% to 0% within hours"). Fitting Nelson-Aalen to i.i.d. draws fits noise.
- No Omniscient bound, so there is no scale on which "good" means anything.

### 6.2 Trace layer — `backtest/up/trace.py`

```python
@dataclass(frozen=True)
class SpotTrace:
    zone: str; instance_type: str
    start: datetime; step: timedelta        # paper: 10 minutes (§6.1)
    available: Sequence[bool]
    def available_at(self, t_hours: float) -> bool
    def window(self, offset_hours: float, length_hours: float) -> SpotTrace
```

Three sources behind one type:

1. **Real** — `load_skypilot_traces(path)`, an adapter over `github.com/skypilot-org/spot-traces`. **The on-disk schema is unknown here (no network); it is an open question, and the adapter exists so pinning it later is a 30-line change, not a redesign.**
2. **Synthetic, alternating-renewal** — a two-state semi-Markov process with independently configurable available and unavailable run-length distributions (log-normal by default), plus a slow random walk on the availability fraction to mimic Figure 3. Parameterised by target spot fraction so the paper's regimes (0.17 to 0.95, §6.3/§A.5) can be hit. **Not** i.i.d. Bernoulli.
3. **Adversarial** — the §A.2.1 Theorem-1 construction (spot becomes available exactly when the policy switches to on-demand, then is revoked after `d`), as a regression fixture for the deadline invariant.

### 6.3 Simulator — `backtest/up/simulator.py`

State `(t, cp, tier, delay_remaining, region)`, stepped at `Δ ≤ min(trace.step, d)`. The tick order is §2.5 exactly. Preemption is `tier == spot ∧ ¬available_at(t)` → tier becomes idle, `delay_remaining` discarded, no progress, no refund for the partial delay.

Records per run: `spot_compute_h`, `od_compute_h`, `spot_delay_h`, `od_delay_h`, `idle_h`, `n_changeovers`, `n_preemptions`, `cost`, `finish_time`, `deadline_met`, and the full decision trace for Figure-4/7/9-style plots.

Invariant asserted every tick: `spot_compute + od_compute + spot_delay + od_delay + idle == t` and `cp == spot_compute + od_compute`.

### 6.4 Omniscient — `backtest/up/omniscient.py`

**Exact DP, not an ILP.** State `(t_index, work_index, tier)`; transitions: stay in tier (`+Δ` progress at `price(tier)·Δ`), go idle (free, no progress), switch to tier B (advance `d/Δ` steps at `price(B)·Δ`, no progress, spot legal only where `available` holds throughout). For `R(0)=60h`, `Δ=10min`, `C(0)=48h` that is `360 × 288 × 3 ≈ 311k` states — milliseconds, and no new dependency.

Equivalence claim, to be verified: this coincides with the paper's ILP (eq. 6–10) on the same grid. The ILP's constraint (8) accounts for changeover time *globally* (`Σ[s+v] ≥ d·Σ(x+y) + C(0)`) rather than causally, but cost depends only on the time-in-tier profile, so the optima should match. **If they diverge, the ILP is the looser bound.** Optional cross-check: add `pulp` as a dev-only extra and assert agreement on a few hundred traces (§9, risk 4).

Stretch: Partial Lookahead Omniscient with 8 slices (§5.3.2, §A.3), which is the paper's "≈6 hours of lookahead" reference line.

### 6.5 Experiment protocol — `backtest/up/experiment.py`

Mirrors §6.1: `C(0) = 48h`; job fraction ∈ {0.6, 0.7, 0.8, 0.9} so `R(0) = C(0)/fraction`; `d` ∈ {0.02, 0.2, 0.4} h (Figure 13); 300 random start points per (instance type, zone), rejecting starts with fewer than `R(0)` hours of trace left; policies = {OnDemandOnly, Greedy, UniformProgress, Omniscient} (+ TimeSliced, UP+oracle as stretch).

Metrics, exactly as the paper defines them:

- **Table 3**: `od_compute_h`, `spot_compute_h` (delays excluded), and **`spot_util = spot_compute_h(policy) / spot_compute_h(Omniscient)`** — C1.
- **Figures 8/11/13**: `cost_savings_pct = 1 − cost(policy)/cost(OnDemandOnly)`, where `OnDemandOnly` launches once at `t=0` and runs `C(0)+d`.
- **Figures 12/14**: `cost_difference_to_omniscient_pct`, normalised by on-demand cost, reported as mean with p25/p75 error bars.
- Always: `deadline_met_pct`, `n_preemptions`, `delay_hours` as a share of cost (Figure 16's breakdown).

### 6.6 Acceptance gates

**Invariants (must hold on every trace family, synthetic and real; these are bugs, not metrics):**

- `deadline_met == True` in 100% of runs for Greedy and UP whenever `feasibility == policy_applies`. Property test over ≥10,000 randomised `(C(0), R(0), d, trace)` draws including the adversarial family.
- `cost(Omniscient) ≤ cost(UP)` and `≤ cost(Greedy)` on every single trace.
- `n_changeovers ≤ slack/d` on every run (§6.4: *"the number of feasible instance switches is limited to at most (R(0)−C(0))/d"*).
- ~~`cost(policy) ≤ cost(OnDemandOnly)` on every trace.~~ **Removed — this is false and the plan's own adversarial fixture disproves it.** §A.2.1 Theorem 1: for any *deterministic* policy the adversary can force `c ≥ k − O(d)`. On the paper's own config (`C(0)=48, R(0)=60, d=0.2, k=2.857`) UP lands ≈16% *worse* than launching on-demand once. §3.1 also scopes this out: *"We assume that spot availability is **non-adversarial**… except for §4.1."* Demote to a statistical expectation on non-adversarial traces only, and **exclude the adversarial family from every cost-ordering gate** while keeping it for `deadline_met`.
- Once Safety Net fires, the policy returns `on_demand` for the rest of the run.
- Carbon off ⟹ decisions byte-identical to the carbon-free build.

**Reproduction, on the authors' traces (Phase 5 only):** at job fraction 0.8, `d=0.2h`, 2-week traces, ≥8 (instance type, zone) pairs × 300 samples — `spot_util(Greedy) ∈ [58%, 68%]`, `spot_util(UP) ∈ [79%, 89%]`, `cost(UP) ≤ cost(Greedy)` in ≥95% of samples, and UP's gap to Omniscient roughly half Greedy's (§6.4's "reduces the gap by ~2×").

**On synthetic traces (Phases 0–4):** only the *ordering* `Omniscient ≤ UP ≤ Greedy ≤ OnDemandOnly` and 100% deadline adherence. **Do not claim reproduction of 63/84 without the authors' traces** — say so in the output.

---

## 7. Phased sequence

| phase | scope | size | gate |
|---|---|---|---|
| **0. Harness first** | `backtest/up/{trace,simulator,omniscient}.py`, `GreedyPolicy`, `states.py`, `progress.py`. **No deletions.** | +900 LoC, +60 tests | Greedy 100% deadline-met over a 10k sweep incl. adversarial; Omniscient ≤ Greedy on every trace; tick accounting invariants hold. |
| **1. UP** | `UniformProgressPolicy`, exhaustive transition tables, `allow_idle_from_on_demand` flag. | +250 LoC, +50 tests | `spot_util(UP) > spot_util(Greedy)` on every trace family; both 100% deadline-met; the C5 flag produces a measurable, reported difference. |
| **2. Production shape** | `controller.py` (§7.1 loop, `Provisioner` protocol, measured `d̄`/`d̂`), `feasibility.py`, `LooseDeadlineWrapper`, `cluster.py`, §5.6 overrun pin. | +350 LoC, +40 tests | Loose-deadline default (fraction 0.02) never launches on-demand while slack is abundant; N=4 and N=16 polarized runs match the single-instance state trace one-for-one. |
| **3a. Carbon, pinned region** | `placement.py`: region chosen once at launch + at Safety-Net launch. Reuses `region_ranking.py`. | +150 LoC, +20 tests | Carbon-off ⟹ identical decisions; deadline invariant unchanged. |
| **3b. Carbon, migrating region** (flag) | Per-region `d_r`, feasibility filter, stickiness, egress accounting. | +150 LoC, +15 tests | 10k sweep with migration on: still 100% deadline-met; assert no standalone migrations occurred. |
| **4. Surfaces + deletion** | `carbonsight up`; `/v1/recommendations` v2; decouple `advise` from `schedule_job`; `backtest up`. **Then** delete `spot/{lifetime,availability,unified_model,policy}.py`, `backtest/spot_runner.py`, and their tests. | −1,900 / +450 LoC, −121 / +25 tests | Full suite green; ruff clean; ARCHITECTURE.md / CURRENT_STATE.md / README updated in the same PR (AGENTS.md §1). |
| **5. Real traces** (follow-up, needs network) | `load_skypilot_traces`, Table-3 reproduction, published side-by-side numbers. | +150 LoC, +10 tests | §6.6 reproduction gates. |

Ordering rationale: the harness comes first so every subsequent claim is measured, and so nothing is deleted until its replacement is proven on the same traces. Phases 0–3 add code without removing any; Phase 4 is the only destructive one.

---

## 8. Open questions

1. **Figure 10(b)'s missing OD→Idle edge (C5).** ~85% confident from the extraction asymmetry plus the thrash arithmetic, but the figure could not be rendered — no `pdftoppm`, `pdftotext`, `mutool` on this machine. **A reviewer with the PDF open should confirm Figure 10(b) has exactly four edges.** Mitigated by the flag and by measuring both.
2. **`spot-traces` on-disk schema.** Unknown here. Column names, timestamp format, per-zone file layout, and whether availability traces and preemption traces share a schema all need pinning.
3. **`is_spot_available` in production.** §7.1 names `create_capacity_reservation`, which on AWS is a chargeable, side-effecting call with its own quota. The paper glosses the cost. CarbonSight has no live AWS path at all — `providers/spot.py:100-105` raises `NotImplementedError`. Until that is built, the CLI/API can only accept an injected availability oracle or simulate. Say so; do not ship a surface that implies live probing.
4. **DP ≡ ILP.** Argued in §6.4, not proved. Worth the PuLP cross-check if a dev dependency is acceptable.
5. **Runtime progress signal.** `cp(t)` is currently a CLI flag (`schedule.py:238`) / API field. A real controller needs the job to report progress on the heartbeat. That is genuine work the paper assumes away ("we assume that both `C(0)` and `R(0)` are given").
6. **On-demand is assumed always available** (§3.1, footnote 3: "This is a simplifying assumption. In practice, some on-demand instance types can hit unavailability"). For GPUs in 2026 this is frequently false, and if an on-demand launch fails the Safety Net's guarantee is void. Mitigation: fall back across the region set the carbon layer already enumerates, plus an explicit alarm. Not in the paper.
7. **Billing granularity** (§3.1, footnote 4: AWS does not charge for spot preempted within the first hour; GCP does). Our simulator charges per second uniformly, which is slightly pessimistic for AWS spot. Note it in the output.
8. **N > 16 is unvalidated** (§6.6, §A.7.4: "Due to monetary budget limits, we leave the extension to larger clusters (N > 16) to future work").
9. **`k` is a global constant** (`pricing.py:56`, `SPOT_PRICE_FRACTION = 0.35`). Harmless for the policy (C9), wrong for cost reporting against Table 5's `k` range of 2.2–11.1.
10. **Should `TimeSliced` (§5.1) and `UP + next-spot oracle` (§5.4) be built?** Neither is needed for the product. Recommend: skip both in Phases 0–4, and if `UP + oracle` is ever built, name it so nobody mistakes an *estimated* `L̄` for the paper's provider-supplied `o(t)` (C8).

## 9. Risks

| risk | severity | mitigation |
|---|---|---|
| C5 read is wrong ⟹ UP is systematically too expensive in low-availability regimes | high | Flag + harness measures both; the 8%-worse arithmetic gives a clear signal on synthetic traces. |
| Synthetic traces flatter UP, and the 63/84 gates are never really tested | high | Gate Phase 5 on real traces; forbid the phrase "reproduces the paper" until then; make the synthetic generator alternating-renewal, not Bernoulli. |
| Deleting `L̄` removes the only thing the region ranking currently prices spot risk with | medium | The paper's answer is that spot risk is *not* priced — it is absorbed by the Safety Net. The region tiebreak operates within a tier, so it never has to compare spot to on-demand. Verify no `placement.py` predicate needs a lifetime. |
| Carbon migration silently erodes deadline slack | medium | Per-region `d_r` feasibility filter, no standalone migrations, pre-launch finish assertion, and a 10k sweep with migration enabled. Ship pinned-region first. |
| `advise` regresses while being decoupled from `schedule_job` | medium | Its WattTime branch is already independent (`advise.py:174-192`); pin the central-cache branch's output shape with a snapshot test before touching it. |
| Deleting 121 tests looks like coverage loss in review | low | Land the ~110–140 replacements in Phases 0–3 *before* the Phase 4 deletion. |
| Loose-deadline wrapper is skipped as "an optimisation" | high | It is not: without it the shipped API default runs a 1-hour job on on-demand for 48 hours (C6). Make it a Phase-2 gate with an explicit test at fraction 0.021. |

---

## 10. Review log — three independent passes

Reviewed by three agents (paper fidelity / code map / adversarial design). The
adversarial agent died twice on infrastructure errors; its priority-1 work
(prototype the tables, attack the guarantee) was done directly instead —
prototype at `/tmp/up_attack.py`.

### Must-fix, already applied above

| id | finding | status |
|---|---|---|
| **M-1** | `LooseDeadlineWrapper` omits the Safety Net and misses the deadline on **9.0%** of correctly-gated runs (542/5,999). One-line fix verified: **0/5,999**. | fixed in §2.4 |
| **R1-1** | Overrun pin was ordered above Exploitation in both tables, forcing a paid changeover off a live spot instance against §5.6 footnote 5. | fixed in §2.4 |
| **R1-2** | `cost(policy) ≤ cost(OnDemandOnly)` listed as an invariant; §A.2.1 Theorem 1 disproves it and the plan's own adversarial fixture triggers it. | fixed in §6.6 |

### Resolved

- **Open question 1 — settled, plan was right.** Figure 10(b) rendered at 10× (PyMuPDF): exactly four edges, **no on-demand → idle**. Panel (a) Time Sliced *does* have one, so the asymmetry is deliberate. Corroborated by §5.2.2 rule 1 (*"switch to on-demand and stay on it"*) and the Figure 10 discussion (*"slice boundaries to jump off an on-demand instance"* — which UP lacks). Drop the extraction-fragment table in C5; the figure is readable. `allow_idle_from_on_demand` demotes from ambiguity-resolver to optional experiment, default off.
- **M-2 — reviewers disagreed; the conjunct is redundant.** Reviewer 1 asserted a reachable deadline miss if row 4 drops `S ≥ 2d`, and constructed a scenario. Simulation says otherwise: hysteresis *implies* `S ≥ 2d` (algebra in §2.4; 179,155 hysteresis-true states, zero violations; targeted search over the reachable window found no counterexample). Keep the conjunct as an assertion; **do not** write the regression test Reviewer 1 requested.

### Corrections to record

- **C6's mechanism was wrong** (both reviewers, independently). Raw UP on the shipped default does **not** idle for 48h. It runs the whole job on on-demand and finishes in 1.33h at **$4.78 vs ~$1.55** — a 3.1× overpay discarding 46.7h of slack. Hysteresis timing (~0.09h) was right. The wrapper is still mandatory; restate the reason. Also: the paper's loosest *tested* fraction is 0.25 — our 0.021 default is an order of magnitude outside anything measured, so the wrapper is extrapolation. Better support exists in Figure 22 (§A.8, measured) and §A.9.
- **`is_thrifty` is misnamed for the port.** `progress.py:112` is `p >= P` ⟺ `C(t) ≤ 0` — the plan's own `estimate_exhausted`, i.e. the *overrun pin* condition, **not** §5.6 Thrifty. The keep list says "carries over unchanged"; doing so reintroduces the exact bug C7 exists to prevent. **Rename to `estimate_exhausted` on the move**; Thrifty keys on `job_finished`. (`is_safety_net` genuinely is the paper's predicate and does carry over.)
- **`MigrationCostEstimator` lives in `unified_model.py`**, which Phase 4 deletes — keep and delete lists contradict. The seam is *cleaner* than described: `E/L̄` is in `utility()` (`unified_model.py:104`), not the estimator, so the class **moves whole, no split**. Watch for a cycle: it needs `carbon_usd_per_hr`, which is also moving to `placement.py`. The 159-line figure is the whole file; the five deleted symbols are 82.
- **Import-time breakage is unaccounted for.** `spot/__init__.py` re-exports all 30 dying symbols; `backtest/__init__.py` re-exports `spot_runner`; `carbonsight_cli/main.py:13,23` imports `commands/schedule`. Consequence: **71 test defs** in "untouched" files (`test_checkpoint.py`, `test_tier1_oracle.py`) stop collecting via `from carbonsight_cli.main import app`, and `test_central_cache_stub.py` loses all 33, not ~13. Mechanical, Phase 4, but must be on the list.
- **§4.5's table is in `def test_` units, labelled as collected items.** "everything else ≈180" is **128**; the padding is exactly the 52 parametrized expansions, all of which live inside the ten named files. Delete-outright is 121 defs / 125 items; untouched ≈204 defs. Retargets understated 1.4–1.75× on all three files checked. `TestSafetyNetCost` is not a clean keep — `test_carbon_can_decide_between_two_od_regions` (`:312-328`) uses `SkyNomadPolicy`/`CandidateState`.
- **New-test volume contradicts itself**: §4.5 says 110–140, §7's phases sum to 220. Phase 4 also budgets +25 but inherits ~47 retargets.
- **Deletion is ~2,330 lines, not 1,900** — omits `commands/schedule.py` (274) and the `--spot` branch of `commands/backtest.py` (~95).
- **`warm_forecast_cache` is dead** — zero callers repo-wide. Drop it from the keep list.
- **`advise` has nothing to snapshot.** The central-cache branch has *zero* executed coverage; the existing test hits the no-creds early return and the rest are WattTime-gated skips. The snapshot must be authored against a stubbed `CARBONSIGHT_API_URL`. Also, rebuilding it from "registry × provider × cost" reproduces `scheduler_service`'s fake p10/p90 (`kg*0.8`/`kg*1.2`), not the WattTime branch's Monte-Carlo band — one shape requires routing through `JobCarbonEstimator`.
- **`t = 0` is ambiguous in the paper, not mandated.** Rule 1's prose says `cp < ep`; Figure 10(b)'s edge label says `cp ≤ ep` (verified at 26× zoom). They disagree only at `t=0`. Keep `<`, but label it as our resolution, like the "changeover in flight" row.
- **C3 is stronger than derived** — cite §A.4 case 3, which *assumes* `R(t′) ≥ C(t′) + 2d` at every spot entry while proving the `R(0)+2(d̂−d)` bound for all policies.
- **Add**: Table 2 (§3.2) marks `(spot, unavailable)` an impossible cell — that's the authority for preemption normalisation. §3.3's *"no gain for waiting an additional d if the job is idle"* justifies the currently-uncited `C(0)+d ≤ R(0) < C(0)+2d` boundary row. §5.4 *replaces* hysteresis rather than layering on it. **There is no pseudocode anywhere in the paper** — §5.2.2's three rules plus Figure 10(b) are the complete spec.
- **Line drift**: `policy.py:126`→`:124`; `advise.py:160-172`→`:160-173`; `schedule.py:223`→`:224`; `providers/spot.py:100-105`→`:101-106`; §7.1 cite `nsdi.txt:1256`→`:1266`. The C2 Greedy quote is stitched from §3.4 and §5.1. Open question 9's `k` range is Table 4, not Table 5. §6.5's job-fraction grid is a superset of the paper's, not "mirrors §6.1".

### Verified sound

Deadline guarantee holds: **0 misses across 3,810 randomised trials** (C(0) 4–48h, `d` 0.02–0.4h, Bernoulli / bursty / never-available / adversarial traces) for Greedy and UP alike — *provided* `feasibility.classify` runs first. All three policies fail identically without it, which confirms the gate is load-bearing rather than decorative. Tick accounting, the `S(t)` invariant, C1's Table-3 metric, the §5.4 thresholds, Tables 4 and 5, the §6.1 protocol, Polarization, the §7.1 interface quote, §5.2's carbon-inertness arithmetic (0.287 kg/hr, $0.0143/hr, $174/ton breakeven), the 43-row count, and the three architectural claims (`advise` decouples cleanly, `region_ranking.py` untouched, `test_spot_scheduler_tracking.py` untouched) all check out. Sizing is ~1:1, 2,240 production LoC replaced by ~2,400.
