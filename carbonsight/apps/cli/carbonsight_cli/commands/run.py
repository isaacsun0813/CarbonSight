"""carbonsight run train.yaml [--dry-run] [--no-exec] [--managed] [--max-cost-premium N]"""

import math
import os
import sqlite3
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path

import typer
import yaml
from carbonsight_core.checkpoint import (
    CheckpointYamlPatch,
    Framework,
    apply_checkpoint_patch_to_yaml,
    build_checkpoint_config,
    build_checkpoint_yaml_patch,
    detect_framework,
    extract_script_path_from_run_command,
    shim_local_path,
)
from carbonsight_core.cloud.aws.availability import InstanceAvailabilityChecker
from carbonsight_core.cloud.aws.enabled_regions import EnabledRegionsProvider
from carbonsight_core.cloud.aws.gpu_catalog import aws_instance_type_for_gpu
from carbonsight_core.cloud.aws.probe import ProbeResult, SpotLaunchProbe
from carbonsight_core.cloud.aws.quota import QuotaChecker
from carbonsight_core.config import Config
from carbonsight_core.estimator.carbon_model import JobCarbonEstimator
from carbonsight_core.estimator.pricing import configure_pricing, estimate_cost_usd
from carbonsight_core.mapping.registry import Registry
from carbonsight_core.models import EstimateResult, JobSpec
from carbonsight_core.paths import (
    carbonsight_package_root_from_cli_command_file,
    resolve_registry_json_file,
)
from carbonsight_core.planning import (
    DynamicEventType,
    DynamicJob,
    DynamicPlanEvent,
    DynamicPlanResult,
    JobMode,
    JobRuntimeState,
    PlanConstraints,
    RegionPlanInput,
    SpotObservation,
    build_region_plan_inputs,
    plan_dynamic_job,
)
from carbonsight_core.planning.survival import (
    ASSUMED_SPOT_MEAN_LIFETIME_HOURS,
    DEFAULT_BUCKET_HOURS,
    NelsonAalenModel,
    VirtualInstanceTracker,
    replay_observations,
)
from carbonsight_core.region_ranking import AwsRegionRankingService
from carbonsight_core.scheduler import pick_lowest_carbon_start
from carbonsight_core.spot_history import SpotObservationStore, StoredObservation
from carbonsight_core.tracking import RunLedger, RunRecord
from carbonsight_core.watttime import WattTimeClient, WattTimeError

from carbonsight_cli.commands.advise import (
    apply_gpu_telemetry_cli,
    carbonsight_block_from_yaml,
    job_spec_from_sky_yaml,
    parse_duration_hours,
    parse_finish_by,
    resolve_carbon_budget_kg,
    resolve_carbon_price,
    resolve_finish_by,
    resolve_live_aws_pricing,
    strip_carbonsight_only_yaml_keys,
)

BASELINE_REGION = "us-east-1"


def _echo_dynamic_plan(dyn, *, title: str = "Dynamic plan (advisory simulation)") -> None:
    """Print concise dynamic plan trajectory."""
    typer.echo(f"\n{title}:")
    typer.echo(
        f"  Deadline met: {dyn.deadline_met}  "
        f"Cost: ${dyn.total_cost_usd:.2f}  Carbon: {dyn.total_carbon_kg:.2f}kg"
    )
    if dyn.initial_launch:
        typer.echo(
            f"  Initial launch: {dyn.initial_launch.region} "
            f"({dyn.initial_launch.mode.value})"
        )
    for ev in dyn.events[:12]:
        typer.echo(
            f"  {ev.time_utc:%Y-%m-%d %H:%M}Z {ev.event_type.value} "
            f"{ev.region or '-'} progress={ev.progress_hours:.2f}h"
        )
    if len(dyn.events) > 12:
        typer.echo(f"  ... {len(dyn.events) - 12} more events")


def _apply_cloud_region_to_resources(
    resources: dict,
    cloud: str,
    region: str,
    *,
    use_spot: bool = False,
    force_on_demand: bool = False,
) -> None:
    """Set SkyPilot ``resources.infra`` and drop legacy cloud/region/zone keys.

    ``force_on_demand`` writes ``use_spot: false`` so a YAML that already says spot cannot
    override an on-demand launch chosen by the planner.
    """
    resources["infra"] = f"{cloud}/{region}"
    for key in ("cloud", "region", "zone"):
        resources.pop(key, None)
    if use_spot:
        resources["use_spot"] = True
    elif force_on_demand:
        resources["use_spot"] = False


def patch_sky_yaml_with_cloud_region(
    yaml_path: Path,
    cloud: str,
    region: str,
    *,
    use_spot: bool = False,
    force_on_demand: bool = False,
) -> str:
    """Inject resources.infra (cloud/region) into SkyPilot YAML; return SkyPilot-ready text."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if "resources" not in data:
        data["resources"] = {}
    _apply_cloud_region_to_resources(
        data["resources"], cloud, region, use_spot=use_spot, force_on_demand=force_on_demand,
    )
    return yaml.dump(
        strip_carbonsight_only_yaml_keys(data),
        default_flow_style=False,
        sort_keys=False,
    )


def _require_sky_cli() -> None:
    sky_check = subprocess.run(["sky", "--version"], capture_output=True)
    if sky_check.returncode != 0:
        typer.echo(
            "SkyPilot not found. Install it with: pip install skypilot[aws]",
            err=True,
        )
        raise typer.Exit(1)


def launch_skypilot_with_patched_yaml(
    patched_yaml: str,
    *,
    managed: bool,
    yes: bool,
    dryrun: bool = False,
    down: bool = False,
    cwd: Path | None = None,
    detach: bool = False,
    cluster_name: str | None = None,
    idle_minutes_to_autostop: int | None = None,
) -> int:
    """
    Write YAML to temp file, run sky launch or sky jobs launch. Returns process exit code.

    ``detach`` (``-d``), ``cluster_name`` (``-c``) and ``idle_minutes_to_autostop`` (``-i``)
    apply to plain ``sky launch`` only; they let a caller submit the job and poll it.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(patched_yaml)
        tmp_path = Path(f.name)

    try:
        cmd = ["sky", "jobs", "launch"] if managed else ["sky", "launch"]
        if yes:
            cmd.append("--yes")
        if dryrun:
            cmd.append("--dryrun")
        if down:
            cmd.append("--down")
        if not managed:
            if detach:
                cmd.append("--detach-run")
            if cluster_name:
                cmd.extend(["--cluster", cluster_name])
            if idle_minutes_to_autostop is not None:
                cmd.extend(["--idle-minutes-to-autostop", str(idle_minutes_to_autostop)])
        cmd.append(str(tmp_path))

        typer.echo(f"Running: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=cwd).returncode
    finally:
        tmp_path.unlink(missing_ok=True)


def build_patched_yaml_text(
    yaml_path: Path,
    cloud: str,
    region: str,
    *,
    use_spot: bool = False,
    ckpt_patch: CheckpointYamlPatch | None = None,
    force_on_demand: bool = False,
) -> str:
    """SkyPilot-ready YAML text for one launch: cloud/region, spot flag, optional checkpoint patch."""
    if ckpt_patch is None:
        return patch_sky_yaml_with_cloud_region(
            yaml_path, cloud, region, use_spot=use_spot, force_on_demand=force_on_demand,
        )
    data = yaml.safe_load(yaml_path.read_text()) or {}
    if "resources" not in data:
        data["resources"] = {}
    _apply_cloud_region_to_resources(
        data["resources"], cloud, region, use_spot=use_spot, force_on_demand=force_on_demand,
    )
    data = apply_checkpoint_patch_to_yaml(data, ckpt_patch)
    return yaml.dump(
        strip_carbonsight_only_yaml_keys(data),
        default_flow_style=False,
        sort_keys=False,
    )


# --- Real SkyPilot execution for the dynamic (deadline) path -------------------------------

Clock = Callable[[], datetime]
Sleep = Callable[[float], None]
# Launch-then-terminate probe for one cloud region; injectable for tests.
Prober = Callable[[str], ProbeResult]

# Job states printed by ``sky queue`` and how the dynamic loop treats them.
_SKY_JOB_STATES = frozenset({
    "INIT", "PENDING", "SETTING_UP", "RUNNING",
    "SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "CANCELLED",
})
_SKY_CLUSTER_STATES = frozenset({"UP", "INIT", "STOPPED"})

AUTOSTOP_IDLE_MINUTES = 10


class SkyStatus(str, Enum):
    """What the dynamic loop sees for the launched job."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PREEMPTED = "preempted"
    FAILED = "failed"
    UNKNOWN = "unknown"


_SKY_JOB_TO_STATUS = {
    "INIT": SkyStatus.RUNNING,
    "PENDING": SkyStatus.RUNNING,
    "SETTING_UP": SkyStatus.RUNNING,
    "RUNNING": SkyStatus.RUNNING,
    "SUCCEEDED": SkyStatus.SUCCEEDED,
    "FAILED": SkyStatus.FAILED,
    "FAILED_SETUP": SkyStatus.FAILED,
    "FAILED_DRIVER": SkyStatus.FAILED,
    "CANCELLED": SkyStatus.FAILED,
}


def parse_sky_queue_job_state(text: str) -> str | None:
    """First job state token found in ``sky queue <cluster>`` output (one job per cluster)."""
    for line in text.splitlines():
        for token in line.split():
            if token in _SKY_JOB_STATES:
                return token
    return None


def parse_sky_cluster_status(text: str, cluster_name: str) -> str | None:
    """Cluster state (UP / INIT / STOPPED) for ``cluster_name`` in ``sky status`` output, else None."""
    for line in text.splitlines():
        tokens = line.split()
        if tokens and tokens[0] == cluster_name:
            for token in tokens[1:]:
                if token in _SKY_CLUSTER_STATES:
                    return token
    return None


@dataclass(frozen=True, slots=True)
class SkyHandle:
    """One launched SkyPilot cluster."""

    cluster_name: str
    region: str
    mode: JobMode


class SkyRunner:
    """Launch, poll and tear down one SkyPilot cluster per launch (subprocess boundary)."""

    def __init__(
        self,
        yaml_for_launch: Callable[[str, bool], str],
        *,
        cwd: Path | None,
        run_id: str,
        yes: bool = False,
        idle_minutes_to_autostop: int = AUTOSTOP_IDLE_MINUTES,
        run_fn: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        launch_fn: Callable[..., int] = launch_skypilot_with_patched_yaml,
    ) -> None:
        self._yaml_for_launch = yaml_for_launch
        self._cwd = cwd
        self._run_id = run_id
        self._yes = yes
        self._idle_minutes = idle_minutes_to_autostop
        self._run_fn = run_fn
        self._launch_fn = launch_fn
        self._launches = 0

    def launch(self, region: str, mode: JobMode) -> SkyHandle | None:
        """Submit the job detached on a fresh cluster. None when SkyPilot reports failure."""
        self._launches += 1
        handle = SkyHandle(
            cluster_name=f"cs-{self._run_id}-{self._launches}", region=region, mode=mode,
        )
        patched = self._yaml_for_launch(region, mode == JobMode.SPOT)
        # Autodown after idle is the safety net if this process dies; teardown() is the normal path.
        code = self._launch_fn(
            patched,
            managed=False,
            yes=self._yes,
            down=True,
            cwd=self._cwd,
            detach=True,
            cluster_name=handle.cluster_name,
            idle_minutes_to_autostop=self._idle_minutes,
        )
        if code != 0:
            self.teardown(handle)
            return None
        # The user confirmed the first launch; do not re-prompt on relaunches in an unattended loop.
        self._yes = True
        return handle

    def poll(self, handle: SkyHandle) -> SkyStatus:
        """Job state from ``sky queue``; if the cluster is unreachable, tell preemption from a blip."""
        queue = self._run_fn(
            ["sky", "queue", handle.cluster_name], capture_output=True, text=True,
        )
        if queue.returncode == 0:
            job_state = parse_sky_queue_job_state(queue.stdout)
            return _SKY_JOB_TO_STATUS.get(job_state, SkyStatus.UNKNOWN)
        status = self._run_fn(
            ["sky", "status", "--refresh", handle.cluster_name],
            capture_output=True,
            text=True,
        )
        if status.returncode != 0:
            return SkyStatus.UNKNOWN
        if parse_sky_cluster_status(status.stdout, handle.cluster_name) == "UP":
            return SkyStatus.UNKNOWN  # reachable per status; the queue call was a transient error
        return SkyStatus.PREEMPTED

    def teardown(self, handle: SkyHandle) -> None:
        """Best-effort ``sky down``; never raises."""
        result = self._run_fn(
            ["sky", "down", "--yes", handle.cluster_name], capture_output=True, text=True,
        )
        if result.returncode != 0:
            typer.echo(
                f"Warning: `sky down {handle.cluster_name}` failed; check for a leftover cluster.",
                err=True,
            )


@dataclass(slots=True)
class DynamicLoopOutcome:
    """Result of the dynamic event loop."""

    state: JobRuntimeState
    handle: SkyHandle | None = None
    failed: str | None = None
    interrupted: bool = False
    # (cloud_region, start_utc, end_utc) for each launched segment, from the real clock.
    segments: list[tuple[str, datetime, datetime]] = field(default_factory=list)


def _run_probe_round(dyn: DynamicJob, prober: Prober, now: datetime) -> None:
    """Probe every candidate region once; record only conclusive results."""
    for region in dyn.region_map:
        result = prober(region)
        if result.terminate_error:
            typer.echo(
                f"Warning: probe instance {result.instance_id} in {region} may still be running "
                f"({result.terminate_error}); terminate it manually.",
                err=True,
            )
        if result.available is None:
            typer.echo(f"Probe {region}: inconclusive, not recorded ({result.detail})", err=True)
            continue
        dyn.trackers[region].record_probe(now, available=result.available, source="probe")
        typer.echo(f"Probe {region}: {'spot available' if result.available else 'no spot capacity'}")


def _run_dynamic_loop(
    dyn: DynamicJob,
    state: JobRuntimeState,
    runner: SkyRunner,
    *,
    clock: Clock,
    sleep: Sleep,
    poll_seconds: float,
    idle_tick_hours: float,
    max_launch_failures: int = 5,
    max_unknown_polls: int = 10,
    prober: Prober | None = None,
    probe_interval_hours: float = 0.0,
) -> DynamicLoopOutcome:
    """
    Time-constrained loop that routes each iteration to the matching ``DynamicJob`` method.

    All scheduling decisions stay in ``DynamicJob``; this function only observes the clock and
    the cluster (via ``runner``) and performs the launch/teardown side effects. With a
    ``prober`` and ``probe_interval_hours > 0`` it also runs launch-then-terminate spot probes
    inline every interval (the first one immediately); results feed the lifetime history.
    """
    job = dyn.job
    deadline = dyn.constraints.deadline_utc
    out = DynamicLoopOutcome(state=state)
    handle: SkyHandle | None = None
    seg_start: datetime | None = None
    launch_failures = 0
    unknown_polls = 0
    next_probe_at: datetime | None = None

    try:
        while state.progress_hours < job.duration_hours:
            now = clock()
            if now >= deadline:
                break

            if prober is not None and probe_interval_hours > 0:
                if next_probe_at is None or now >= next_probe_at:
                    _run_probe_round(dyn, prober, now)
                    next_probe_at = now + timedelta(hours=probe_interval_hours)

            if state.mode != JobMode.IDLE:
                assert handle is not None
                status = runner.poll(handle)
                if status is SkyStatus.UNKNOWN:
                    unknown_polls += 1
                    if unknown_polls > max_unknown_polls:
                        out.failed = f"could not read job status on {handle.cluster_name}"
                        break
                    sleep(poll_seconds)
                    continue
                unknown_polls = 0
                if status is SkyStatus.FAILED:
                    out.failed = f"job failed on {handle.cluster_name}"
                    break

                tick_hours = max((now - state.now_utc).total_seconds() / 3600.0, 0.0)
                if status is SkyStatus.PREEMPTED:
                    runner.teardown(handle)
                    if seg_start is not None:
                        out.segments.append((handle.region, seg_start, now))
                    handle, seg_start = None, None
                    state = dyn.advance_work(
                        replace(state, now_utc=now), tick_hours, preempted=True,
                    )
                elif status is SkyStatus.SUCCEEDED:
                    state = dyn.advance_work(state, tick_hours, completed=True)
                    runner.teardown(handle)
                    if seg_start is not None:
                        out.segments.append((handle.region, seg_start, now))
                    handle, seg_start = None, None
                else:
                    state = dyn.advance_work(state, tick_hours)
                sleep(poll_seconds)
                continue

            state = replace(state, now_utc=now)
            decision = dyn.decide(state)
            chosen = decision.chosen

            if chosen is None:
                state = dyn.wait_tick(state, decision, idle_tick_hours)
                sleep(idle_tick_hours * 3600.0)
                continue

            new_handle = runner.launch(chosen.region, chosen.mode)
            if new_handle is None:
                launch_failures += 1
                if chosen.mode == JobMode.SPOT:
                    dyn.trackers[chosen.region].record_probe(
                        now, available=False, source="launch"
                    )
                dyn.events.append(
                    DynamicPlanEvent(
                        time_utc=now,
                        event_type=DynamicEventType.IDLE,
                        region=chosen.region,
                        mode=JobMode.IDLE,
                        progress_hours=state.progress_hours,
                        cost_usd_delta=0.0,
                        carbon_kg_delta=0.0,
                        notes=f"launch failed: {chosen.region} {chosen.mode.value}",
                    )
                )
                if launch_failures >= max_launch_failures:
                    out.failed = f"launch failed {launch_failures} times in a row"
                    break
                sleep(poll_seconds)
                continue

            launch_failures = 0
            handle = new_handle
            if decision.safety_net:
                state = dyn.fallback_to_ondemand(state, decision, run_to_completion=False)
            else:
                state = dyn.launch_instance(state, decision)
            seg_start = clock()
            state = replace(state, now_utc=seg_start)
            sleep(poll_seconds)
    except KeyboardInterrupt:
        out.interrupted = True

    if handle is not None and seg_start is not None:
        end = clock()
        if end > seg_start:
            out.segments.append((handle.region, seg_start, end))
    out.state = state
    out.handle = handle
    return out


def _echo_actual_segments(
    job: JobSpec,
    watt_time: WattTimeClient,
    region_inputs: list[RegionPlanInput],
    segments: list[tuple[str, datetime, datetime]],
) -> None:
    """Sum historical-MOER carbon over each launched segment (estimates, not certified)."""
    if not segments:
        return
    wt_regions = {r.cloud_region: r.watttime_regions for r in region_inputs}
    estimator = JobCarbonEstimator(watt_time)
    total_co2 = 0.0
    total_cost = 0.0
    total_hours = 0.0
    for region, start, end in segments:
        try:
            actual = estimator.compute_actual_run(
                job=job,
                cloud="aws",
                cloud_region=region,
                watttime_regions=wt_regions[region],
                actual_start=start,
                actual_end=end,
                estimated_co2_kg=0.0,
            )
        except (KeyError, ValueError, WattTimeError) as err:
            typer.echo(f"Warning: could not compute actual CO₂ for {region}: {err}", err=True)
            continue
        total_co2 += actual.actual_co2_kg
        total_cost += actual.actual_cost_usd
        total_hours += actual.actual_duration_hours
    typer.echo(
        f"\nActual CO₂ (estimate over {len(segments)} segment(s)): {total_co2:.3f}kg | "
        f"Instance time: {total_hours:.1f}h | Cost: ${total_cost:.2f}"
    )


def _execute_dynamic_run(
    job: JobSpec,
    constraints: PlanConstraints,
    region_inputs: list[RegionPlanInput],
    runner: SkyRunner,
    watt_time: WattTimeClient,
    *,
    poll_seconds: float,
    idle_tick_hours: float,
    clock: Clock,
    sleep: Sleep,
    observations: Mapping[str, Sequence[SpotObservation]] | None = None,
    on_observation: Callable[[str, SpotObservation], None] | None = None,
    prober: Prober | None = None,
    probe_interval_hours: float = 0.0,
) -> int:
    """Run the dynamic loop against a real cluster. Returns a process exit code."""
    t0 = clock()  # time snapshot: constraints and DynamicJob are anchored here
    constraints = replace(constraints, now_utc=t0)
    events: list[DynamicPlanEvent] = []
    bucket_hours = _lifetime_bucket_hours(probe_interval_hours)
    dyn = DynamicJob(
        job,
        constraints,
        region_inputs,
        events,
        simulate_preemption=False,
        observations=observations,
        on_observation=on_observation,
        lifetime_bucket_hours=bucket_hours,
    )
    state = dyn.make_initial_state()

    typer.echo(
        f"\nDynamic run: deadline {constraints.deadline_utc.isoformat()}, "
        f"polling every {poll_seconds:g}s, idle tick {idle_tick_hours:g}h."
    )
    _echo_lifetime_prior(observations, bucket_hours)
    outcome = _run_dynamic_loop(
        dyn,
        state,
        runner,
        clock=clock,
        sleep=sleep,
        poll_seconds=poll_seconds,
        idle_tick_hours=idle_tick_hours,
        prober=prober,
        probe_interval_hours=probe_interval_hours,
    )
    result = dyn.finalize(outcome.state)
    _echo_dynamic_plan(result, title="Dynamic run (events; cost/carbon are estimates)")
    _echo_actual_segments(job, watt_time, region_inputs, outcome.segments)

    if outcome.handle is not None:
        name = outcome.handle.cluster_name
        typer.echo(
            f"Warning: cluster {name} is still up. Inspect or tear it down with: sky down {name}",
            err=True,
        )
    if outcome.interrupted:
        typer.echo("Interrupted.", err=True)
        return 130
    if outcome.failed:
        typer.echo(f"Error: {outcome.failed}", err=True)
        return 1
    if outcome.state.progress_hours < job.duration_hours:
        typer.echo("Error: finish-by deadline reached before the job finished.", err=True)
        return 1
    if (
        constraints.carbon_budget_kg is not None
        and result.total_carbon_kg > constraints.carbon_budget_kg
    ):
        typer.echo(
            f"Warning: estimated carbon {result.total_carbon_kg:.2f}kg exceeds budget "
            f"{constraints.carbon_budget_kg:.2f}kg.",
            err=True,
        )
    return 0


def _resolve_db_path(db_opt: Path | None) -> Path:
    if db_opt is not None:
        return db_opt
    env = os.environ.get("CARBONSIGHT_DB")
    if env:
        return Path(env)
    return Path.home() / ".carbonsight" / "runs.db"


def _lifetime_bucket_hours(probe_interval_hours: float) -> float:
    """A lifetime is only known to within the gap between probes, so bucket at that gap."""
    return probe_interval_hours if probe_interval_hours > 0 else DEFAULT_BUCKET_HOURS


def _load_spot_history(
    db_path: Path,
    regions: Sequence[str],
    instance_type: str,
) -> dict[str, list[SpotObservation]]:
    """Read-only history for the candidate regions; empty (with a warning) if unreadable."""
    if not instance_type:
        return {}
    try:
        with SpotObservationStore(db_path) as store:
            return store.history_for(regions, instance_type)
    except (sqlite3.Error, OSError) as err:
        typer.echo(f"Warning: could not read spot history ({err}); using assumed prior.", err=True)
        return {}


def _echo_lifetime_prior(
    observations: Mapping[str, Sequence[SpotObservation]] | None,
    bucket_hours: float,
) -> None:
    """One line saying where the spot lifetime estimate comes from."""
    trackers: list[VirtualInstanceTracker] = []
    for region, obs in (observations or {}).items():
        tracker = VirtualInstanceTracker(
            region=region, model=NelsonAalenModel(bucket_hours=bucket_hours)
        )
        replay_observations(tracker, obs)
        trackers.append(tracker)
    n_obs = sum(len(t.observations) for t in trackers)
    n_samples = sum(t.model.n_samples for t in trackers)
    if n_samples:
        detail = (
            f"from {n_obs} observations in local history ({n_samples} lifetime samples, "
            f"{len(trackers)} region(s)); not AWS-published data"
        )
    elif n_obs:
        detail = (
            f"assumed default {ASSUMED_SPOT_MEAN_LIFETIME_HOURS:g}h "
            f"({n_obs} observations, no lifetime samples yet)"
        )
    else:
        detail = f"assumed default {ASSUMED_SPOT_MEAN_LIFETIME_HOURS:g}h (no observations)"
    typer.echo(f"Spot lifetime prior: {detail}")


def _echo_finish_by_source(effective_finish_by: datetime, cli_finish_by: datetime | None) -> None:
    source = "CLI" if cli_finish_by is not None else "YAML"
    typer.echo(f"Finish-by ({source}): {effective_finish_by.isoformat()}")


def _build_dynamic_plan_inputs(
    reg: Registry,
    watt_time: WattTimeClient,
    job: JobSpec,
    estimates: list[EstimateResult],
    *,
    now_utc: datetime,
    effective_finish_by: datetime,
    effective_carbon_budget: float | None,
    effective_carbon_price: float,
    max_cost_premium: float,
) -> tuple[PlanConstraints, list[RegionPlanInput]]:
    """Constraints (deadline = finish-by) and per-region WattTime forecasts for dynamic planning."""
    if effective_finish_by <= now_utc:
        typer.echo("Error: finish-by must be in the future.", err=True)
        raise typer.Exit(1)

    visible, _, _ = AwsRegionRankingService.rank_greenest_first_then_cost_ceiling(
        estimates, max_cost_premium,
    )
    allowed = frozenset(e.cloud_region for e in visible)
    constraints = PlanConstraints.from_finish_by(
        effective_finish_by,
        now_utc,
        carbon_budget_kg=effective_carbon_budget,
        max_cost_premium=max_cost_premium,
        lambda_co2_usd_per_kg=effective_carbon_price,
    )
    try:
        region_inputs = build_region_plan_inputs(
            reg, watt_time, job, constraints, region_codes=allowed,
        )
    except WattTimeError as err:
        typer.echo(f"Error fetching forecasts for dynamic plan: {err}", err=True)
        raise typer.Exit(1)
    if not region_inputs:
        typer.echo("No regions with WattTime forecasts for dynamic planning.", err=True)
        raise typer.Exit(1)
    return constraints, region_inputs


def _pick_region_with_dynamic_plan(
    job: JobSpec,
    estimates: list[EstimateResult],
    constraints: PlanConstraints,
    region_inputs: list[RegionPlanInput],
    *,
    effective_finish_by: datetime,
    observations: Mapping[str, Sequence[SpotObservation]] | None = None,
    lifetime_bucket_hours: float = DEFAULT_BUCKET_HOURS,
) -> tuple[str, bool, EstimateResult, DynamicPlanResult]:
    # Same read-only history as the real loop; the simulation never persists anything.
    _echo_lifetime_prior(observations, lifetime_bucket_hours)
    dyn = plan_dynamic_job(
        job,
        constraints,
        region_inputs,
        observations=observations,
        lifetime_bucket_hours=lifetime_bucket_hours,
    )
    _echo_dynamic_plan(dyn)
    if not dyn.deadline_met or dyn.initial_launch is None:
        typer.echo("Dynamic plan: no feasible launch before finish-by.", err=True)
        raise typer.Exit(1)

    region = dyn.initial_launch.region
    use_spot = dyn.initial_launch.mode == JobMode.SPOT
    best = next((e for e in estimates if e.cloud_region == region), None)
    if best is None:
        typer.echo(f"No estimate row for planned region {region}.", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"\nChosen (dynamic): aws/{region}  "
        f"mode={dyn.initial_launch.mode.value}  "
        f"finish-by={effective_finish_by.isoformat()}"
    )
    return region, use_spot, best, dyn


def _pick_static_region(
    estimates: list[EstimateResult],
    max_cost_premium: float,
) -> tuple[str, str, EstimateResult]:
    best = AwsRegionRankingService.pick_best_region_for_launch(estimates, max_cost_premium)
    if best is None:
        typer.echo("No region selected.", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"\nChosen: {best.cloud}/{best.cloud_region}  "
        f"CO2={best.expected_co2_kg_mean:.2f}kg  "
        f"Cost=${best.expected_cost_usd:.2f}  "
        f"Confidence={best.mapping_confidence:.2f}"
    )
    return best.cloud, best.cloud_region, best


def _maybe_echo_carbon_schedule(
    reg: Registry,
    watt_time: WattTimeClient,
    job: JobSpec,
    region: str,
    max_delay: str,
) -> None:
    max_delay_hours = parse_duration_hours(max_delay)
    if max_delay_hours <= 0:
        return
    try:
        chosen_entry = reg.get_entry("aws", region)
        if chosen_entry is None or not chosen_entry.wt_regions:
            return
        horizon = math.ceil(max_delay_hours + job.duration_hours)
        forecast = watt_time.get_forecast(
            chosen_entry.wt_regions[0][0],
            horizon_hours=horizon,
        )
        pts = forecast.get("data", [])
        if not pts:
            return
        sched = pick_lowest_carbon_start(pts, job.duration_hours, max_delay_hours)
        if sched.delay_hours < 0.01:
            typer.echo("Schedule: run now (already the lowest-carbon window).")
        else:
            typer.echo(
                f"Schedule: wait {sched.delay_hours:.1f}h "
                f"(start {sched.start_utc:%H:%M} UTC) -> "
                f"-{sched.moer_reduction_pct:.1f}% carbon vs running now."
            )
    except Exception as e:
        typer.echo(f"Warning: could not compute schedule: {e}", err=True)


def run_launch(
    yaml_path: Path,
    *,
    dry_run: bool = False,
    no_exec: bool = False,
    skip_preflight: bool = False,
    managed: bool = False,
    yes: bool = False,
    max_cost_premium: float = 0.20,
    registry_path: Path | None = None,
    gpu_util: float | None = None,
    nvidia_smi: bool = False,
    use_spot: bool = True,
    max_delay: str = "0h",
    db_path: Path | None = None,
    checkpoint: bool = True,
    checkpoint_bucket: str | None = None,
    checkpoint_interval: int = 500,
    script_path: Path | None = None,
    live_pricing: bool = False,
    static_pricing: bool = False,
    finish_by: datetime | None = None,
    carbon_budget_kg: float | None = None,
    carbon_price: float | None = None,
    validate_sky: bool = False,
    sky_smoke: bool = False,
    poll_seconds: float = 60.0,
    idle_tick_hours: float = 1.0,
    clock: Clock | None = None,
    sleep: Sleep | None = None,
    probe_interval_hours: float = 0.0,
) -> None:
    """Shared implementation for ``run`` and ``train``.

    With a finish-by deadline the launch is an event loop (``_execute_dynamic_run``); ``clock``
    and ``sleep`` default to wall-clock time and are injectable for tests.
    """
    if poll_seconds <= 0 or idle_tick_hours <= 0:
        typer.echo("Error: --poll-seconds and --idle-tick-hours must be positive.", err=True)
        raise typer.Exit(1)
    if probe_interval_hours < 0:
        typer.echo("Error: --probe-interval-hours must be >= 0.", err=True)
        raise typer.Exit(1)

    if not yaml_path.exists():
        typer.echo(f"Error: file not found: {yaml_path}", err=True)
        raise typer.Exit(1)

    if gpu_util is not None and not (0.0 <= gpu_util <= 1.0):
        typer.echo("Error: --gpu-util must be between 0 and 1.", err=True)
        raise typer.Exit(1)

    job = job_spec_from_sky_yaml(yaml_path)
    job, nvidia_failed = apply_gpu_telemetry_cli(job, gpu_util=gpu_util, nvidia_smi=nvidia_smi)
    if nvidia_failed:
        typer.echo(
            "Warning: --nvidia-smi could not read utilization (no GPU driver here?). "
            "Using default sampled GPU power.",
            err=True,
        )
    package_root = carbonsight_package_root_from_cli_command_file(Path(__file__))
    rpath = resolve_registry_json_file(registry_path, carbonsight_package_root=package_root, cwd=Path.cwd())
    if not rpath.exists():
        typer.echo("No registry found. Use --registry or add seed_registry.json.", err=True)
        raise typer.Exit(1)

    reg = Registry()
    reg.load_json(rpath)

    config = Config.from_env()
    configure_pricing(
        config,
        live_aws_pricing=resolve_live_aws_pricing(
            config, live_pricing=live_pricing, static_pricing=static_pricing,
        ),
    )
    if not config.watttime_username or not config.watttime_password:
        typer.echo("Set WATTTIME_USERNAME and WATTTIME_PASSWORD.", err=True)
        raise typer.Exit(1)

    watt_time = WattTimeClient(config)
    quota_checker = QuotaChecker() if not skip_preflight else None
    availability_checker = InstanceAvailabilityChecker(config) if not skip_preflight else None
    enabled_regions_provider = EnabledRegionsProvider(config) if not skip_preflight else None
    ranking = AwsRegionRankingService(
        reg,
        watt_time,
        quota_checker=quota_checker,
        availability_checker=availability_checker,
        enabled_regions_provider=enabled_regions_provider,
    )

    def _enabled_region_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _quota_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _availability_skip(region_code: str, reason: str) -> None:
        typer.echo(f"  Skipping {region_code}: {reason}", err=True)

    def _estimate_warn(region_code: str, err: BaseException) -> None:
        typer.echo(f"  Warning {region_code}: {err}", err=True)

    estimates = ranking.collect_estimates(
        job,
        use_spot=use_spot,
        on_enabled_region_skip=_enabled_region_skip,
        on_quota_skip=_quota_skip,
        on_availability_skip=_availability_skip,
        on_estimate_error=_estimate_warn,
    )

    if not estimates:
        typer.echo(
            "No regions available after enabled-region, quota, availability, and WattTime checks.",
            err=True,
        )
        raise typer.Exit(1)

    carbonsight_block = carbonsight_block_from_yaml(yaml_path)
    effective_finish_by = resolve_finish_by(
        finish_by, yaml_path, carbonsight_block=carbonsight_block,
    )
    effective_carbon_budget = resolve_carbon_budget_kg(
        carbon_budget_kg, yaml_path, carbonsight_block=carbonsight_block,
    )
    effective_carbon_price = resolve_carbon_price(
        carbon_price, yaml_path, carbonsight_block=carbonsight_block,
    )

    cloud = "aws"
    region: str
    best: EstimateResult
    spot_instance_type = aws_instance_type_for_gpu(job.gpu_type) or ""
    spot_history: dict[str, list[SpotObservation]] = {}
    bucket_hours = _lifetime_bucket_hours(probe_interval_hours)

    if effective_finish_by is not None:
        _echo_finish_by_source(effective_finish_by, finish_by)
        plan_constraints, region_inputs = _build_dynamic_plan_inputs(
            reg,
            watt_time,
            job,
            estimates,
            now_utc=datetime.now(UTC),
            effective_finish_by=effective_finish_by,
            effective_carbon_budget=effective_carbon_budget,
            effective_carbon_price=effective_carbon_price,
            max_cost_premium=max_cost_premium,
        )
        spot_history = _load_spot_history(
            _resolve_db_path(db_path),
            [r.cloud_region for r in region_inputs],
            spot_instance_type,
        )
        region, use_spot, best, _dyn = _pick_region_with_dynamic_plan(
            job,
            estimates,
            plan_constraints,
            region_inputs,
            effective_finish_by=effective_finish_by,
            observations=spot_history,
            lifetime_bucket_hours=bucket_hours,
        )
    else:
        cloud, region, best = _pick_static_region(estimates, max_cost_premium)
        _maybe_echo_carbon_schedule(reg, watt_time, job, region, max_delay)

    baseline_cost = estimate_cost_usd(
        job.gpu_type, job.gpu_count, job.duration_hours, BASELINE_REGION, use_spot=use_spot,
    )
    baseline_est = next(
        (e for e in estimates if e.cloud_region == BASELINE_REGION),
        None,
    )
    baseline_co2 = baseline_est.expected_co2_kg_mean if baseline_est else best.expected_co2_kg_mean

    rec = RunRecord(
        run_id=uuid.uuid4().hex[:12],
        created_utc=datetime.now(UTC),
        cloud_region=region,
        gpu_type=job.gpu_type,
        gpu_count=job.gpu_count,
        duration_hours=job.duration_hours,
        use_spot=use_spot,
        baseline_region=BASELINE_REGION,
        estimated_co2_kg=best.expected_co2_kg_mean,
        baseline_co2_kg=baseline_co2,
        estimated_cost_usd=best.expected_cost_usd,
        baseline_cost_usd=baseline_cost,
    )
    ledger = RunLedger(_resolve_db_path(db_path))
    ledger.record(rec)

    ckpt_patch = None
    if checkpoint:
        raw = yaml.safe_load(yaml_path.read_text()) or {}
        script_source = ""
        if script_path and script_path.is_file():
            script_source = script_path.read_text()
        else:
            extracted = extract_script_path_from_run_command(raw.get("run", ""))
            if extracted:
                candidate = yaml_path.parent / extracted
                if not candidate.is_file():
                    candidate = Path.cwd() / extracted
                if candidate.is_file():
                    script_source = candidate.read_text()

        framework = detect_framework(script_source) if script_source else Framework.UNKNOWN
        task_name = raw.get("name", "carbonsight-job")
        original_run = raw.get("run", "").strip()

        ckpt_cfg = build_checkpoint_config(
            task_name, framework,
            save_interval_steps=checkpoint_interval,
            bucket_uri=checkpoint_bucket,
        )
        ckpt_patch = build_checkpoint_yaml_patch(ckpt_cfg, original_run, shim_local_path())

        # The dynamic loop owns recovery (unmanaged launches), so managed only applies to static runs.
        if use_spot and not managed and effective_finish_by is None:
            managed = True
            typer.echo("Auto-enabling --managed for spot + checkpoint resilience.")

        typer.echo(f"Checkpoint: framework={framework.value}, storage={ckpt_cfg.storage_name or checkpoint_bucket}")

    patched = build_patched_yaml_text(
        yaml_path, cloud, region, use_spot=use_spot, ckpt_patch=ckpt_patch,
    )

    sky_cwd = yaml_path.parent.resolve()

    if dry_run:
        typer.echo("\n--- Patched YAML for SkyPilot (dry run) ---")
        typer.echo(patched)
        return

    if no_exec:
        typer.echo("\n--- Patched YAML for SkyPilot (--no-exec) ---")
        typer.echo(patched)
        typer.echo("Skipping launch (--no-exec).")
        return

    if validate_sky:
        _require_sky_cli()
        typer.echo("\nValidating patched YAML with SkyPilot (`sky launch --dryrun`)...")
        exit_code = launch_skypilot_with_patched_yaml(
            patched, managed=managed, yes=True, dryrun=True, cwd=sky_cwd,
        )
        if exit_code == 0:
            typer.echo("SkyPilot accepted the patched YAML.")
        raise typer.Exit(exit_code)

    if sky_smoke:
        _require_sky_cli()
        smoke_data = yaml.safe_load(patched) or {}
        smoke_data["run"] = "echo 'carbonsight sky-smoke ok'\n"
        smoke_data.pop("file_mounts", None)
        smoke_data.pop("envs", None)
        setup = smoke_data.get("setup")
        if isinstance(setup, str) and "/ckpt" in setup:
            smoke_data["setup"] = "echo 'carbonsight sky-smoke setup'\n"
        smoke_yaml = yaml.dump(smoke_data, default_flow_style=False, sort_keys=False)
        typer.echo("\nSky smoke test: minimal job with `sky launch --down` (auto teardown)...")
        exit_code = launch_skypilot_with_patched_yaml(
            smoke_yaml, managed=False, yes=True, down=True, cwd=sky_cwd,
        )
        raise typer.Exit(exit_code)

    _require_sky_cli()

    if effective_finish_by is not None:
        # Deadline path: monitored event loop (launch, poll, relaunch after preemption).
        runner = SkyRunner(
            lambda launch_region, spot: build_patched_yaml_text(
                yaml_path,
                cloud,
                launch_region,
                use_spot=spot,
                ckpt_patch=ckpt_patch,
                force_on_demand=not spot,
            ),
            cwd=sky_cwd,
            run_id=rec.run_id,
            yes=yes,
        )
        prober: Prober | None = None
        if probe_interval_hours > 0:
            if not spot_instance_type:
                typer.echo(
                    f"Warning: no EC2 instance type known for {job.gpu_type!r}; probing disabled.",
                    err=True,
                )
            else:
                spot_probe = SpotLaunchProbe(config)

                def _probe_region(probe_region: str) -> ProbeResult:
                    return spot_probe.probe(probe_region, job.gpu_type)

                prober = _probe_region

                typer.echo(
                    f"Probing spot capacity every {probe_interval_hours:g}h per region by launching "
                    f"{spot_instance_type} and terminating it immediately. This bills real "
                    "instances and needs vCPU quota."
                )

        observation_store: SpotObservationStore | None = None
        try:
            if spot_instance_type:
                try:
                    observation_store = SpotObservationStore(_resolve_db_path(db_path))
                except (sqlite3.Error, OSError) as err:
                    typer.echo(
                        f"Warning: spot history not persisted ({err}).", err=True,
                    )

            def _persist_observation(obs_region: str, obs: SpotObservation) -> None:
                if observation_store is None:
                    return
                try:
                    observation_store.add(
                        StoredObservation(
                            observed_utc=obs.time_utc,
                            region=obs_region,
                            instance_type=spot_instance_type,
                            available=obs.available,
                            source=obs.source,
                            run_id=rec.run_id,
                        )
                    )
                except sqlite3.Error as err:
                    typer.echo(f"Warning: could not persist spot observation ({err}).", err=True)

            loop_exit_code = _execute_dynamic_run(
                job,
                plan_constraints,
                region_inputs,
                runner,
                watt_time,
                poll_seconds=poll_seconds,
                idle_tick_hours=idle_tick_hours,
                clock=clock or (lambda: datetime.now(UTC)),
                sleep=sleep or time.sleep,
                observations=spot_history,
                on_observation=_persist_observation,
                prober=prober,
                probe_interval_hours=probe_interval_hours,
            )
        finally:
            if observation_store is not None:
                observation_store.close()
        if loop_exit_code != 0:
            raise typer.Exit(loop_exit_code)
        return

    start_time = datetime.now(UTC)
    exit_code = launch_skypilot_with_patched_yaml(
        patched, managed=managed, yes=yes, cwd=sky_cwd,
    )
    if exit_code != 0:
        raise typer.Exit(exit_code)

    end_time = datetime.now(UTC)
    try:
        actual = JobCarbonEstimator(watt_time).compute_actual_run(
            job=job,
            cloud=cloud,
            cloud_region=region,
            watttime_regions=best.watttime_regions,
            actual_start=start_time,
            actual_end=end_time,
            estimated_co2_kg=best.expected_co2_kg_mean,
        )
        pct = actual.co2_savings_vs_estimate_pct
        sign = "+" if pct >= 0 else ""
        typer.echo(
            f"\nActual CO₂: {actual.actual_co2_kg:.3f}kg | "
            f"Duration: {actual.actual_duration_hours:.1f}h | "
            f"Cost: ${actual.actual_cost_usd:.2f} | "
            f"vs estimate: {sign}{pct:.1f}%"
        )
    except (ValueError, WattTimeError) as e:
        typer.echo(f"\nWarning: could not compute actual CO₂: {e}", err=True)


def run_cmd(
    yaml_path: Path = typer.Argument(..., help="Path to SkyPilot-style job YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print patched YAML only, do not launch"),
    no_exec: bool = typer.Option(False, "--no-exec", help="Pick region and patch YAML but skip launch"),
    skip_preflight: bool = typer.Option(
        False,
        "--skip-preflight",
        help="Skip AWS preflight (enabled regions, GPU quota, instance offerings)",
    ),
    managed: bool = typer.Option(False, "--managed", help="Use sky jobs launch (managed spot) instead of sky launch"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Pass --yes to SkyPilot (skip confirmation prompt)"),
    max_cost_premium: float = typer.Option(
        0.20, "--max-cost-premium",
        help="Max extra cost vs cheapest region (0.20 = up to 20%% more). Greener region wins within budget.",
    ),
    registry_path: Path | None = typer.Option(None, "--registry", help="Override path to registry JSON"),
    gpu_util: float | None = typer.Option(
        None,
        "--gpu-util",
        help="Observed GPU utilization in [0, 1] for the power model.",
    ),
    nvidia_smi: bool = typer.Option(
        False,
        "--nvidia-smi",
        help="Sample GPU utilization from nvidia-smi on this machine (overrides YAML when successful).",
    ),
    spot: bool = typer.Option(
        True,
        "--spot/--no-spot",
        help="Use spot instances for cost estimates and YAML patch (default: on).",
    ),
    max_delay: str = typer.Option(
        "0h",
        "--max-delay",
        help="Max hours to delay start for greener carbon (e.g. 6h). 0h = run immediately.",
    ),
    db_path: Path | None = typer.Option(
        None, "--db", help="SQLite DB path for run tracking (default: ~/.carbonsight/runs.db).",
    ),
    checkpoint: bool = typer.Option(
        True,
        "--checkpoint/--no-checkpoint",
        help="Enable checkpoint resilience for spot training (default: on).",
    ),
    checkpoint_bucket: str | None = typer.Option(
        None,
        "--checkpoint-bucket",
        help="S3 bucket URI for checkpoint storage (default: auto-managed SkyPilot Storage).",
    ),
    checkpoint_interval: int = typer.Option(
        500,
        "--checkpoint-interval",
        help="Save checkpoint every N training steps.",
    ),
    live_pricing: bool = typer.Option(
        False,
        "--live-pricing",
        help="Use live AWS EC2 spot prices for cost estimates (when --spot).",
    ),
    static_pricing: bool = typer.Option(
        False,
        "--static-pricing",
        help="Force static cost tables instead of live AWS pricing.",
    ),
    finish_by: str | None = typer.Option(
        None,
        "--finish-by",
        help="ISO-8601 finish-by deadline (enables dynamic planner). When omitted, uses YAML carbonsight.finish_by.",
    ),
    carbon_budget: float | None = typer.Option(
        None,
        "--carbon-budget",
        help="Hard carbon budget in kg CO2 for dynamic planning. When omitted, uses YAML carbonsight.carbon_budget_kg.",
    ),
    carbon_price: float | None = typer.Option(
        None,
        "--carbon-price",
        help="Shadow price in USD per kg CO2 for dynamic utility scoring. When omitted, uses YAML carbonsight.carbon_price.",
    ),
    validate_sky: bool = typer.Option(
        False,
        "--validate-sky",
        help="After patching, run `sky launch --dryrun` to verify SkyPilot accepts the YAML.",
    ),
    sky_smoke: bool = typer.Option(
        False,
        "--sky-smoke",
        help="Launch a minimal echo job via SkyPilot with `--down` (provisions briefly, then tears down).",
    ),
    poll_seconds: float = typer.Option(
        60.0,
        "--poll-seconds",
        help="With a finish-by deadline: seconds between SkyPilot job status checks.",
    ),
    idle_tick_hours: float = typer.Option(
        1.0,
        "--idle-tick-hours",
        help="With a finish-by deadline: hours to wait between decisions while waiting for a better launch window.",
    ),
    probe_interval_hours: float = typer.Option(
        0.0,
        "--probe-interval-hours",
        help=(
            "With a finish-by deadline: every N hours launch one spot instance per candidate "
            "region and terminate it immediately to record availability (0 = off). COST: "
            "successful probes bill real GPU instances and need vCPU quota."
        ),
    ),
) -> None:
    """Pick the greenest affordable region, patch YAML, and launch via SkyPilot."""
    parsed_finish_by = parse_finish_by(finish_by) if finish_by else None
    run_launch(
        yaml_path,
        dry_run=dry_run,
        no_exec=no_exec,
        skip_preflight=skip_preflight,
        managed=managed,
        yes=yes,
        max_cost_premium=max_cost_premium,
        registry_path=registry_path,
        gpu_util=gpu_util,
        nvidia_smi=nvidia_smi,
        use_spot=spot,
        max_delay=max_delay,
        db_path=db_path,
        checkpoint=checkpoint,
        checkpoint_bucket=checkpoint_bucket,
        checkpoint_interval=checkpoint_interval,
        live_pricing=live_pricing,
        static_pricing=static_pricing,
        finish_by=parsed_finish_by,
        carbon_budget_kg=carbon_budget,
        carbon_price=carbon_price,
        validate_sky=validate_sky,
        sky_smoke=sky_smoke,
        poll_seconds=poll_seconds,
        idle_tick_hours=idle_tick_hours,
        probe_interval_hours=probe_interval_hours,
    )
