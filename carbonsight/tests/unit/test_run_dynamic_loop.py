"""Dynamic event loop in ``run.py``: fake clock + fake SkyPilot runner (no real sky calls)."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import typer
from carbonsight_core.cloud.aws.probe import ProbeResult
from carbonsight_core.models import JobSpec
from carbonsight_core.planning import (
    DynamicEventType,
    DynamicJob,
    JobMode,
    PlanConstraints,
    RegionPlanInput,
    SchedulerCandidate,
    SchedulerDecision,
)

from carbonsight_cli.commands import run as run_mod
from carbonsight_cli.commands.run import (
    SkyHandle,
    SkyRunner,
    SkyStatus,
    _run_dynamic_loop,
    parse_sky_cluster_status,
    parse_sky_queue_job_state,
)

UTC = timezone.utc
T0 = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
REGION = "us-west-2"


class FakeClock:
    """``sleep`` advances the clock, so the loop runs instantly."""

    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeRunner:
    """Scripted SkyRunner: ``launches`` gives launch results, ``polls`` per-launch status lists."""

    def __init__(self, launches: list[bool], polls: list[list[SkyStatus]]) -> None:
        self._launch_ok = list(launches)
        self._polls = [list(p) for p in polls]
        self.launched: list[SkyHandle] = []
        self.launch_requests: list[tuple[str, JobMode]] = []
        self.torn_down: list[SkyHandle] = []
        self._current = -1

    def launch(self, region: str, mode: JobMode) -> SkyHandle | None:
        self.launch_requests.append((region, mode))
        ok = self._launch_ok.pop(0) if self._launch_ok else True
        if not ok:
            return None
        self._current += 1
        handle = SkyHandle(cluster_name=f"cs-test-{self._current}", region=region, mode=mode)
        self.launched.append(handle)
        return handle

    def poll(self, handle: SkyHandle) -> SkyStatus:
        script = self._polls[self.launched.index(handle)]
        return script.pop(0) if len(script) > 1 else script[0]

    def teardown(self, handle: SkyHandle) -> None:
        self.torn_down.append(handle)


class ScriptedJob(DynamicJob):
    """DynamicJob whose ``decide`` returns scripted spot launches (deterministic for tests)."""

    def decide(self, state):  # type: ignore[override]
        return _spot_decision()


def _spot_decision() -> SchedulerDecision:
    chosen = SchedulerCandidate(
        region=REGION,
        mode=JobMode.SPOT,
        cost_rate_usd_per_hr=1.0,
        carbon_rate_kg_per_hr=0.1,
        expected_lifetime_hours=4.0,
    )
    return SchedulerDecision(chosen=chosen, safety_net=False, utility=1.0, reason="test")


def _regions() -> list[RegionPlanInput]:
    return [
        RegionPlanInput(
            cloud_region=REGION,
            watttime_regions=[("PACW", 1.0)],
            forecast_points=[
                {"point_time": (T0 + timedelta(hours=i)).isoformat(), "value": 200.0}
                for i in range(48)
            ],
        ),
    ]


def _constraints(deadline_hours: float, cold_start_hours: float = 0.0) -> PlanConstraints:
    return PlanConstraints(
        deadline_utc=T0 + timedelta(hours=deadline_hours),
        now_utc=T0,
        cold_start_hours=cold_start_hours,
    )


def _loop(dyn: DynamicJob, runner: FakeRunner, clock: FakeClock, **kwargs):
    return _run_dynamic_loop(
        dyn,
        dyn.make_initial_state(),
        runner,  # type: ignore[arg-type]
        clock=clock,
        sleep=clock.sleep,
        poll_seconds=kwargs.pop("poll_seconds", 900.0),
        idle_tick_hours=kwargs.pop("idle_tick_hours", 1.0),
        **kwargs,
    )


def _types(dyn: DynamicJob) -> list[DynamicEventType]:
    return [e.event_type for e in dyn.events]


class TestDynamicLoop:
    def test_launch_then_succeed(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = ScriptedJob(job, _constraints(10.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([True], [[SkyStatus.RUNNING, SkyStatus.RUNNING, SkyStatus.SUCCEEDED]])
        out = _loop(dyn, runner, FakeClock(T0))

        assert out.failed is None and not out.interrupted
        assert out.state.progress_hours >= job.duration_hours
        assert _types(dyn) == [DynamicEventType.LAUNCH, DynamicEventType.COMPLETE]
        assert runner.torn_down == runner.launched
        assert out.handle is None
        assert len(out.segments) == 1
        assert dyn.finalize(out.state).deadline_met

    def test_early_real_preemption_uses_observed_launch_times(self) -> None:
        clock = FakeClock(T0)

        class ShortLaunchRunner(FakeRunner):
            def launch(self, region, mode):
                clock.sleep(60)
                return super().launch(region, mode)

        seen = []
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2)
        dyn = ScriptedJob(
            job, _constraints(10, cold_start_hours=0.5), _regions(), [],
            simulate_preemption=False, on_observation=lambda region, obs: seen.append(obs),
        )
        runner = ShortLaunchRunner(
            [True, True],
            [[SkyStatus.RUNNING, SkyStatus.PREEMPTED], [SkyStatus.SUCCEEDED]],
        )
        out = _loop(dyn, runner, clock, poll_seconds=60)

        assert out.failed is None and out.handle is None
        assert runner.torn_down == runner.launched
        assert [(obs.source, obs.time_utc) for obs in seen] == [
            ("launch", T0 + timedelta(seconds=60)),
            ("preemption", T0 + timedelta(seconds=180)),
            ("launch", T0 + timedelta(seconds=300)),
        ]
        assert dyn.trackers[REGION].model.complete_samples == [2 / 60]
        assert dyn.events[0].time_utc == T0 + timedelta(seconds=60)

    def test_preempt_then_relaunch(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        dyn = ScriptedJob(job, _constraints(20.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner(
            [True, True],
            [[SkyStatus.RUNNING, SkyStatus.PREEMPTED], [SkyStatus.SUCCEEDED]],
        )
        out = _loop(dyn, runner, FakeClock(T0))

        assert _types(dyn) == [
            DynamicEventType.LAUNCH,
            DynamicEventType.PREEMPT,
            DynamicEventType.LAUNCH,
            DynamicEventType.COMPLETE,
        ]
        assert len(runner.launched) == 2
        assert runner.torn_down == runner.launched
        assert len(out.segments) == 2
        assert out.failed is None
        assert dyn.finalize(out.state).deadline_met

    def test_safety_net_launches_ondemand_without_simulating_completion(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=2.0)
        # slack (2.5h - 2h) < 2 * cold_start (1h): the Safety Net fires immediately
        dyn = DynamicJob(
            job, _constraints(2.5, cold_start_hours=0.5), _regions(), [], simulate_preemption=False,
        )
        runner = FakeRunner([True], [[SkyStatus.RUNNING, SkyStatus.SUCCEEDED]])
        out = _loop(dyn, runner, FakeClock(T0), poll_seconds=60.0)

        assert runner.launch_requests == [(REGION, JobMode.ON_DEMAND)]
        assert _types(dyn)[0] == DynamicEventType.SAFETY_NET
        assert dyn.events[0].time_utc == T0
        assert _types(dyn)[-1] == DynamicEventType.COMPLETE
        assert len(out.segments) == 1

    def test_launch_failure_retries_then_launches(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = ScriptedJob(job, _constraints(10.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([False, True], [[SkyStatus.SUCCEEDED]])
        out = _loop(dyn, runner, FakeClock(T0))

        assert _types(dyn) == [
            DynamicEventType.IDLE,
            DynamicEventType.LAUNCH,
            DynamicEventType.COMPLETE,
        ]
        assert "launch failed" in dyn.events[0].notes
        assert out.failed is None

    def test_repeated_launch_failures_give_up(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = ScriptedJob(job, _constraints(10.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([False] * 5, [[SkyStatus.RUNNING]])
        out = _loop(dyn, runner, FakeClock(T0), max_launch_failures=3)

        assert out.failed is not None and "launch failed" in out.failed
        assert runner.launched == []

    def test_deadline_with_job_still_running(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=10.0)
        dyn = ScriptedJob(job, _constraints(1.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([True], [[SkyStatus.RUNNING]])
        out = _loop(dyn, runner, FakeClock(T0))

        assert out.state.progress_hours < job.duration_hours
        assert out.handle is not None  # left up; run_launch tells the user to `sky down`
        assert len(out.segments) == 1
        result = dyn.finalize(out.state)
        assert not result.deadline_met
        assert result.events[-1].event_type == DynamicEventType.MISSED_DEADLINE

    def test_unreadable_status_gives_up(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = ScriptedJob(job, _constraints(10.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([True], [[SkyStatus.UNKNOWN]])
        out = _loop(dyn, runner, FakeClock(T0), max_unknown_polls=3)

        assert out.failed is not None and "job status" in out.failed

    def test_failed_job_stops_loop(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = ScriptedJob(job, _constraints(10.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([True], [[SkyStatus.FAILED]])
        out = _loop(dyn, runner, FakeClock(T0))

        assert out.failed is not None and "job failed" in out.failed

    def test_wait_when_nothing_to_launch(self) -> None:
        class WaitingJob(DynamicJob):
            def decide(self, state):  # type: ignore[override]
                return SchedulerDecision(chosen=None, safety_net=False, utility=0.0, reason="wait")

        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = WaitingJob(job, _constraints(3.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([], [])
        out = _loop(dyn, runner, FakeClock(T0), idle_tick_hours=1.0)

        assert runner.launch_requests == []
        assert set(_types(dyn)) == {DynamicEventType.IDLE}
        assert len(dyn.events) == 3  # one per hour until the deadline
        assert out.state.progress_hours == 0.0


class TestLaunchObservations:
    def test_failed_spot_launch_is_tagged_launch(self) -> None:
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        dyn = ScriptedJob(job, _constraints(10.0), _regions(), [], simulate_preemption=False)
        runner = FakeRunner([False, True], [[SkyStatus.SUCCEEDED]])
        _loop(dyn, runner, FakeClock(T0))
        sources = [(o.source, o.available) for o in dyn.trackers[REGION].observations]
        assert sources == [("launch", False), ("launch", True)]


class TestLifetimePriorWiring:
    def test_prior_line_matches_shared_model_provenance(self, capsys) -> None:
        dyn = DynamicJob(
            JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1),
            _constraints(10), _regions(), [],
        )
        run_mod._echo_lifetime_prior(dyn.prior_sources())
        out = capsys.readouterr().out
        assert "assumed exponential mean 4h" in out
        assert "bounded by job/deadline horizon" in out

    def test_probe_before_plan_feeds_history_and_store(self, tmp_path) -> None:
        from carbonsight_core.spot_history import SpotObservationStore

        db = tmp_path / "runs.db"
        calls: list[str] = []
        results = {REGION: True, "eu-west-1": False, "ap-south-1": None}

        def prober(region: str) -> ProbeResult:
            calls.append(region)
            return ProbeResult(region, "p4d.24xlarge", results[region], "fake")

        history: dict = {REGION: [], "eu-west-1": [], "ap-south-1": []}
        run_mod._probe_before_plan(
            prober,
            [REGION, "eu-west-1", "ap-south-1"],
            history,
            store_path=db,
            instance_type="p4d.24xlarge",
            now=T0,
        )
        assert calls == [REGION, "eu-west-1", "ap-south-1"]  # exactly one probe per region
        assert [(o.available, o.source) for o in history[REGION]] == [(True, "probe")]
        assert [o.available for o in history["eu-west-1"]] == [False]
        assert history["ap-south-1"] == []  # inconclusive is not recorded
        with SpotObservationStore(db) as store:
            rows = store.observations(REGION, "p4d.24xlarge")
            assert [(r.source, r.advisor_band) for r in rows] == [("probe", None)]
            assert len(store.observations("eu-west-1", "p4d.24xlarge")) == 1
            assert store.observations("ap-south-1", "p4d.24xlarge") == []

    def test_make_prober_needs_instance_type(self, capsys) -> None:
        assert run_mod._make_prober(object(), "UNKNOWN", "") is None
        assert "probing disabled" in capsys.readouterr().err


class TestSkyOutputParsers:
    def test_queue_job_state(self) -> None:
        text = (
            "Job queue of cluster cs-abc-1\n"
            "ID  NAME  SUBMITTED  STARTED  DURATION  RESOURCES  STATUS  LOG\n"
            "1   task  1 min ago  1 min    -         1x[A100]   RUNNING sky logs 1\n"
        )
        assert parse_sky_queue_job_state(text) == "RUNNING"
        assert parse_sky_queue_job_state(text.replace("RUNNING", "SUCCEEDED")) == "SUCCEEDED"
        assert parse_sky_queue_job_state("No jobs.\n") is None

    def test_cluster_status(self) -> None:
        text = (
            "NAME       LAUNCHED   RESOURCES   STATUS  AUTOSTOP  COMMAND\n"
            "cs-abc-1   1 min ago  1x AWS(A100) UP      10m (down) sky launch\n"
        )
        assert parse_sky_cluster_status(text, "cs-abc-1") == "UP"
        assert parse_sky_cluster_status(text, "cs-abc-2") is None
        assert parse_sky_cluster_status("No existing clusters.\n", "cs-abc-1") is None


def _cp(returncode: int, stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


class TestSkyRunner:
    HANDLE = SkyHandle(cluster_name="cs-abc-1", region=REGION, mode=JobMode.SPOT)

    def _runner(self, run_fn, launch_fn=None) -> SkyRunner:
        return SkyRunner(
            lambda region, spot: f"yaml {region} {spot}",
            cwd=None,
            run_id="abc",
            run_fn=run_fn,
            launch_fn=launch_fn or (lambda *a, **k: 0),
        )

    def test_poll_reads_job_state(self) -> None:
        runner = self._runner(lambda cmd, **k: _cp(0, "1 task RUNNING"))
        assert runner.poll(self.HANDLE) is SkyStatus.RUNNING
        runner = self._runner(lambda cmd, **k: _cp(0, "1 task SUCCEEDED"))
        assert runner.poll(self.HANDLE) is SkyStatus.SUCCEEDED
        runner = self._runner(lambda cmd, **k: _cp(0, "1 task FAILED_SETUP"))
        assert runner.poll(self.HANDLE) is SkyStatus.FAILED

    def test_poll_missing_cluster_is_preemption(self) -> None:
        def run_fn(cmd, **k):
            return _cp(1) if cmd[1] == "queue" else _cp(0, "No existing clusters.\n")

        assert self._runner(run_fn).poll(self.HANDLE) is SkyStatus.PREEMPTED

    def test_poll_transient_errors_are_unknown(self) -> None:
        up = "cs-abc-1 1 min ago 1x AWS UP 10m"

        def queue_blip(cmd, **k):
            return _cp(1) if cmd[1] == "queue" else _cp(0, up)

        assert self._runner(queue_blip).poll(self.HANDLE) is SkyStatus.UNKNOWN
        assert self._runner(lambda cmd, **k: _cp(1)).poll(self.HANDLE) is SkyStatus.UNKNOWN

    def test_launch_success_names_cluster_and_stops_reprompting(self) -> None:
        calls: list[dict] = []

        def launch_fn(patched, **kwargs):
            calls.append({"patched": patched, **kwargs})
            return 0

        runner = self._runner(lambda cmd, **k: _cp(0), launch_fn)
        h1 = runner.launch("us-west-2", JobMode.SPOT)
        h2 = runner.launch("us-east-1", JobMode.ON_DEMAND)

        assert h1 is not None and h1.cluster_name == "cs-abc-1"
        assert h2 is not None and h2.cluster_name == "cs-abc-2"
        assert calls[0]["patched"] == "yaml us-west-2 True"
        assert calls[1]["patched"] == "yaml us-east-1 False"
        assert calls[0]["detach"] and calls[0]["down"] and not calls[0]["managed"]
        assert calls[0]["yes"] is False and calls[1]["yes"] is True

    def test_launch_failure_tears_down_and_returns_none(self) -> None:
        commands: list[list[str]] = []

        def run_fn(cmd, **k):
            commands.append(cmd)
            return _cp(0)

        runner = self._runner(run_fn, lambda *a, **k: 1)
        assert runner.launch("us-west-2", JobMode.SPOT) is None
        assert commands == [["sky", "down", "--yes", "cs-abc-1"]]


def test_execute_dynamic_run_exit_codes(monkeypatch) -> None:
    monkeypatch.setattr(run_mod, "_echo_actual_segments", lambda *a, **k: None)
    job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
    clock = FakeClock(T0)

    class ScriptedRun(DynamicJob):
        def decide(self, state):  # type: ignore[override]
            return _spot_decision()

    monkeypatch.setattr(run_mod, "DynamicJob", ScriptedRun)

    ok = run_mod._execute_dynamic_run(
        job,
        _constraints(10.0),
        _regions(),
        FakeRunner([True], [[SkyStatus.SUCCEEDED]]),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        poll_seconds=900.0,
        idle_tick_hours=1.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert ok == 0

    clock = FakeClock(T0)
    late = run_mod._execute_dynamic_run(
        JobSpec(gpu_type="A100", gpu_count=1, duration_hours=10.0),
        _constraints(1.0),
        _regions(),
        FakeRunner([True], [[SkyStatus.RUNNING]]),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        poll_seconds=900.0,
        idle_tick_hours=1.0,
        clock=clock,
        sleep=clock.sleep,
    )
    assert late == 1


class TestBudgetVsDeadlineMessages:
    def _call(self, budget: float | None):
        job = JobSpec(gpu_type="A100", gpu_count=1, duration_hours=1.0)
        constraints = _constraints(8.0)
        constraints = replace(constraints, carbon_budget_kg=budget)
        return run_mod._pick_region_with_dynamic_plan(
            job, [], constraints, _regions(), effective_finish_by=T0 + timedelta(hours=8)
        )

    def test_budget_overrun_is_not_reported_as_infeasible_deadline(self, capsys) -> None:
        with pytest.raises(typer.Exit):
            self._call(1e-9)
        err = capsys.readouterr().err
        assert "exceeds the budget" in err
        assert "no feasible launch" not in err
