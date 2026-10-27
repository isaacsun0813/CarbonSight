"""Dynamic event loop in ``run.py``: fake clock + fake SkyPilot runner (no real sky calls)."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

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
    def test_prior_line_assumed_without_history(self, capsys) -> None:
        run_mod._echo_lifetime_prior({}, 0.25)
        assert "assumed default 4h (no observations)" in capsys.readouterr().out

    def test_load_history_and_prior_line_from_store(self, tmp_path, capsys) -> None:
        from carbonsight_core.spot_history import SpotObservationStore, StoredObservation

        db = tmp_path / "runs.db"
        with SpotObservationStore(db) as store:
            store.add_many(
                [
                    StoredObservation(T0, REGION, "p4d.24xlarge", True, "launch", "r1"),
                    StoredObservation(
                        T0 + timedelta(hours=3), REGION, "p4d.24xlarge", False, "preemption", "r1"
                    ),
                ]
            )
        history = run_mod._load_spot_history(db, [REGION, "eu-west-1"], "p4d.24xlarge")
        assert len(history[REGION]) == 2 and history["eu-west-1"] == []
        run_mod._echo_lifetime_prior(history, 0.25)
        out = capsys.readouterr().out
        assert "from 2 observations" in out and "1 lifetime samples" in out
        assert run_mod._load_spot_history(db, [REGION], "") == {}

    def test_advisor_bands_printed_with_honest_label(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(
            run_mod, "fetch_advisor_bands", lambda itype, regions: {REGION: 4}
        )
        bands = run_mod._load_advisor_bands("p4d.24xlarge", [REGION])
        assert bands == {REGION: 4}
        out = capsys.readouterr().out
        assert ">20%" in out and "relative regional weights" in out

    def test_advisor_failure_fails_open(self, monkeypatch, capsys) -> None:
        def boom(itype, regions):
            raise run_mod.SpotAdvisorError("down")

        monkeypatch.setattr(run_mod, "fetch_advisor_bands", boom)
        assert run_mod._load_advisor_bands("p4d.24xlarge", [REGION]) is None
        assert "spot advisor unavailable" in capsys.readouterr().err

    def test_advisor_skipped_without_instance_type(self) -> None:
        assert run_mod._load_advisor_bands("", [REGION]) is None

    def test_prior_line_mentions_advisor(self, capsys) -> None:
        run_mod._echo_lifetime_prior({}, 0.25, {REGION: 1})
        assert "scaled by advisor band" in capsys.readouterr().out

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
            advisor_bands={REGION: 4},
            now=T0,
        )
        assert calls == [REGION, "eu-west-1", "ap-south-1"]  # exactly one probe per region
        assert [(o.available, o.source) for o in history[REGION]] == [(True, "probe")]
        assert [o.available for o in history["eu-west-1"]] == [False]
        assert history["ap-south-1"] == []  # inconclusive is not recorded
        with SpotObservationStore(db) as store:
            rows = store.observations(REGION, "p4d.24xlarge")
            assert [(r.source, r.advisor_band) for r in rows] == [("probe", 4)]
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
