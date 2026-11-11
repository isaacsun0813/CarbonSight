"""SpotLaunchProbe with a fake EC2 client (no AWS calls) and the static SPS request map."""

import pytest
from carbonsight_core.cloud.aws import probe as probe_mod
from carbonsight_core.cloud.aws.gpu_catalog import spot_placement_score_request
from carbonsight_core.cloud.aws.probe import PROBE_TAG_KEY, SpotLaunchProbe
from carbonsight_core.config import Config


class _FakeClientError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeEC2:
    def __init__(self, *, run_error: str | None = None, terminate_error: bool = False) -> None:
        self.run_error = run_error
        self.terminate_error = terminate_error
        self.run_kwargs: dict | None = None
        self.terminated: list[list[str]] = []

    def run_instances(self, **kwargs):
        self.run_kwargs = kwargs
        if self.run_error:
            raise _FakeClientError(self.run_error)
        return {"Instances": [{"InstanceId": "i-abc"}]}

    def terminate_instances(self, InstanceIds):  # noqa: N803
        self.terminated.append(InstanceIds)
        if self.terminate_error:
            raise RuntimeError("boom")


class FakeSSM:
    def get_parameter(self, Name):  # noqa: N803
        return {"Parameter": {"Value": "ami-123"}}


@pytest.fixture
def make_probe(monkeypatch):
    # Patch the ClientError the probe catches so the fake error is classified like botocore's.
    monkeypatch.setattr(probe_mod, "ClientError", _FakeClientError)
    monkeypatch.setattr(probe_mod, "HAS_BOTO", True)

    def _make(ec2: FakeEC2) -> SpotLaunchProbe:
        p = SpotLaunchProbe(Config(), cache_ttl_seconds=60)
        p._probe_client = lambda service, region: ec2 if service == "ec2" else FakeSSM()  # type: ignore[method-assign]
        return p

    return _make


class TestSpotLaunchProbe:
    def test_success_means_available_and_terminates(self, make_probe) -> None:
        ec2 = FakeEC2()
        result = make_probe(ec2).probe("us-west-2", "A100")
        assert result.available is True
        assert result.instance_id == "i-abc" and result.terminate_error is None
        assert ec2.terminated == [["i-abc"]]
        kwargs = ec2.run_kwargs
        assert kwargs["InstanceType"] == "p4d.24xlarge" and kwargs["ImageId"] == "ami-123"
        assert kwargs["InstanceMarketOptions"]["MarketType"] == "spot"
        assert kwargs["InstanceMarketOptions"]["SpotOptions"]["SpotInstanceType"] == "one-time"
        assert kwargs["TagSpecifications"][0]["Tags"][0]["Key"] == PROBE_TAG_KEY

    def test_capacity_error_means_unavailable(self, make_probe) -> None:
        ec2 = FakeEC2(run_error="InsufficientInstanceCapacity")
        result = make_probe(ec2).probe("us-west-2", "A100")
        assert result.available is False
        assert ec2.terminated == []

    @pytest.mark.parametrize(
        "code", ["VcpuLimitExceeded", "UnauthorizedOperation", "RequestLimitExceeded", "MaxSpotInstanceCountExceeded"]
    )
    def test_other_errors_are_inconclusive(self, make_probe, code) -> None:
        result = make_probe(FakeEC2(run_error=code)).probe("us-west-2", "A100")
        assert result.available is None
        assert code in result.detail

    def test_terminate_failure_is_reported(self, make_probe, caplog) -> None:
        ec2 = FakeEC2(terminate_error=True)
        result = make_probe(ec2).probe("us-west-2", "A100")
        assert result.available is True
        assert result.terminate_error == "boom"
        assert "i-abc" in caplog.text

    def test_terminate_runs_in_finally_even_if_result_building_fails(self, make_probe, monkeypatch) -> None:
        ec2 = FakeEC2()

        def explode(*args, **kwargs):
            raise KeyboardInterrupt

        p = make_probe(ec2)
        monkeypatch.setattr(probe_mod, "ProbeResult", explode)
        with pytest.raises(KeyboardInterrupt):
            p.probe("us-west-2", "A100")
        assert ec2.terminated == [["i-abc"]]

    def test_unknown_gpu_is_inconclusive(self, make_probe) -> None:
        result = make_probe(FakeEC2()).probe("us-west-2", "MI300")
        assert result.available is None


class TestSpotPlacementScoreRequest:
    def test_shape(self) -> None:
        assert spot_placement_score_request("A100") == {
            "InstanceTypes": ["p4d.24xlarge"],
            "TargetCapacity": 1,
            "TargetCapacityUnitType": "units",
        }

    def test_regions_are_optional_filter(self) -> None:
        req = spot_placement_score_request("H100", ["us-east-1", "us-west-2"])
        assert req["RegionNames"] == ["us-east-1", "us-west-2"]

    def test_unknown_gpu_is_none(self) -> None:
        assert spot_placement_score_request("MI300") is None
