"""Launch-then-terminate Spot probe (the SkyNomad paper's availability probe).

WARNING: a successful probe starts a real EC2 Spot instance and bills for it (at least
the minimum billing period) until it is terminated. For GPU families such as p4d/p5 that is
not cheap, and it needs vCPU quota. Probing is opt-in at the CLI.

Success means Spot capacity was available *at that moment* in a (default-VPC) subnet picked by
AWS. It says nothing about how long a real instance would survive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider, ClientError
from carbonsight_core.cloud.aws.gpu_catalog import aws_instance_type_for_gpu
from carbonsight_core.config import Config

logger = logging.getLogger(__name__)

PROBE_TAG_KEY = "carbonsight-probe"

# Public SSM parameter for the latest Amazon Linux 2023 x86_64 AMI (needs ssm:GetParameter).
_AMI_SSM_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"

# Error codes that mean "no Spot capacity right now". Everything else is inconclusive.
# NOTE: unverified against a live account; extend as real responses are observed.
CAPACITY_ERROR_CODES = frozenset({"InsufficientInstanceCapacity"})


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Outcome of one probe. ``available`` is None when the probe was inconclusive."""

    region: str
    instance_type: str
    available: bool | None
    detail: str = ""
    instance_id: str | None = None
    terminate_error: str | None = None


class SpotLaunchProbe(BaseAWSProvider):
    """Probe Spot capacity by launching one instance and terminating it immediately."""

    def __init__(self, config: Config | None = None, *, cache_ttl_seconds: int | None = None) -> None:
        super().__init__(config, cache_ttl_seconds=cache_ttl_seconds)

    def _probe_client(self, service_name: str, region_name: str) -> Any:
        """Client with short timeouts and a single retry so a probe never stalls the loop."""
        key = f"probe:{service_name}:{region_name}"
        if key not in self._clients:
            from botocore.config import Config as BotoConfig

            self._clients[key] = self._get_session().client(
                service_name,
                region_name=region_name,
                config=BotoConfig(
                    connect_timeout=5,
                    read_timeout=30,
                    retries={"max_attempts": 1, "mode": "standard"},
                ),
            )
        return self._clients[key]

    def _ami_id(self, region: str) -> str:
        cache_key = ("probe-ami", region)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        ssm = self._probe_client("ssm", region)
        value = ssm.get_parameter(Name=_AMI_SSM_PARAMETER)["Parameter"]["Value"]
        return self._set_cached(cache_key, value)

    def probe(self, region: str, gpu_type: str) -> ProbeResult:
        instance_type = aws_instance_type_for_gpu(gpu_type)
        if instance_type is None:
            return ProbeResult(region, "", None, f"no EC2 instance type known for {gpu_type!r}")
        if not HAS_BOTO:
            return ProbeResult(region, instance_type, None, "boto3 not installed")

        try:
            ami = self._ami_id(region)
            ec2 = self._probe_client("ec2", region)
        except Exception as exc:  # noqa: BLE001 - inconclusive, never fatal
            return ProbeResult(region, instance_type, None, f"setup failed: {exc}")

        instance_id: str | None = None
        outcome: ProbeResult
        try:
            response = ec2.run_instances(
                ImageId=ami,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
                InstanceInitiatedShutdownBehavior="terminate",
                TagSpecifications=[
                    {
                        "ResourceType": "instance",
                        "Tags": [{"Key": PROBE_TAG_KEY, "Value": "1"}],
                    }
                ],
            )
            instances = response.get("Instances", [])
            if instances:
                instance_id = instances[0].get("InstanceId")
            outcome = ProbeResult(
                region,
                instance_type,
                True,
                "spot instance launched, terminated immediately",
                instance_id=instance_id,
            )
        except ClientError as exc:
            code = _error_code(exc)
            if code in CAPACITY_ERROR_CODES:
                outcome = ProbeResult(region, instance_type, False, f"no spot capacity ({code})")
            else:
                # Quota, auth, throttling, missing default VPC, ...: not evidence about capacity.
                outcome = ProbeResult(region, instance_type, None, f"inconclusive ({code or exc})")
        except Exception as exc:  # noqa: BLE001 - timeouts and the like are inconclusive
            outcome = ProbeResult(region, instance_type, None, f"inconclusive ({exc})")
        finally:
            # Always terminate whatever was launched, even if the code above raised.
            terminate_error = (
                self._terminate(region, ec2, instance_id) if instance_id is not None else None
            )
        if terminate_error is not None:
            outcome = replace(outcome, terminate_error=terminate_error)
        return outcome

    def _terminate(self, region: str, ec2: Any, instance_id: str) -> str | None:
        try:
            ec2.terminate_instances(InstanceIds=[instance_id])
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "FAILED to terminate probe instance %s in %s (%s). It may still be running and "
                "billing: terminate it manually (tag %s).",
                instance_id,
                region,
                exc,
                PROBE_TAG_KEY,
            )
            return str(exc)


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return str(response.get("Error", {}).get("Code", ""))
