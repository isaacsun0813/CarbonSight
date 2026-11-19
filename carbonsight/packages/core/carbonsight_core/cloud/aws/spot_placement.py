from collections.abc import Iterable

from carbonsight_core.cloud.aws.base import HAS_BOTO, BaseAWSProvider

MAX_REGIONS_PER_REQUEST = 10


def get_spot_placement_scores(
    instance_type: str,
    num_nodes: int,
    regions: Iterable[str],
) -> dict[str, int]:
    """Return available regional scores for the exact request shape, omitting failures."""
    if not HAS_BOTO or not instance_type or type(num_nodes) is not int or num_nodes < 1:
        return {}

    from botocore.exceptions import BotoCoreError, ClientError

    candidates = list(dict.fromkeys(regions))
    if not candidates:
        return {}

    try:
        ec2 = BaseAWSProvider()._client("ec2", candidates[0])
        paginator = ec2.get_paginator("get_spot_placement_scores")
    except (BotoCoreError, ClientError, OSError):
        return {}

    scores: dict[str, int] = {}
    for start in range(0, len(candidates), MAX_REGIONS_PER_REQUEST):
        batch = candidates[start : start + MAX_REGIONS_PER_REQUEST]
        try:
            pages = paginator.paginate(
                InstanceTypes=[instance_type],
                TargetCapacity=num_nodes,
                TargetCapacityUnitType="units",
                SingleAvailabilityZone=False,
                RegionNames=batch,
            )
            for page in pages:
                for entry in page.get("SpotPlacementScores", []):
                    region, score = entry.get("Region"), entry.get("Score")
                    if region in batch and not entry.get("AvailabilityZoneId"):
                        if type(score) is int and 1 <= score <= 10:
                            scores[region] = score
        except (BotoCoreError, ClientError, OSError):
            continue

    return scores
