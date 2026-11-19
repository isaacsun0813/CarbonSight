from unittest.mock import MagicMock, patch

import pytest
from carbonsight_core.cloud.aws.spot_placement import get_spot_placement_scores


def test_spot_placement_scores_keep_partial_results() -> None:
    exceptions = pytest.importorskip("botocore.exceptions")
    regions = [f"region-{number}" for number in range(11)]
    ec2 = MagicMock()
    paginate = ec2.get_paginator.return_value.paginate
    paginate.side_effect = [
        [
            {
                "SpotPlacementScores": [
                    {"Region": regions[0], "Score": 2},
                    {"Region": regions[1], "Score": 0},
                    {"Region": "unrequested", "Score": 2},
                ]
            },
            {"SpotPlacementScores": [{"Region": regions[2], "Score": 1}]},
        ],
        exceptions.ClientError(
            {"Error": {"Code": "UnauthorizedOperation"}}, "GetSpotPlacementScores"
        ),
    ]
    with patch(
        "carbonsight_core.cloud.aws.spot_placement.BaseAWSProvider._client",
        return_value=ec2,
    ):
        assert get_spot_placement_scores("p4d.24xlarge", 3, regions) == {
            regions[0]: 2,
            regions[2]: 1,
        }

    ec2.get_paginator.assert_called_once_with("get_spot_placement_scores")
    assert paginate.call_args_list[0].kwargs == {
        "InstanceTypes": ["p4d.24xlarge"],
        "TargetCapacity": 3,
        "TargetCapacityUnitType": "units",
        "SingleAvailabilityZone": False,
        "RegionNames": regions[:10],
    }
    assert paginate.call_args_list[1].kwargs["RegionNames"] == regions[10:]


def test_spot_placement_scores_without_aws_sdk() -> None:
    with patch("carbonsight_core.cloud.aws.spot_placement.HAS_BOTO", False):
        assert get_spot_placement_scores("p4d.24xlarge", 1, ["us-east-1"]) == {}
