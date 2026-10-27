"""Spot Advisor band parsing and fetching (no network)."""

import httpx
import pytest
from carbonsight_core.cloud.aws.spot_advisor import (
    SpotAdvisorError,
    fetch_advisor_bands,
    parse_advisor_bands,
)

RANGES = [
    {"index": 0, "label": "<5%", "dots": 0, "max": 5},
    {"index": 1, "label": "5-10%", "dots": 1, "max": 11},
    {"index": 2, "label": "10-15%", "dots": 2, "max": 16},
    {"index": 3, "label": "15-20%", "dots": 3, "max": 22},
    {"index": 4, "label": ">20%", "dots": 4, "max": 100},
]

DATA = {
    "instance_types": {},
    "ranges": RANGES,
    "spot_advisor": {
        "us-west-2": {"Linux": {"p4d.24xlarge": {"s": 70, "r": 1}}, "Windows": {}},
        "us-east-1": {"Linux": {"p4d.24xlarge": {"s": 60, "r": 4}}},
        "eu-west-1": {"Linux": {"g5.xlarge": {"s": 50, "r": 0}}},
    },
}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestParse:
    def test_returns_band_per_listed_region(self) -> None:
        bands = parse_advisor_bands(
            DATA, "p4d.24xlarge", ["us-west-2", "us-east-1", "eu-west-1", "ap-south-1"]
        )
        assert bands == {"us-west-2": 1, "us-east-1": 4}

    def test_unlisted_type_is_empty(self) -> None:
        assert parse_advisor_bands(DATA, "p5.48xlarge", ["us-west-2"]) == {}

    def test_unexpected_format_raises(self) -> None:
        with pytest.raises(SpotAdvisorError):
            parse_advisor_bands({"nope": 1}, "p4d.24xlarge", ["us-west-2"])

    def test_changed_band_labels_raise(self) -> None:
        bad = {**DATA, "ranges": [{"label": "<1%"}] + RANGES[1:]}
        with pytest.raises(SpotAdvisorError):
            parse_advisor_bands(bad, "p4d.24xlarge", ["us-west-2"])


class TestFetch:
    def test_fetch_parses_response(self) -> None:
        client = _client(lambda request: httpx.Response(200, json=DATA))
        assert fetch_advisor_bands("p4d.24xlarge", ["us-west-2"], client=client) == {
            "us-west-2": 1
        }

    def test_http_error_raises_advisor_error(self) -> None:
        client = _client(lambda request: httpx.Response(503))
        with pytest.raises(SpotAdvisorError):
            fetch_advisor_bands("p4d.24xlarge", ["us-west-2"], client=client)

    def test_non_json_raises_advisor_error(self) -> None:
        client = _client(lambda request: httpx.Response(200, text="<html>"))
        with pytest.raises(SpotAdvisorError):
            fetch_advisor_bands("p4d.24xlarge", ["us-west-2"], client=client)

    def test_network_error_raises_advisor_error(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        with pytest.raises(SpotAdvisorError):
            fetch_advisor_bands("p4d.24xlarge", ["us-west-2"], client=_client(boom))
