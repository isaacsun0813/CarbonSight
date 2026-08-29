"""Unit tests for mapping registry validate (drift checks)."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from carbonsight_cli.commands.mappings import mappings_group
from carbonsight_core.mapping.registry import CloudRegionEntry, CloudSite, Registry
from carbonsight_core.mapping.validate import (
    drift_report_to_dict,
    validate_registry_mappings,
)
from carbonsight_core.watttime import WattTimeError
from typer.testing import CliRunner


def _entry_with_sites(
    region_code: str,
    sites: list[CloudSite],
    wt_regions: list[tuple[str, float]] | None = None,
) -> CloudRegionEntry:
    return CloudRegionEntry(
        provider="aws",
        region_code=region_code,
        display_name=region_code,
        sites=sites,
        wt_regions=wt_regions or [],
    )


def _single_site_entry(region_code: str, site_id: str, lat: float, lon: float) -> CloudRegionEntry:
    return _entry_with_sites(
        region_code,
        [
            CloudSite(
                site_id=site_id,
                provider="aws",
                region_code=region_code,
                lat=lat,
                lon=lon,
            ),
        ],
    )


class TestValidateRegistryMappings:
    def test_no_drift_single_site(self) -> None:
        reg = Registry()
        entry = _single_site_entry("us-east-1", "ashburn-1", 38.96, -77.56)
        entry.wt_regions = [("PJM_DC", 1.0)]
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.return_value = {"region": "PJM_DC"}

        result = validate_registry_mappings(reg, wt)
        assert not result.has_drift
        assert result.regions_ok == 1
        assert result.regions_drift == 0
        assert result.regions[0].drift is False

    def test_two_sites_same_code_no_drift(self) -> None:
        reg = Registry()
        entry = _entry_with_sites(
            "us-east-1",
            [
                CloudSite(site_id="a", provider="aws", region_code="us-east-1", lat=1.0, lon=2.0),
                CloudSite(site_id="b", provider="aws", region_code="us-east-1", lat=3.0, lon=4.0),
            ],
            wt_regions=[("PJM_DC", 1.0)],
        )
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.return_value = {"region": "PJM_DC"}

        result = validate_registry_mappings(reg, wt)
        assert not result.has_drift
        assert result.regions[0].live_wt_regions == [("PJM_DC", 1.0)]

    def test_two_sites_mixture_no_drift(self) -> None:
        reg = Registry()
        entry = _entry_with_sites(
            "us-east-1",
            [
                CloudSite(site_id="a", provider="aws", region_code="us-east-1", lat=1.0, lon=2.0),
                CloudSite(site_id="b", provider="aws", region_code="us-east-1", lat=3.0, lon=4.0),
            ],
            wt_regions=[("PJM_DC", 0.5), ("PJM_EASTERN_OH", 0.5)],
        )
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.side_effect = [
            {"region": "PJM_DC"},
            {"region": "PJM_EASTERN_OH"},
        ]

        result = validate_registry_mappings(reg, wt)
        assert not result.has_drift

    def test_drift_detected(self) -> None:
        reg = Registry()
        entry = _single_site_entry("us-east-1", "s1", 1.0, 2.0)
        entry.wt_regions = [("OLD", 1.0)]
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.return_value = {"region": "PJM_DC"}

        result = validate_registry_mappings(reg, wt)
        assert result.has_drift
        assert result.regions_drift == 1
        assert result.regions[0].stored_wt_regions == [("OLD", 1.0)]
        assert result.regions[0].live_wt_regions == [("PJM_DC", 1.0)]

    def test_watttime_error_no_drift_if_all_fail(self) -> None:
        reg = Registry()
        entry = _single_site_entry("us-east-1", "s1", 1.0, 2.0)
        entry.wt_regions = [("OLD", 1.0)]
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.side_effect = WattTimeError("rate limited")

        result = validate_registry_mappings(reg, wt)
        assert not result.has_drift
        assert not result.regions
        assert len(result.warnings) >= 2

    def test_partial_site_success_drift(self) -> None:
        reg = Registry()
        entry = _entry_with_sites(
            "us-east-1",
            [
                CloudSite(site_id="a", provider="aws", region_code="us-east-1", lat=1.0, lon=2.0),
                CloudSite(site_id="b", provider="aws", region_code="us-east-1", lat=3.0, lon=4.0),
            ],
            wt_regions=[("PJM_DC", 0.5), ("PJM_EASTERN_OH", 0.5)],
        )
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.side_effect = [
            {"region": "PJM_DC"},
            WattTimeError("fail"),
        ]

        result = validate_registry_mappings(reg, wt)
        assert result.has_drift
        assert result.regions[0].live_wt_regions == [("PJM_DC", 1.0)]


class TestDriftReportToDict:
    def test_json_shape(self) -> None:
        reg = Registry()
        entry = _single_site_entry("us-east-1", "s1", 1.0, 2.0)
        entry.wt_regions = [("PJM_DC", 1.0)]
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.return_value = {"region": "PJM_DC"}

        result = validate_registry_mappings(reg, wt)
        payload = drift_report_to_dict(result)
        assert payload["status"] == "ok"
        assert payload["regions_ok"] == 1
        assert payload["regions_drift"] == 0
        assert "warnings" in payload
        region = payload["regions"][0]
        assert region["provider"] == "aws"
        assert region["region_code"] == "us-east-1"
        assert region["stored_wt_regions"] == [{"wt_region": "PJM_DC", "weight": 1.0}]
        assert region["live_wt_regions"] == [{"wt_region": "PJM_DC", "weight": 1.0}]
        assert region["drift"] is False
        assert "mapping_confidence" in region
        assert "confidence_label" in region
        assert region["sites"][0]["site_id"] == "s1"
        assert region["sites"][0]["live_wt_region"] == "PJM_DC"


class TestValidateCli:
    @patch("carbonsight_cli.commands.mappings.WattTimeClient")
    def test_json_ok(self, mock_client_class: MagicMock) -> None:
        runner = CliRunner()
        mock_wt = MagicMock()
        mock_wt.region_from_loc.return_value = {"region": "PJM_DC"}
        mock_client_class.return_value = mock_wt

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            payload = {
                "regions": [
                    {
                        "provider": "aws",
                        "region_code": "us-east-1",
                        "display_name": "US East",
                        "sites": [
                            {
                                "site_id": "s1",
                                "provider": "aws",
                                "region_code": "us-east-1",
                                "lat": 38.96,
                                "lon": -77.56,
                            },
                        ],
                        "wt_regions": [{"wt_region": "PJM_DC", "weight": 1.0}],
                    },
                ],
            }
            path.write_text(json.dumps(payload))

            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("WATTTIME_USERNAME", "u")
                mp.setenv("WATTTIME_PASSWORD", "p")
                result = runner.invoke(
                    mappings_group,
                    ["validate", "--registry", str(path), "--json"],
                )
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert data["status"] == "ok"

    @patch("carbonsight_cli.commands.mappings.WattTimeClient")
    def test_drift_exits_one(self, mock_client_class: MagicMock) -> None:
        runner = CliRunner()
        mock_wt = MagicMock()
        mock_wt.region_from_loc.return_value = {"region": "PJM_DC"}
        mock_client_class.return_value = mock_wt

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            payload = {
                "regions": [
                    {
                        "provider": "aws",
                        "region_code": "us-east-1",
                        "display_name": "US East",
                        "sites": [
                            {
                                "site_id": "s1",
                                "provider": "aws",
                                "region_code": "us-east-1",
                                "lat": 38.96,
                                "lon": -77.56,
                            },
                        ],
                        "wt_regions": [{"wt_region": "OLD", "weight": 1.0}],
                    },
                ],
            }
            path.write_text(json.dumps(payload))

            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("WATTTIME_USERNAME", "u")
                mp.setenv("WATTTIME_PASSWORD", "p")
                result = runner.invoke(
                    mappings_group,
                    ["validate", "--registry", str(path), "--json"],
                )
                assert result.exit_code == 1
                data = json.loads(result.output)
                assert data["status"] == "drift"

    def test_no_creds_skipped(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            payload = {
                "regions": [
                    {
                        "provider": "aws",
                        "region_code": "us-east-1",
                        "display_name": "US East",
                        "sites": [],
                        "wt_regions": [],
                    },
                ],
            }
            path.write_text(json.dumps(payload))

            with pytest.MonkeyPatch.context() as mp:
                mp.delenv("WATTTIME_USERNAME", raising=False)
                mp.delenv("WATTTIME_PASSWORD", raising=False)
                result = runner.invoke(
                    mappings_group,
                    ["validate", "--registry", str(path), "--json"],
                )
                assert result.exit_code == 0
                data = json.loads(result.output)
                assert data["status"] == "skipped"
