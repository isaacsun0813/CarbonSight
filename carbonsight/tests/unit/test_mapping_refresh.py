"""Unit tests for mapping registry refresh and JSON serialization."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from carbonsight_cli.commands.mappings import mappings_group
from carbonsight_core.mapping.refresh import (
    build_wt_regions_from_site_codes,
    parse_wt_region_code,
    refresh_registry_mappings,
)
from carbonsight_core.mapping.registry import (
    CloudRegionEntry,
    CloudSite,
    Registry,
    entry_to_json_dict,
    registry_to_json_dict,
    write_registry_json,
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


class TestParseAndMixture:
    def test_parse_wt_region_code_prefers_region(self) -> None:
        assert parse_wt_region_code({"region": "PJM_DC", "abbrev": "X"}) == "PJM_DC"

    def test_parse_wt_region_code_uses_abbrev(self) -> None:
        assert parse_wt_region_code({"abbrev": "SE"}) == "SE"

    def test_parse_wt_region_code_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="missing region"):
            parse_wt_region_code({})

    def test_build_mixture_single_code(self) -> None:
        assert build_wt_regions_from_site_codes(["A", "A"]) == [("A", 1.0)]

    def test_build_mixture_two_codes(self) -> None:
        assert build_wt_regions_from_site_codes(["A", "B"]) == [("A", 0.5), ("B", 0.5)]


class TestRefreshRegistryMappings:
    def test_single_site_updates_wt_regions_and_metadata(self) -> None:
        reg = Registry()
        entry = _single_site_entry("us-east-1", "ashburn-1", 38.96, -77.56)
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.return_value = {"region": "PJM_DC"}

        result = refresh_registry_mappings(reg, wt)
        assert entry.wt_regions == [("PJM_DC", 1.0)]
        assert entry.s_recency == 1.0
        assert entry.sites[0].last_verified_at is not None
        assert result.regions_updated == 1
        assert result.regions_unchanged == 0

    def test_two_sites_same_code_weight_one(self) -> None:
        reg = Registry()
        entry = _entry_with_sites(
            "us-east-1",
            [
                CloudSite(site_id="a", provider="aws", region_code="us-east-1", lat=1.0, lon=2.0),
                CloudSite(site_id="b", provider="aws", region_code="us-east-1", lat=3.0, lon=4.0),
            ],
        )
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.return_value = {"region": "PJM_DC"}

        refresh_registry_mappings(reg, wt)
        assert entry.wt_regions == [("PJM_DC", 1.0)]

    def test_two_sites_different_codes_mixture(self) -> None:
        reg = Registry()
        entry = _entry_with_sites(
            "us-east-1",
            [
                CloudSite(site_id="a", provider="aws", region_code="us-east-1", lat=1.0, lon=2.0),
                CloudSite(site_id="b", provider="aws", region_code="us-east-1", lat=3.0, lon=4.0),
            ],
        )
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.side_effect = [
            {"region": "PJM_DC"},
            {"region": "PJM_EASTERN_OH"},
        ]

        refresh_registry_mappings(reg, wt)
        assert entry.wt_regions == [("PJM_DC", 0.5), ("PJM_EASTERN_OH", 0.5)]

    def test_watttime_error_on_site_warns_and_skips_region_update(self) -> None:
        reg = Registry()
        entry = _single_site_entry("us-east-1", "s1", 1.0, 2.0)
        entry.wt_regions = [("OLD", 1.0)]
        reg._regions[("aws", "us-east-1")] = entry

        wt = MagicMock()
        wt.region_from_loc.side_effect = WattTimeError("rate limited")

        result = refresh_registry_mappings(reg, wt)
        assert entry.wt_regions == [("OLD", 1.0)]
        assert len(result.warnings) >= 1
        assert result.regions_unchanged == 0
        assert result.regions_updated == 0
        assert not result.changes


class TestRegistrySerialization:
    def test_round_trip_preserves_shape(self) -> None:
        reg = Registry()
        entry = _single_site_entry("eu-north-1", "stockholm-1", 59.3, 18.0)
        entry.wt_regions = [("SE", 1.0)]
        reg._regions[("aws", "eu-north-1")] = entry

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            write_registry_json(reg, path)
            loaded = Registry()
            loaded.load_json(path)
            loaded_entry = loaded._regions[("aws", "eu-north-1")]
            assert loaded_entry.wt_regions == [("SE", 1.0)]
            assert loaded_entry.sites[0].site_id == "stockholm-1"

    def test_entry_to_json_dict_wt_regions_format(self) -> None:
        entry = _single_site_entry("us-west-2", "oregon-1", 45.5, -122.6)
        entry.wt_regions = [("PACW", 1.0)]
        data = entry_to_json_dict(entry)
        assert data["wt_regions"] == [{"wt_region": "PACW", "weight": 1.0}]

    def test_registry_to_json_dict_has_regions_key(self) -> None:
        reg = Registry()
        reg._regions[("aws", "us-east-1")] = _single_site_entry("us-east-1", "s1", 1.0, 2.0)
        payload = registry_to_json_dict(reg)
        assert "regions" in payload
        assert len(payload["regions"]) == 1


class TestRefreshCli:
    @patch("carbonsight_cli.commands.mappings.WattTimeClient")
    def test_dry_run_does_not_write_file(self, mock_client_class: MagicMock) -> None:
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
                    ["refresh", "--registry", str(path)],
                )
                assert result.exit_code == 0
                assert "Dry-run" in result.output
                reloaded = json.loads(path.read_text())
                assert reloaded["regions"][0]["wt_regions"][0]["wt_region"] == "OLD"

    @patch("carbonsight_cli.commands.mappings.WattTimeClient")
    def test_write_out_updates_file(self, mock_client_class: MagicMock) -> None:
        runner = CliRunner()
        mock_wt = MagicMock()
        mock_wt.region_from_loc.return_value = {"region": "PJM_DC"}
        mock_client_class.return_value = mock_wt

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "registry.json"
            out = Path(tmp) / "refreshed.json"
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
            src.write_text(json.dumps(payload))

            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("WATTTIME_USERNAME", "u")
                mp.setenv("WATTTIME_PASSWORD", "p")
                result = runner.invoke(
                    mappings_group,
                    ["refresh", "--registry", str(src), "--write", "--out", str(out)],
                )
                assert result.exit_code == 0
                written = json.loads(out.read_text())
                assert written["regions"][0]["wt_regions"][0]["wt_region"] == "PJM_DC"
