"""WattTime client unit checks (no live HTTP)."""

import pytest

from carbonsight_core.watttime import SUPPORTED_MOER_UNIT, WattTimeClient, WattTimeError


class TestAssertMoerUnits:
    def test_accepts_top_level_units(self) -> None:
        WattTimeClient()._assert_moer_units({"units": SUPPORTED_MOER_UNIT})

    def test_accepts_v3_meta_units(self) -> None:
        WattTimeClient()._assert_moer_units(
            {
                "meta": {"units": SUPPORTED_MOER_UNIT, "region": "PJM_DC"},
                "data": [{"point_time": "2026-01-01T00:00:00+00:00", "value": 400.0}],
            }
        )

    def test_accepts_units_on_first_data_point(self) -> None:
        WattTimeClient()._assert_moer_units(
            {
                "data": [
                    {"point_time": "2026-01-01T00:00:00+00:00", "value": 400.0, "units": SUPPORTED_MOER_UNIT},
                ],
            }
        )

    def test_rejects_missing_units(self) -> None:
        with pytest.raises(WattTimeError, match="Missing 'units'"):
            WattTimeClient()._assert_moer_units({"data": [{"value": 1.0}]})

    def test_rejects_unsupported_units(self) -> None:
        with pytest.raises(WattTimeError, match="Unsupported unit"):
            WattTimeClient()._assert_moer_units({"meta": {"units": "gco2_per_kwh"}})
