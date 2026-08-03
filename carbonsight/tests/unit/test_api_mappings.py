"""Unit tests for API mappings revalidate endpoint."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from carbonsight_api.main import app
from fastapi.testclient import TestClient


def _write_minimal_registry(path: Path, wt_region: str = "PJM_DC") -> None:
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
                "wt_regions": [{"wt_region": wt_region, "weight": 1.0}],
            },
        ],
    }
    path.write_text(json.dumps(payload))


class TestApiMappingsRevalidate:
    def test_missing_creds_returns_503(self) -> None:
        client = TestClient(app)
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("WATTTIME_USERNAME", raising=False)
            mp.delenv("WATTTIME_PASSWORD", raising=False)
            response = client.post("/v1/mappings/revalidate")
            assert response.status_code == 503

    @patch("carbonsight_api.routes.mappings.WattTimeClient")
    def test_revalidate_returns_validate_json(self, mock_client_class: MagicMock) -> None:
        mock_wt = MagicMock()
        mock_wt.region_from_loc.return_value = {"region": "PJM_DC"}
        mock_client_class.return_value = mock_wt

        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "registry.json"
            _write_minimal_registry(reg_path)

            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("WATTTIME_USERNAME", "u")
                mp.setenv("WATTTIME_PASSWORD", "p")
                app.state.registry_path = reg_path
                client = TestClient(app)
                response = client.post("/v1/mappings/revalidate")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
                assert data["regions_ok"] == 1
                assert "triggered" not in data.get("message", "")

    @patch("carbonsight_api.routes.mappings.WattTimeClient")
    def test_revalidate_reports_drift(self, mock_client_class: MagicMock) -> None:
        mock_wt = MagicMock()
        mock_wt.region_from_loc.return_value = {"region": "PJM_DC"}
        mock_client_class.return_value = mock_wt

        with tempfile.TemporaryDirectory() as tmp:
            reg_path = Path(tmp) / "registry.json"
            _write_minimal_registry(reg_path, wt_region="OLD")

            with pytest.MonkeyPatch.context() as mp:
                mp.setenv("WATTTIME_USERNAME", "u")
                mp.setenv("WATTTIME_PASSWORD", "p")
                app.state.registry_path = reg_path
                client = TestClient(app)
                response = client.post("/v1/mappings/revalidate")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "drift"
                assert data["regions_drift"] == 1
