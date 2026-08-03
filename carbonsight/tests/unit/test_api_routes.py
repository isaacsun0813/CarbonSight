"""API contract: real regions only, and the policy decision reaching the client."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from carbonsight_api.main import app

EXPECTED_REGION_COUNT = 21
TRUNCATED_BY_A = {"ap-east-1", "sa-east-1", "me-south-1", "af-south-1"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _recommend(client, **body):
    r = client.post("/v1/recommendations", json={"duration_hours": 1.0, **body})
    assert r.status_code == 200
    return r.json()


# --- /v1/regions ------------------------------------------------------------


def test_regions_returns_the_registry_and_nothing_invented(client):
    rows = client.get("/v1/regions").json()
    codes = {r["region_code"] for r in rows}
    assert len(rows) == EXPECTED_REGION_COUNT
    assert TRUNCATED_BY_A <= codes
    assert not any(c.startswith("synthetic-") for c in codes)


def test_regions_rows_carry_a_mixture_and_a_confidence(client):
    for row in client.get("/v1/regions").json():
        assert row["wt_regions"] and all("wt_region" in m for m in row["wt_regions"])
        assert 0.0 <= row["confidence"] <= 1.0
        assert row["label"]


# --- /v1/recommendations ----------------------------------------------------


def test_recommendations_returns_real_regions_greenest_first(client):
    body = _recommend(client)
    regions = body["regions"]
    codes = {r["cloud_region"] for r in regions}
    assert len(regions) == EXPECTED_REGION_COUNT
    assert TRUNCATED_BY_A <= codes
    assert not any(c.startswith("synthetic-") for c in codes)
    co2 = [r["expected_co2_kg_mean"] for r in regions]
    assert co2 == sorted(co2)
    assert not any(r["expected_co2_kg_mean"] in (1.0, 2.0, 3.0) for r in regions[:3])


def test_recommendations_cost_is_over_the_requested_duration(client):
    one = {r["cloud_region"]: r for r in _recommend(client, duration_hours=1.0)["regions"]}
    two = {r["cloud_region"]: r for r in _recommend(client, duration_hours=2.0)["regions"]}
    for code, row in two.items():
        assert row["expected_cost_usd"] == pytest.approx(one[code]["expected_cost_usd"] * 2)


def test_recommendations_surfaces_the_rank_decision(client):
    body = _recommend(client, deadline_hours=45.0)
    assert body["action"] == "rank"
    assert body["decision"]["rule"] == "rank"
    assert body["decision"]["region"] in {r["cloud_region"] for r in body["regions"]}
    assert body["value_v"] > 0
    assert body["deadline_passed"] is False


def test_recommendations_surfaces_the_safety_net(client):
    """A job out of slack must be told to go on-demand, not handed 21 spot rows."""
    body = _recommend(client, duration_hours=1.0, deadline_hours=1.0)
    assert body["action"] == "safety_net"
    assert body["decision"]["kind"] == "launch"
    assert body["decision"]["mode"] == "on_demand"
    assert body["decision"]["region"] in {r["cloud_region"] for r in body["regions"]}
    assert body["decision"]["estimated_total_cost_usd"] > 0
    assert "safety_net" in body["decision"]["reason"]


def test_recommendations_surfaces_thrifty(client):
    body = _recommend(client, duration_hours=2.0, progress_hours_done=2.0)
    assert body["action"] == "thrifty"
    assert body["decision"]["kind"] == "idle"


def test_recommendations_past_the_deadline_stays_valid_json(client):
    """V is infinite there; it must not reach the wire as a bare Infinity."""
    raw = client.post(
        "/v1/recommendations",
        json={"duration_hours": 1.0, "deadline_hours": 1.0, "elapsed_hours": 5.0},
    )
    assert raw.status_code == 200
    assert "Infinity" not in raw.text and "NaN" not in raw.text
    body = json.loads(raw.text)
    assert body["deadline_passed"] is True
    assert body["value_v"] is None
    assert all(r["utility_score"] is None for r in body["regions"])
    assert all("U_s=inf" not in n for r in body["regions"] for n in r["notes"])


def test_recommendations_honours_carbon_weight(client):
    """Hardcoding carbon_weight=1.0 in the handler must fail this.

    Asserting only ``light_u != heavy_u`` was vacuous: the response is anchored
    on the wall clock, so two *identical* requests already differ. This measures
    the effect instead — under a heavy weight the dirtiest region must lose far
    more utility than the cleanest, by an amount set by their kg/hr gap.
    """
    light = _recommend(client, carbon_weight=1.0)
    heavy = _recommend(client, carbon_weight=100_000.0)

    light_rows = {r["cloud_region"]: r for r in light["regions"]}
    heavy_u = {r["cloud_region"]: r["utility_score"] for r in heavy["regions"]}
    # regions arrive greenest-first, so these are the extremes of the spread
    cleanest = light["regions"][0]["cloud_region"]
    dirtiest = light["regions"][-1]["cloud_region"]

    def drop(code: str) -> float:
        return light_rows[code]["utility_score"] - heavy_u[code]

    assert drop(dirtiest) > drop(cleanest)
    # The penalty is (kg/hr)/1000 * price * (w_heavy - w_light); the extra weight
    # dwarfs any wall-clock jitter in the underlying MOER.
    kg_gap = (
        light_rows[dirtiest]["expected_co2_kg_mean"]
        - light_rows[cleanest]["expected_co2_kg_mean"]
    )
    expected = kg_gap / 1000.0 * 50.0 * (100_000.0 - 1.0)
    assert drop(dirtiest) - drop(cleanest) == pytest.approx(expected, rel=0.05)


def test_recommendations_honours_deadline_pressure(client):
    on_track = _recommend(client, deadline_hours=45.0)
    behind = _recommend(
        client, deadline_hours=45.0, progress_hours_done=0.1, elapsed_hours=20.0
    )
    assert behind["value_v"] > on_track["value_v"] * 2


def test_get_recommendations_matches_the_post_default(client):
    got = client.get("/v1/recommendations").json()
    assert got["action"] == "rank"
    assert len(got["regions"]) == EXPECTED_REGION_COUNT


# --- /v1/carbon -------------------------------------------------------------


def test_carbon_forecast_reports_its_source(client):
    body = client.get(
        "/v1/carbon/forecast", params={"region": "CAISO_NORTH", "lbar_hours": 3}
    ).json()
    assert body["source"] in {"cache", "db", "live", "synthetic"}
    assert body["data"] and body["data"] == body["points"]
    assert body["moer_avg"] > 0
    assert body["units"] == "lbs_co2_per_mwh"


def test_carbon_regions_lists_grid_codes(client):
    rows = client.get("/v1/carbon/regions").json()
    assert len(rows) == 17
    assert all("wt_region" in r for r in rows)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# --- input validation -------------------------------------------------------


class TestRequestValidation:
    """Bad input must be a 422 naming the field, not a plain-text 500.

    JobSpec was constructed inside the handler, so its constraints surfaced as
    unhandled exceptions; 1e400 was worse still, because FastAPI's default
    handler echoed the offending ``inf`` into an error body that then failed to
    serialise.
    """

    @pytest.mark.parametrize(
        "body",
        [
            {"duration_hours": 0},
            {"duration_hours": -1},
            {"deadline_hours": 0},
            {"deadline_hours": -5},
            {"gpu_count": -3},
            {"gpu_utilization": 5},
            {"gpu_utilization": -0.1},
            {"carbon_weight": -1},
            {"carbon_price_usd_per_ton": -10},
            {"checkpoint_size_gb": -1},
            {"progress_hours_done": -1},
            {"elapsed_hours": -1},
            {"gpu_type": ""},
        ],
    )
    def test_invalid_input_is_422(self, client, body):
        response = client.post("/v1/recommendations", json={"duration_hours": 1.0, **body})
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["detail"]

    @pytest.mark.parametrize("literal", ["1e400", "-1e400"])
    def test_non_finite_input_is_422_with_a_serialisable_body(self, client, literal):
        response = client.post(
            "/v1/recommendations",
            content='{"duration_hours": %s}' % literal,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
        detail = response.json()["detail"]
        assert detail[0]["loc"] == ["body", "duration_hours"]
        assert "Infinity" not in response.text and "NaN" not in response.text

    def test_valid_input_still_works(self, client):
        assert client.post("/v1/recommendations", json={"duration_hours": 1.0}).status_code == 200
