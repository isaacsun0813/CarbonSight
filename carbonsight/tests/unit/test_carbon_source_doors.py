"""The CLI must never show a traceback for a misconfigured carbon source.

`get_carbon_provider` raises `CarbonProviderError` when a *configured* source is
unreachable — deliberately, so a broken source can't be papered over with
fabricated numbers. That only helps if the doors turn it into a sentence.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import typer
from carbonsight_cli.commands.advise import run_advise
from typer.testing import CliRunner

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "train_minimal.yaml"


@pytest.fixture(autouse=True)
def _no_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("WATTTIME_USERNAME", "WATTTIME_PASSWORD", "CARBONSIGHT_API_URL"):
        monkeypatch.delenv(var, raising=False)


def test_no_source_configured_names_both_options(capsys: pytest.CaptureFixture[str]) -> None:
    run_advise(FIXTURE)
    out = capsys.readouterr().out
    assert "No carbon data source" in out
    assert "WATTTIME_USERNAME" in out and "CARBONSIGHT_API_URL" in out


def test_no_source_configured_still_emits_valid_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scripts parse this; the empty case must stay machine-readable."""
    run_advise(FIXTURE, json_out=True)
    assert capsys.readouterr().out.strip() == "[]"


def test_unreachable_api_exits_cleanly_not_with_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CARBONSIGHT_API_URL", "http://localhost:9999")

    def refused(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", refused)

    with pytest.raises(typer.Exit) as exc:
        run_advise(FIXTURE)
    assert exc.value.exit_code == 1

    err = capsys.readouterr().err
    assert "unreachable" in err
    assert "Traceback" not in err


def test_the_error_says_which_url_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Naming the URL is the difference between a useful error and a shrug."""
    monkeypatch.setenv("CARBONSIGHT_API_URL", "http://not-a-real-host:1234")

    def refused(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", refused)

    with pytest.raises(typer.Exit):
        run_advise(FIXTURE)
    assert "http://not-a-real-host:1234" in capsys.readouterr().err


def test_cli_entrypoint_exits_one_on_a_dead_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end through Typer, so the exit code a shell sees is pinned too."""
    from carbonsight_cli.main import app

    monkeypatch.setenv("CARBONSIGHT_API_URL", "http://localhost:9999")

    def refused(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", refused)

    result = CliRunner().invoke(app, ["advise", "--yaml", str(FIXTURE)])
    assert result.exit_code == 1
    assert "unreachable" in result.output or "unreachable" in (result.stderr or "")
