"""FastAPI app: recommendations, carbon forecast, runs, regions, mappings."""

import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from carbonsight_api.routes import carbon, mappings, recommendations, regions, runs


def _registry_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry_path = _registry_path()
    # Warm central MOER cache (synthetic or WattTime) so first client is fast
    try:
        from carbonsight_api.routes.carbon import warm_all_regions

        app.state.cache_warmed = warm_all_regions()
    except Exception:
        app.state.cache_warmed = 0
    yield


app = FastAPI(title="CarbonSight API", version="0.2.0", lifespan=lifespan)

app.include_router(recommendations.router, prefix="/v1", tags=["recommendations"])
app.include_router(carbon.router, prefix="/v1", tags=["carbon"])
app.include_router(runs.router, prefix="/v1", tags=["runs"])
app.include_router(regions.router, prefix="/v1", tags=["regions"])
app.include_router(mappings.router, prefix="/v1", tags=["mappings"])


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so an error body can always be serialised."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 with a serialisable body.

    FastAPI's default handler echoes the offending input back, so a request
    carrying ``1e400`` produced a validation error whose *own* body contained
    ``inf`` and blew up in ``json.dumps`` — a plain-text 500, worse than the
    Infinity it was meant to reject.
    """
    return JSONResponse(status_code=422, content={"detail": _json_safe(exc.errors())})


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
