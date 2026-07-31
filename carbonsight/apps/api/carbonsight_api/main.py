"""FastAPI app: recommendations, carbon forecast, runs, regions, mappings."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
