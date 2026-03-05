"""FastAPI app: recommendations, runs, regions, mappings revalidate."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from carbonsight_api.routes import mappings, recommendations, regions, runs

# Registry path for API (same layout as CLI)
def _registry_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "packages" / "core" / "carbonsight_core" / "mapping" / "seed_registry.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry_path = _registry_path()
    yield


app = FastAPI(title="CarbonSight API", version="0.1.0", lifespan=lifespan)

app.include_router(recommendations.router, prefix="/v1", tags=["recommendations"])
app.include_router(runs.router, prefix="/v1", tags=["runs"])
app.include_router(regions.router, prefix="/v1", tags=["regions"])
app.include_router(mappings.router, prefix="/v1", tags=["mappings"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
