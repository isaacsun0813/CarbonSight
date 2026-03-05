"""POST /v1/mappings/revalidate -> trigger drift job."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/mappings/revalidate")
def post_mappings_revalidate() -> dict:
    """Trigger mapping drift revalidation (async in production)."""
    return {"status": "triggered", "message": "Revalidation job triggered. Run 'carbonsight mappings validate' for details."}
