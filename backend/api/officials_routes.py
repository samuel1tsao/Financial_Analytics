from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import models
from database import get_db
from auth import get_current_user
from officials_data import get_all_officials, get_official_by_id

router = APIRouter(prefix="/api/v1", tags=["officials"])


@router.get("/officials")
def list_officials():
    """Return all tracked public officials and their portfolio data."""
    officials = get_all_officials()
    return {"officials": officials}


@router.get("/officials/{official_id}")
def get_official(official_id: str):
    """Return a single official's full profile."""
    official = get_official_by_id(official_id)
    if not official:
        raise HTTPException(status_code=404, detail="Official not found")
    return official


@router.post("/officials/{official_id}/mimic")
def mimic_official(
    official_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Copy an official's portfolio as a new saved profile for the user.
    Creates a portfolio named 'Mimic: {official_name}'.
    """
    official = get_official_by_id(official_id)
    if not official:
        raise HTTPException(status_code=404, detail="Official not found")

    portfolio = models.Portfolio(
        user_id=current_user.id,
        profile_name=f"Mimic: {official['name']}",
        is_current=False,
        weight_json=json.dumps(official["portfolio"]),
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return {
        "id": portfolio.id,
        "profile_name": portfolio.profile_name,
        "weights": official["portfolio"],
    }
