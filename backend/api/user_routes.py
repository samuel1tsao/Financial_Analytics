from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import models
import schemas
from database import get_db
from auth import get_current_user

router = APIRouter(prefix="/api/v1", tags=["user"])


@router.get("/user/data")
def get_user_data(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the full user profile: info, portfolios, and favorites."""
    portfolios = (
        db.query(models.Portfolio)
        .filter(models.Portfolio.user_id == current_user.id)
        .all()
    )
    favorites = (
        db.query(models.Favorite)
        .filter(models.Favorite.user_id == current_user.id)
        .all()
    )

    return {
        "user": {
            "id": current_user.id,
            "email": current_user.email,
            "created_at": str(current_user.created_at),
        },
        "portfolios": [
            {
                "id": p.id,
                "profile_name": p.profile_name,
                "is_current": p.is_current,
                "weights": json.loads(p.weight_json),
                "created_at": str(p.created_at),
            }
            for p in portfolios
        ],
        "favorites": [f.official_id for f in favorites],
    }


@router.post("/user/favorites")
def toggle_favorite(
    official_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add or remove a public official from the user's favorites."""
    existing = (
        db.query(models.Favorite)
        .filter(
            models.Favorite.user_id == current_user.id,
            models.Favorite.official_id == official_id,
        )
        .first()
    )

    if existing:
        db.delete(existing)
        db.commit()
        return {"action": "removed", "official_id": official_id}
    else:
        fav = models.Favorite(user_id=current_user.id, official_id=official_id)
        db.add(fav)
        db.commit()
        return {"action": "added", "official_id": official_id}
