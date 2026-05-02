from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import models
import schemas
from database import get_db
from auth import get_current_user

router = APIRouter(prefix="/api/v1", tags=["portfolio"])


@router.post("/profile/save")
def save_portfolio(
    payload: schemas.PortfolioSave,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a generated portfolio as 'current' or a named tracked profile."""
    # If saving as current, unset any existing current
    if payload.is_current:
        existing_current = (
            db.query(models.Portfolio)
            .filter(
                models.Portfolio.user_id == current_user.id,
                models.Portfolio.is_current == True,
            )
            .all()
        )
        for p in existing_current:
            p.is_current = False

    portfolio = models.Portfolio(
        user_id=current_user.id,
        profile_name=payload.profile_name,
        is_current=payload.is_current,
        weight_json=json.dumps(payload.weights),
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return {
        "id": portfolio.id,
        "profile_name": portfolio.profile_name,
        "is_current": portfolio.is_current,
    }


@router.delete("/profile")
def delete_portfolios(
    payload: schemas.PortfolioDeleteRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Bulk delete saved portfolio profiles by ID."""
    deleted_count = (
        db.query(models.Portfolio)
        .filter(
            models.Portfolio.id.in_(payload.ids),
            models.Portfolio.user_id == current_user.id,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    return {"deleted": deleted_count}


@router.delete("/portfolio/{portfolio_id}")
def delete_single_portfolio(
    portfolio_id: int,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a specific portfolio."""
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.user_id == current_user.id
    ).first()

    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found.")

    db.delete(portfolio)
    db.commit()

    return {"message": "Portfolio deleted successfully"}
