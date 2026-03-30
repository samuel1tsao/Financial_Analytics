from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import models
import schemas
from database import get_db
from auth import get_current_user
from vector_encoder import encode_questionnaire, simulate_portfolio

router = APIRouter(prefix="/api/v1", tags=["recommendation"])


@router.post("/recommend")
def recommend_portfolio(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Generate a portfolio recommendation from the user's latest questionnaire.
    Encodes answers into a semantic vector, optimises weights, and auto-saves
    as the user's 'Current' portfolio.
    """
    # 1. Fetch latest questionnaire
    q = (
        db.query(models.Questionnaire)
        .filter(models.Questionnaire.user_id == current_user.id)
        .order_by(models.Questionnaire.created_at.desc())
        .first()
    )
    if not q:
        raise HTTPException(status_code=400, detail="No questionnaire found. Complete the questionnaire first.")

    answers = json.loads(q.raw_json)

    # 2. Encode questionnaire into portfolio weights
    result = encode_questionnaire(answers)
    weights = result["weights"]

    # 3. Unset any existing 'current' portfolios
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

    # 4. Save new portfolio as 'Current'
    portfolio = models.Portfolio(
        user_id=current_user.id,
        profile_name="Recommended Portfolio",
        is_current=True,
        weight_json=json.dumps(weights),
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return {
        "portfolio": {
            "id": portfolio.id,
            "profile_name": portfolio.profile_name,
            "is_current": True,
            "weights": weights,
        },
        "analysis": {
            "risk_score": result["risk_score"],
            "fomo_score": result["fomo_score"],
            "goals": result["goals"],
        },
    }


@router.post("/simulate")
def run_simulation(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    portfolio_id: int | None = None,
    initial_investment: float = 100000,
    projection_years: int = 30,
    custom_goals_json: str | None = None,
):
    """
    Run a variance-covariance simulation on the specified (or current) portfolio.
    Returns expected path, confidence bounds (±2σ), and cash-out events.
    """
    # 1. Find the portfolio
    if portfolio_id:
        portfolio = (
            db.query(models.Portfolio)
            .filter(
                models.Portfolio.id == portfolio_id,
                models.Portfolio.user_id == current_user.id,
            )
            .first()
        )
    else:
        portfolio = (
            db.query(models.Portfolio)
            .filter(
                models.Portfolio.user_id == current_user.id,
                models.Portfolio.is_current == True,
            )
            .first()
        )

    if not portfolio:
        raise HTTPException(status_code=404, detail="No portfolio found to simulate.")

    weights = json.loads(portfolio.weight_json)

    # 2. Extract goals for cash-out events
    goals = []
    if custom_goals_json:
        try:
            raw_custom = json.loads(custom_goals_json)
            for g in raw_custom:
                goals.append({
                    "name": g.get("name", "Goal"),
                    "amount": g.get("amount", 0),
                    "years": g.get("years", 10),
                    "is_short_term": g.get("years", 10) <= 5,
                })
        except:
            pass
    else:
        q = (
            db.query(models.Questionnaire)
            .filter(models.Questionnaire.user_id == current_user.id)
            .order_by(models.Questionnaire.created_at.desc())
            .first()
        )
        if q:
            answers = json.loads(q.raw_json)
            raw_goals = answers.get("goals", [])
            for g in raw_goals:
                goals.append({
                    "name": g.get("name", "Goal"),
                    "amount": g.get("amount", 0),
                    "years": g.get("years", 10),
                    "is_short_term": g.get("years", 10) <= 5,
                })

    # 3. Run simulation
    sim = simulate_portfolio(
        weights=weights,
        goals=goals,
        initial_investment=initial_investment,
        years=projection_years,
    )

    return {
        "portfolio_id": portfolio.id,
        "portfolio_name": portfolio.profile_name,
        "simulation": sim,
    }
