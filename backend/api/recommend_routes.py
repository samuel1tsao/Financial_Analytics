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
    Generate a multi-horizon portfolio recommendation using the RL Transformer.
    """
    q = (
        db.query(models.Questionnaire)
        .filter(models.Questionnaire.user_id == current_user.id)
        .order_by(models.Questionnaire.created_at.desc())
        .first()
    )
    if not q:
        raise HTTPException(status_code=400, detail="No questionnaire found. Complete the questionnaire first.")

    answers = json.loads(q.raw_json)
    if "answers" in answers:
        answers = answers["answers"]

    # 2. Use multi-horizon RL encoder
    from vector_encoder import encode_multi_horizon
    result = encode_multi_horizon(answers)
    segments = result["segments"]

    # 3. Save as current portfolio
    # We store the entire segments list in weight_json
    db.query(models.Portfolio).filter(
        models.Portfolio.user_id == current_user.id,
        models.Portfolio.is_current == True,
    ).update({"is_current": False})

    portfolio = models.Portfolio(
        user_id=current_user.id,
        profile_name="RL Multi-Horizon Portfolio",
        is_current=True,
        weight_json=json.dumps(segments),
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return {
        "portfolio": {
            "id": portfolio.id,
            "profile_name": portfolio.profile_name,
            "is_current": True,
            "segments": segments,
        },
        "analysis": result
    }


@router.post("/simulate")
def run_simulation(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
    portfolio_id: int | None = None,
    initial_investment: float = 100000,
    projection_years: int = 30,
):
    """
    Run multi-horizon simulation including Balance-At-Step tracking.
    """
    if portfolio_id:
        portfolio = db.query(models.Portfolio).filter(
            models.Portfolio.id == portfolio_id,
            models.Portfolio.user_id == current_user.id,
        ).first()
    else:
        portfolio = db.query(models.Portfolio).filter(
            models.Portfolio.user_id == current_user.id,
            models.Portfolio.is_current == True,
        ).first()

    if not portfolio:
        raise HTTPException(status_code=404, detail="No portfolio found.")

    segments = json.loads(portfolio.weight_json)
    
    # Extract goals from latest questionnaire
    q = db.query(models.Questionnaire).filter(
        models.Questionnaire.user_id == current_user.id
    ).order_by(models.Questionnaire.created_at.desc()).first()
    
    goals = []
    monthly_contrib = 500
    if q:
        ans = json.loads(q.raw_json)
        if "answers" in ans: ans = ans["answers"]
        goals = ans.get("goals", [])
        monthly_contrib = ans.get("monthly_contrib", 500)

    from vector_encoder import simulate_multi_horizon_portfolio
    sim = simulate_multi_horizon_portfolio(
        segments=segments,
        goals=goals,
        initial_investment=initial_investment,
        monthly_contrib=monthly_contrib
    )

    return {
        "portfolio_id": portfolio.id,
        "portfolio_name": portfolio.profile_name,
        "simulation": sim,
    }
