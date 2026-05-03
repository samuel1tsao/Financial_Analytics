from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import models
import schemas
from database import get_db
from auth import get_current_user
from vector_encoder import encode_questionnaire, simulate_portfolio
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["recommendation"])

# Simple in-memory cache for benchmarks to avoid redundant heavy MC simulations
_BENCHMARK_CACHE = {}


@router.post("/recommend")
def recommend_portfolio(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Main recommendation endpoint for Questionnaire V1.
    Uses multi-horizon RL strategy.
    """
    # 1. Fetch questionnaire
    q = db.query(models.Questionnaire).filter(
        models.Questionnaire.user_id == current_user.id
    ).order_by(models.Questionnaire.created_at.desc()).first()

    if not q:
        raise HTTPException(status_code=404, detail="Questionnaire not found.")

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
        weight_json=json.dumps(result),
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
    req: schemas.SimulationRequest,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run multi-horizon simulation including Balance-At-Step tracking.
    """
    portfolio_id = req.portfolio_id
    initial_investment = req.initial_investment
    projection_years = req.projection_years
    custom_goals_json = req.custom_goals_json
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

    data = json.loads(portfolio.weight_json)
    segments = data.get("segments", data) if isinstance(data, dict) else data
    
    # No longer tracking goals or cash deposits in backend simulation
    logger.info(f"Running simulation for user {current_user.email}, portfolio_id={portfolio_id}, active_id={portfolio.id}")
    try:
        from vector_encoder import simulate_multi_horizon_portfolio
        sim = simulate_multi_horizon_portfolio(
            segments=segments,
            projection_years=projection_years
        )
        
        if "error" in sim:
            logger.warning(f"Simulation returned error for {current_user.email}: {sim['error']}")

        return {
            "portfolio_id": portfolio.id,
            "portfolio_name": portfolio.profile_name,
            "simulation": sim,
        }
    except Exception as e:
        logger.error(f"Simulation crash for {current_user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
def backtest_portfolio(
    payload: dict, # portfolio_id, initial_investment, monthly_contrib
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Run a historical backtest of the portfolio over the last 10-15 years.
    """
    import numpy as np
    portfolio_id = payload.get("portfolio_id")
    initial_investment = payload.get("initial_investment", 100000)
    monthly_contrib = payload.get("monthly_contrib", 500)

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
        logger.error(f"Backtest failed: Portfolio {portfolio_id} not found for user {current_user.email}")
        raise HTTPException(status_code=404, detail="Portfolio not found.")

    logger.info(f"Running backtest for user {current_user.email}, portfolio_id={portfolio.id}")

    data = json.loads(portfolio.weight_json)
    segments = data.get("segments", [])
    if not segments and isinstance(data, list):
        segments = data

    from vector_encoder import RLRecommender, encode_multi_horizon
    recommender = RLRecommender()
    if not recommender._initialized or recommender.daily_returns is None:
        raise HTTPException(status_code=500, detail="Market data unavailable")

    # Use the actual historical returns
    returns = recommender.daily_returns.fillna(0.0)
    # We'll backtest over the last 10 years of available data
    years_to_test = 10
    days_to_test = years_to_test * 252
    
    if len(returns) < days_to_test:
        days_to_test = len(returns)
        
    back_returns = returns.iloc[-days_to_test:]
    dates = back_returns.index
    
    # Simple backtest logic
    weights_dict = segments[0]["weights"] if segments else {}
    
    num_assets = len(returns.columns)
    wa = np.zeros(num_assets)
    col_to_idx = {col: i for i, col in enumerate(returns.columns)}
    for t, wt in weights_dict.items():
        if t in col_to_idx:
            wa[col_to_idx[t]] = wt
    if wa.sum() > 0: wa /= wa.sum()
    
    # Vectorized calculation for speed
    daily_rets = back_returns.values @ wa
    
    current_bal = initial_investment
    path = []
    path.append({"date": str(dates[0].date()), "balance": round(current_bal, 2)})
    
    for i in range(1, len(dates)):
        current_bal *= (1 + daily_rets[i])
        # Add monthly contribution
        if i % 21 == 0:
            current_bal += monthly_contrib
        path.append({"date": str(dates[i].date()), "balance": round(current_bal, 2)})
        
    return {
        "portfolio_id": portfolio.id,
        "backtest": path,
        "stats": {
            "start_date": str(dates[0].date()),
            "end_date": str(dates[-1].date()),
            "total_return": round((current_bal / initial_investment - 1) * 100, 2),
            "final_balance": round(current_bal, 2)
        }
    }


@router.post("/simulate_benchmark")
def run_simulation_benchmark(
    req: schemas.SimulationBenchmarkRequest,
    current_user: models.User = Depends(get_current_user),
):
    """
    Run simulation for a standard benchmark (S&P500, etc) with caching.
    """
    cache_key = f"{req.benchmark_type}_{req.projection_years}"
    if cache_key in _BENCHMARK_CACHE:
        logger.info(f"Using cached benchmark simulation for: {cache_key}")
        return _BENCHMARK_CACHE[cache_key]

    logger.info(f"Calculating fresh benchmark simulation for: {cache_key}")
    if req.benchmark_type == "sp500":
        segments = [{"horizon_years": (0, req.projection_years), "weights": {"VOO": 1.0}}]
    elif req.benchmark_type == "conservative_60_40":
        segments = [{"horizon_years": (0, req.projection_years), "weights": {"VOO": 0.6, "BND": 0.4}}]
    else:
        raise HTTPException(status_code=400, detail="Invalid benchmark type")

    try:
        from vector_encoder import simulate_multi_horizon_portfolio
        sim = simulate_multi_horizon_portfolio(
            segments=segments,
            projection_years=req.projection_years
        )
        _BENCHMARK_CACHE[cache_key] = sim
        return sim
    except Exception as e:
        logger.error(f"Benchmark simulation crash: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
