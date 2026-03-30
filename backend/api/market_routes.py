from fastapi import APIRouter, HTTPException, Query
from market_data import get_market_data
from database import SessionLocal
from market_models import StockPrice
from datetime import date
from dateutil.relativedelta import relativedelta

router = APIRouter()
@router.get("/market/stats")
def market_stats():
    """
    Return current market data from the database:
    expected returns, volatilities, and correlation matrix for the asset universe.
    """
    md = get_market_data()

    return {
        "last_updated": str(md.last_updated) if md.last_updated else None,
        "tickers": list(md.expected_returns.keys()),
        "expected_returns": md.expected_returns,
        "volatilities": md.volatilities,
        "correlation_matrix": md.correlation_matrix,
        "data_source": "database" if md.expected_returns else "fallback_hardcoded",
    }


@router.get("/market/history")
def market_history(
    ticker: str = Query(..., description="Ticker symbol (e.g., VOO)"),
    range: str = Query("1y", description="Time horizon: 1y, 3y, 5y, ytd, or max")
):
    """
    Return historical daily closing prices for a specific asset to render UI charts.
    Also calculates the ending % change over the selected horizon.
    """
    ticker = ticker.upper()
    db = SessionLocal()
    
    try:
        # Determine start date
        today = date.today()
        if range == "1y":
            start_date = today - relativedelta(years=1)
        elif range == "3y":
            start_date = today - relativedelta(years=3)
        elif range == "5y":
            start_date = today - relativedelta(years=5)
        elif range == "ytd":
            start_date = date(today.year, 1, 1)
        else:
            start_date = date(1900, 1, 1)  # max
            
        prices = db.query(StockPrice).filter(
            StockPrice.ticker == ticker,
            StockPrice.date >= start_date
        ).order_by(StockPrice.date.asc()).all()
        
        if not prices:
            raise HTTPException(status_code=404, detail=f"No historical data found for '{ticker}' in the '{range}' range.")
            
        history = [{"date": str(p.date), "adj_close": p.adj_close} for p in prices]
        
        first_price = history[0]["adj_close"]
        last_price = history[-1]["adj_close"]
        percent_change = (last_price / first_price - 1.0) * 100 if first_price else 0.0
        
        return {
            "ticker": ticker,
            "range": range,
            "history": history,
            "starting_price": first_price,
            "ending_price": last_price,
            "percent_change": round(percent_change, 2)
        }
        
    finally:
        db.close()
