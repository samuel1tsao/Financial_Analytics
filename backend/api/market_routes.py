from fastapi import APIRouter
from market_data import get_market_data

router = APIRouter(prefix="/api/v1", tags=["market"])


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
