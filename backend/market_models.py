"""
SQLAlchemy models for storing historical market data from yfinance.

- StockPrice: raw daily adjusted close prices
- TickerFeature: computed rolling 1-year returns and volatilities (feature vectors)
"""
from sqlalchemy import Column, String, Float, Date, Index
from database import Base


class StockPrice(Base):
    """Daily adjusted close price for a ticker, sourced from yfinance."""
    __tablename__ = "stock_prices"

    ticker = Column(String, primary_key=True, index=True)
    date = Column(Date, primary_key=True, index=True)
    adj_close = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_stock_prices_ticker_date", "ticker", "date"),
    )


class TickerFeature(Base):
    """
    Rolling 1-year annualized return and volatility for a ticker on a given date.
    These serve as feature vectors for the recommendation model.

    - rolling_1yr_return: (price_today / price_252_days_ago) - 1
    - rolling_1yr_volatility: std(daily_returns over last 252 days) * sqrt(252)
    """
    __tablename__ = "ticker_features"

    ticker = Column(String, primary_key=True, index=True)
    date = Column(Date, primary_key=True, index=True)
    rolling_1yr_return = Column(Float, nullable=False)
    rolling_1yr_volatility = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_ticker_features_ticker_date", "ticker", "date"),
    )
