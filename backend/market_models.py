"""
SQLAlchemy models for storing historical market data from yfinance.

- StockPrice: raw daily adjusted close prices
- TickerFeature: computed rolling 1-year returns and volatilities (feature vectors)
- CompanyInfo: rich fundamental data per ticker
- TickerLookupCache: permanent cache of yfinance lookup results (valid + invalid)
"""
from sqlalchemy import Column, String, Float, Date, DateTime, Index, Integer, Boolean
from database import Base
from datetime import datetime

class TickerLookupCache(Base):
    """
    Permanent cache of yfinance ticker lookup results.
    Prevents re-querying Yahoo Finance for tickers we've already checked.
    is_valid=True  → confirmed real ticker (may or may not be fully synced yet)
    is_valid=False → confirmed invalid / not found on Yahoo Finance
    """
    __tablename__ = "ticker_lookup_cache"

    ticker     = Column(String, primary_key=True, index=True)
    is_valid   = Column(Boolean, nullable=False)
    source     = Column(String, nullable=True)  # 'yfinance', 'wikipedia', 'manual'
    checked_at = Column(DateTime, default=datetime.utcnow)

class CompanyInfo(Base):
    """Fundamental data for a given ticker."""
    __tablename__ = "company_info"

    ticker = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    asset_type = Column(String, nullable=True)  # e.g., 'EQUITY', 'ETF'
    description = Column(String, nullable=True) # Full length company bio
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    
    # Audit timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)

    # Health & Valuation
    market_cap = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    forward_pe = Column(Float, nullable=True)
    trailing_pe = Column(Float, nullable=True)
    price_to_book = Column(Float, nullable=True)
    price_to_sales = Column(Float, nullable=True)
    enterprise_to_ebitda = Column(Float, nullable=True)
    debt_to_equity = Column(Float, nullable=True)
    current_ratio = Column(Float, nullable=True)
    quick_ratio = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)

    # Margins & Growth
    profit_margins = Column(Float, nullable=True)
    operating_margins = Column(Float, nullable=True)
    return_on_equity = Column(Float, nullable=True)
    return_on_assets = Column(Float, nullable=True)
    revenue_growth = Column(Float, nullable=True)
    earnings_growth = Column(Float, nullable=True)

    # Sentiment & Stock Context
    held_percent_insiders = Column(Float, nullable=True)
    held_percent_institutions = Column(Float, nullable=True)
    short_ratio = Column(Float, nullable=True)
    fifty_two_week_high = Column(Float, nullable=True)
    fifty_two_week_low = Column(Float, nullable=True)
    fifty_two_week_change = Column(Float, nullable=True)
    target_mean_price = Column(Float, nullable=True)

    # ETF Specifics
    ytd_return = Column(Float, nullable=True)
    three_year_average_return = Column(Float, nullable=True)
    five_year_average_return = Column(Float, nullable=True)
    total_assets = Column(Float, nullable=True)
    annual_report_expense_ratio = Column(Float, nullable=True)
    fund_family = Column(String, nullable=True)

    # Logistics
    full_time_employees = Column(Integer, nullable=True)
    country = Column(String, nullable=True)
    state = Column(String, nullable=True)

class FinancialStatement(Base):
    """Historical financial statements (Income, Balance, Cashflow)."""
    __tablename__ = "financial_statements"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    report_date = Column(Date, nullable=False)
    period_type = Column(String, nullable=False) # 'annual' or 'quarterly'
    
    total_revenue = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    gross_profit = Column(Float, nullable=True)
    operating_cash_flow = Column(Float, nullable=True)
    free_cash_flow = Column(Float, nullable=True)
    total_assets = Column(Float, nullable=True)
    total_debt = Column(Float, nullable=True)
    total_liabilities = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_fin_stmt_ticker_date", "ticker", "report_date"),
    )

class CorporateAction(Base):
    """Historical dividends and stock splits."""
    __tablename__ = "corporate_actions"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True, nullable=False)
    date = Column(Date, nullable=False)
    action_type = Column(String, nullable=False) # 'dividend' or 'split'
    value = Column(Float, nullable=False)

    __table_args__ = (
        Index("ix_corp_act_ticker_date", "ticker", "date"),
    )


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

    # Expanded Historical Volatility
    rolling_30d_volatility = Column(Float, nullable=True)
    rolling_90d_volatility = Column(Float, nullable=True)

    # Theoretical ATM Greeks (30D)
    greeks_30d_delta = Column(Float, nullable=True)
    greeks_30d_gamma = Column(Float, nullable=True)
    greeks_30d_theta = Column(Float, nullable=True)
    greeks_30d_vega = Column(Float, nullable=True)

    # Theoretical ATM Greeks (90D)
    greeks_90d_delta = Column(Float, nullable=True)
    greeks_90d_gamma = Column(Float, nullable=True)
    greeks_90d_theta = Column(Float, nullable=True)
    greeks_90d_vega = Column(Float, nullable=True)

    # Theoretical ATM Greeks (1YR)
    greeks_1yr_delta = Column(Float, nullable=True)
    greeks_1yr_gamma = Column(Float, nullable=True)
    greeks_1yr_theta = Column(Float, nullable=True)
    greeks_1yr_vega = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_ticker_features_ticker_date", "ticker", "date"),
    )
