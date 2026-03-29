"""
Market Data Service: Fetches, stores, and serves historical market data from yfinance.

On startup, syncs the StockPrice table with yfinance (incremental — only missing days).
Computes rolling 1-year returns/volatilities into TickerFeature table.
Serves aggregated stats (returns, volatilities, correlation matrix) from DB.
"""
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from database import SessionLocal
from market_models import StockPrice, TickerFeature

logger = logging.getLogger(__name__)

# ─── Asset Universe ──────────────────────────────────────────────────────────
EQUITY_TICKERS = ["VOO", "QQQ", "VTI", "VXUS", "VGT", "ARKK", "VNQ", "VWO"]
BOND_TICKERS = ["BND", "SGOV", "TLT", "TIPS"]
ALL_TICKERS = EQUITY_TICKERS + BOND_TICKERS

# Minimum years of history to fetch on first sync
MIN_HISTORY_YEARS = 15

# Trading days in a year (for annualization)
TRADING_DAYS = 252


@dataclass
class MarketData:
    """Aggregated market statistics read from the database."""
    expected_returns: dict[str, float] = field(default_factory=dict)
    volatilities: dict[str, float] = field(default_factory=dict)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    last_updated: date | None = None


def _get_latest_date_in_db(db: Session, ticker: str) -> date | None:
    """Return the most recent date stored for a ticker, or None if no data."""
    row = (
        db.query(StockPrice.date)
        .filter(StockPrice.ticker == ticker)
        .order_by(StockPrice.date.desc())
        .first()
    )
    return row[0] if row else None


def _fetch_and_store_prices(db: Session, ticker: str, start_date: date, end_date: date) -> int:
    """
    Fetch daily adjusted close from yfinance and bulk-insert into StockPrice.
    Returns number of rows inserted.
    """
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        logger.info(f"Fetching {ticker} from {start_str} to {end_str}")

        data = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)

        if data.empty:
            logger.warning(f"No data returned for {ticker}")
            return 0

        # yf.download returns MultiIndex columns when single ticker; flatten
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            logger.warning(f"No 'Close' column for {ticker}, columns: {list(data.columns)}")
            return 0

        rows_inserted = 0
        for idx, row in data.iterrows():
            price_date = idx.date() if hasattr(idx, "date") else idx
            adj_close = float(row["Close"])

            # Check if row already exists (avoid duplicates)
            existing = (
                db.query(StockPrice)
                .filter(StockPrice.ticker == ticker, StockPrice.date == price_date)
                .first()
            )
            if not existing:
                db.add(StockPrice(ticker=ticker, date=price_date, adj_close=adj_close))
                rows_inserted += 1

        db.commit()
        logger.info(f"Inserted {rows_inserted} rows for {ticker}")
        return rows_inserted

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        db.rollback()
        return 0


def _compute_features_for_ticker(db: Session, ticker: str) -> int:
    """
    Compute rolling 1-year return and volatility for each date and store in TickerFeature.
    Returns number of feature rows created.
    """
    # Fetch all prices for this ticker, ordered by date
    prices = (
        db.query(StockPrice)
        .filter(StockPrice.ticker == ticker)
        .order_by(StockPrice.date.asc())
        .all()
    )

    if len(prices) < TRADING_DAYS + 1:
        logger.warning(f"Not enough data for {ticker} to compute rolling features "
                       f"(need {TRADING_DAYS + 1}, have {len(prices)})")
        return 0

    # Find the latest feature date already computed
    latest_feature = (
        db.query(TickerFeature.date)
        .filter(TickerFeature.ticker == ticker)
        .order_by(TickerFeature.date.desc())
        .first()
    )
    latest_feature_date = latest_feature[0] if latest_feature else None

    # Build arrays
    dates = [p.date for p in prices]
    closes = np.array([p.adj_close for p in prices])

    # Daily returns
    daily_returns = np.diff(closes) / closes[:-1]  # len = len(closes) - 1

    rows_created = 0
    # Start from index TRADING_DAYS (first date with 252 trailing days of returns)
    for i in range(TRADING_DAYS, len(closes)):
        current_date = dates[i]

        # Skip already-computed dates
        if latest_feature_date and current_date <= latest_feature_date:
            continue

        # Rolling 1-year return: price_today / price_252_days_ago - 1
        rolling_return = (closes[i] / closes[i - TRADING_DAYS]) - 1.0

        # Rolling 1-year volatility: std of last 252 daily returns, annualized
        trailing_returns = daily_returns[i - TRADING_DAYS: i]
        rolling_vol = float(np.std(trailing_returns, ddof=1) * np.sqrt(TRADING_DAYS))

        db.add(TickerFeature(
            ticker=ticker,
            date=current_date,
            rolling_1yr_return=round(float(rolling_return), 6),
            rolling_1yr_volatility=round(rolling_vol, 6),
        ))
        rows_created += 1

    db.commit()
    if rows_created > 0:
        logger.info(f"Computed {rows_created} feature rows for {ticker}")
    return rows_created


def sync_market_data() -> dict:
    """
    Sync market data from yfinance into the database.
    Only fetches data that is missing (incremental sync).
    Called on backend startup.

    Returns a summary dict of what was synced.
    """
    db = SessionLocal()
    summary = {"synced": [], "skipped": [], "errors": []}
    today = date.today()
    # Consider data stale if latest row is before the previous trading day.
    # Use 3 days ago to account for weekends (Friday close → Monday startup).
    staleness_threshold = today - timedelta(days=3)

    try:
        for ticker in ALL_TICKERS:
            latest = _get_latest_date_in_db(db, ticker)

            if latest and latest >= staleness_threshold:
                summary["skipped"].append(ticker)
                continue

            # Determine start date for fetch
            if latest:
                # Incremental: fetch from the day after the latest stored date
                start = latest + timedelta(days=1)
            else:
                # First sync: fetch max history (15+ years)
                start = today - timedelta(days=MIN_HISTORY_YEARS * 365)

            rows = _fetch_and_store_prices(db, ticker, start, today)
            if rows > 0:
                summary["synced"].append({"ticker": ticker, "rows": rows})
            elif rows == 0 and not latest:
                summary["errors"].append(ticker)

            # Compute/update feature vectors
            _compute_features_for_ticker(db, ticker)

    finally:
        db.close()

    logger.info(f"Market data sync complete: {summary}")
    return summary


def get_market_data() -> MarketData:
    """
    Read aggregated market statistics from the database.
    Returns expected returns, volatilities, and correlation matrix for all tickers.
    This is a fast DB read — no network calls.
    """
    db = SessionLocal()
    try:
        result = MarketData()

        # ─── Get latest feature for each ticker ──────────────────────────────
        for ticker in ALL_TICKERS:
            latest_feature = (
                db.query(TickerFeature)
                .filter(TickerFeature.ticker == ticker)
                .order_by(TickerFeature.date.desc())
                .first()
            )
            if latest_feature:
                result.expected_returns[ticker] = latest_feature.rolling_1yr_return
                result.volatilities[ticker] = latest_feature.rolling_1yr_volatility
                if result.last_updated is None or latest_feature.date > result.last_updated:
                    result.last_updated = latest_feature.date

        # ─── Compute full pairwise correlation matrix from daily returns ─────
        # Fetch all prices for tickers that have data
        available_tickers = list(result.expected_returns.keys())
        if len(available_tickers) >= 2:
            price_frames = {}
            for ticker in available_tickers:
                prices = (
                    db.query(StockPrice.date, StockPrice.adj_close)
                    .filter(StockPrice.ticker == ticker)
                    .order_by(StockPrice.date.asc())
                    .all()
                )
                if prices:
                    df = pd.DataFrame(prices, columns=["date", ticker])
                    df.set_index("date", inplace=True)
                    price_frames[ticker] = df

            if price_frames:
                # Merge all price series on date
                combined = pd.concat(price_frames.values(), axis=1, join="inner")
                # Daily returns
                returns_df = combined.pct_change().dropna()

                if len(returns_df) > TRADING_DAYS:
                    corr = returns_df.corr()
                    result.correlation_matrix = {
                        t1: {t2: round(float(corr.loc[t1, t2]), 4)
                             for t2 in corr.columns if t2 in available_tickers}
                        for t1 in corr.index if t1 in available_tickers
                    }

        return result

    finally:
        db.close()


def get_covariance_matrix(tickers: list[str], market_data: MarketData | None = None) -> np.ndarray:
    """
    Build a real variance-covariance matrix from DB data.
    Falls back to simplified category-based approach for any missing tickers.
    """
    if market_data is None:
        market_data = get_market_data()

    n = len(tickers)
    cov = np.zeros((n, n))

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            vol1 = market_data.volatilities.get(t1)
            vol2 = market_data.volatilities.get(t2)

            if vol1 is None or vol2 is None:
                # Fallback: use hardcoded defaults
                from vector_encoder import FALLBACK_VOLATILITIES
                vol1 = vol1 or FALLBACK_VOLATILITIES.get(t1, 0.15)
                vol2 = vol2 or FALLBACK_VOLATILITIES.get(t2, 0.15)

            if i == j:
                cov[i][j] = vol1 ** 2
            else:
                # Try real correlation
                corr_val = (
                    market_data.correlation_matrix
                    .get(t1, {})
                    .get(t2)
                )
                if corr_val is not None:
                    cov[i][j] = corr_val * vol1 * vol2
                else:
                    # Fallback: category-based correlation
                    from vector_encoder import FALLBACK_CATEGORY_CORRELATIONS, classify_asset
                    cat1 = classify_asset(t1)
                    cat2 = classify_asset(t2)
                    pair = tuple(sorted([cat1, cat2]))
                    fallback_corr = FALLBACK_CATEGORY_CORRELATIONS.get(pair, 0.5)
                    cov[i][j] = fallback_corr * vol1 * vol2

    return cov
