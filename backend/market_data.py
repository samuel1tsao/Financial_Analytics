"""
Market Data Service: Fetches, stores, and serves historical market data from yfinance.

On startup, syncs the StockPrice table with yfinance (incremental — only missing days).
Computes rolling 1-year returns/volatilities into TickerFeature table.
Serves aggregated stats (returns, volatilities, correlation matrix) from DB.
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal
from market_models import StockPrice, TickerFeature, CompanyInfo, FinancialStatement, CorporateAction
from congress_models import CongressTrade
from options_data import get_theoretical_atm_greeks

logger = logging.getLogger(__name__)

# ─── Asset Universe ──────────────────────────────────────────────────────────
EQUITY_TICKERS = ["VOO", "QQQ", "VTI", "VXUS", "VGT", "ARKK", "VNQ", "VWO"]
BOND_TICKERS = ["BND", "SGOV", "TLT", "TIPS"]
ALL_TICKERS = EQUITY_TICKERS + BOND_TICKERS

# Minimum years of history to fetch on first sync (50 years effectively fetches maximum available history)
MIN_HISTORY_YEARS = 50

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
        
        # Polite scraping etiquette to avoid yfinance rate limits
        time.sleep(1.0)

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

        # Rolling volatilities
        trailing_returns = daily_returns[i - TRADING_DAYS: i]
        vol_30d = float(np.std(daily_returns[i - 30: i], ddof=1) * np.sqrt(252))
        vol_90d = float(np.std(daily_returns[i - 90: i], ddof=1) * np.sqrt(252))
        vol_1yr = float(np.std(trailing_returns, ddof=1) * np.sqrt(TRADING_DAYS))

        # Greeks
        S = closes[i]
        greeks_30d = get_theoretical_atm_greeks(S, vol_30d, 30)
        greeks_90d = get_theoretical_atm_greeks(S, vol_90d, 90)
        greeks_1yr = get_theoretical_atm_greeks(S, vol_1yr, 365)

        db.add(TickerFeature(
            ticker=ticker,
            date=current_date,
            rolling_1yr_return=round(float(rolling_return), 6),
            rolling_1yr_volatility=round(vol_1yr, 6),
            rolling_30d_volatility=round(vol_30d, 6),
            rolling_90d_volatility=round(vol_90d, 6),
            greeks_30d_delta=greeks_30d["delta"],
            greeks_30d_gamma=greeks_30d["gamma"],
            greeks_30d_theta=greeks_30d["theta"],
            greeks_30d_vega=greeks_30d["vega"],
            greeks_90d_delta=greeks_90d["delta"],
            greeks_90d_gamma=greeks_90d["gamma"],
            greeks_90d_theta=greeks_90d["theta"],
            greeks_90d_vega=greeks_90d["vega"],
            greeks_1yr_delta=greeks_1yr["delta"],
            greeks_1yr_gamma=greeks_1yr["gamma"],
            greeks_1yr_theta=greeks_1yr["theta"],
            greeks_1yr_vega=greeks_1yr["vega"],
        ))
        rows_created += 1

    db.commit()
    if rows_created > 0:
        logger.info(f"Computed {rows_created} feature rows for {ticker}")
    return rows_created


def sync_market_data(override_tickers: list = None) -> dict:
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
        # Dynamically append any custom tickers traded by congress members
        if override_tickers is None:
            db_tickers = {t[0] for t in db.query(CongressTrade.ticker).distinct().all()}
            all_sync_tickers = list(set(ALL_TICKERS).union(db_tickers))
        else:
            all_sync_tickers = override_tickers

        for ticker in all_sync_tickers:
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

def sync_company_info(override_tickers: list = None) -> dict:
    """Fetch yfinance fundamentals for all tracked tickers and store in CompanyInfo."""
    db = SessionLocal()
    summary = {"updated": 0, "errors": []}
    try:
        if override_tickers is None:
            db_tickers = {t[0] for t in db.query(CongressTrade.ticker).distinct().all()}
            all_sync_tickers = list(set(ALL_TICKERS).union(db_tickers))
        else:
            all_sync_tickers = override_tickers
        
        for ticker in all_sync_tickers:
            try:
                time.sleep(1.0) # Polite scraping
                info = yf.Ticker(ticker).info
                if not info:
                    continue
                
                existing = db.query(CompanyInfo).filter(CompanyInfo.ticker == ticker).first()
                if not existing:
                    existing = CompanyInfo(ticker=ticker)
                    db.add(existing)
                
                existing.name = info.get("shortName") or info.get("longName")
                existing.asset_type = info.get("quoteType")
                existing.description = info.get("longBusinessSummary") or info.get("description")
                existing.sector = info.get("sector")
                existing.industry = info.get("industry")
                
                # Health & Valuation
                existing.market_cap = info.get("marketCap")
                existing.beta = info.get("beta")
                existing.forward_pe = info.get("forwardPE")
                existing.trailing_pe = info.get("trailingPE")
                existing.price_to_book = info.get("priceToBook")
                existing.price_to_sales = info.get("priceToSalesTrailing12Months")
                existing.enterprise_to_ebitda = info.get("enterpriseToEbitda")
                existing.debt_to_equity = info.get("debtToEquity")
                existing.current_ratio = info.get("currentRatio")
                existing.quick_ratio = info.get("quickRatio")
                existing.dividend_yield = info.get("dividendYield")

                # Margins & Growth
                existing.profit_margins = info.get("profitMargins")
                existing.operating_margins = info.get("operatingMargins")
                existing.return_on_equity = info.get("returnOnEquity")
                existing.return_on_assets = info.get("returnOnAssets")
                existing.revenue_growth = info.get("revenueGrowth")
                existing.earnings_growth = info.get("earningsGrowth")

                # Sentiment & Context
                existing.held_percent_insiders = info.get("heldPercentInsiders")
                existing.held_percent_institutions = info.get("heldPercentInstitutions")
                existing.short_ratio = info.get("shortRatio")
                existing.fifty_two_week_high = info.get("fiftyTwoWeekHigh")
                existing.fifty_two_week_low = info.get("fiftyTwoWeekLow")
                existing.fifty_two_week_change = info.get("52WeekChange")
                existing.target_mean_price = info.get("targetMeanPrice")

                # ETF Specifics
                existing.ytd_return = info.get("ytdReturn")
                existing.three_year_average_return = info.get("threeYearAverageReturn")
                existing.five_year_average_return = info.get("fiveYearAverageReturn")
                existing.total_assets = info.get("totalAssets")
                existing.annual_report_expense_ratio = info.get("annualReportExpenseRatio")
                existing.fund_family = info.get("fundFamily")

                # Logistics
                existing.full_time_employees = info.get("fullTimeEmployees")
                existing.country = info.get("country")
                existing.state = info.get("state")

                # Audit timestamps automatically handle created_at/updated_at defaults
                
                summary["updated"] += 1
            except Exception as e:
                logger.error(f"Failed pulling info for {ticker}: {e}")
                summary["errors"].append(ticker)
        db.commit()
    finally:
        db.close()
    
    logger.info(f"Company Info sync complete: {summary}")
    return summary

def sync_historical_financials(override_tickers: list = None) -> dict:
    """Fetch deep history (statements, dividends, splits) politely."""
    db = SessionLocal()
    summary = {"updated": 0, "errors": []}
    try:
        if override_tickers is None:
            db_tickers = {t[0] for t in db.query(CongressTrade.ticker).distinct().all()}
            all_sync_tickers = list(set(ALL_TICKERS).union(db_tickers))
        else:
            all_sync_tickers = override_tickers
        
        for ticker in all_sync_tickers:
            try:
                time.sleep(2.0) # Very polite scraping
                tk = yf.Ticker(ticker)
                info = tk.info
                is_etf = info.get("quoteType", "") == "ETF"
                
                # Historic Corp Actions
                divs = tk.dividends
                splits = tk.splits
                
                if not divs.empty:
                    for d_date, val in divs.items():
                        dt = d_date.date() if hasattr(d_date, 'date') else d_date
                        ext = db.query(CorporateAction).filter_by(ticker=ticker, date=dt, action_type="dividend").first()
                        if not ext: db.add(CorporateAction(ticker=ticker, date=dt, action_type="dividend", value=float(val)))
                            
                if not splits.empty:
                    for s_date, val in splits.items():
                        dt = s_date.date() if hasattr(s_date, 'date') else s_date
                        ext = db.query(CorporateAction).filter_by(ticker=ticker, date=dt, action_type="split").first()
                        if not ext: db.add(CorporateAction(ticker=ticker, date=dt, action_type="split", value=float(val)))

                # Historic Statements for non-ETFs
                if not is_etf:
                    ann_inc = tk.financials
                    if ann_inc is not None and not ann_inc.empty:
                        for col in ann_inc.columns:
                            dt = col.date() if hasattr(col, 'date') else col
                            ext = db.query(FinancialStatement).filter_by(ticker=ticker, report_date=dt, period_type="annual").first()
                            if not ext:
                                fs = FinancialStatement(ticker=ticker, report_date=dt, period_type="annual")
                                if "Total Revenue" in ann_inc.index and pd.notna(ann_inc.loc["Total Revenue", col]):
                                    fs.total_revenue = float(ann_inc.loc["Total Revenue", col])
                                if "Net Income" in ann_inc.index and pd.notna(ann_inc.loc["Net Income", col]):
                                    fs.net_income = float(ann_inc.loc["Net Income", col])
                                if "Gross Profit" in ann_inc.index and pd.notna(ann_inc.loc["Gross Profit", col]):
                                    fs.gross_profit = float(ann_inc.loc["Gross Profit", col])
                                db.add(fs)
                
                summary["updated"] += 1
            except Exception as e:
                logger.error(f"Failed historical deep sync for {ticker}: {e}")
                summary["errors"].append(ticker)
        db.commit()
    finally:
        db.close()
    
    logger.info(f"Historical Deep Sync complete: {summary}")
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
        
        # Dynamically append tracked tickers
        db_tickers = {t[0] for t in db.query(CongressTrade.ticker).distinct().all()}
        all_stat_tickers = list(set(ALL_TICKERS).union(db_tickers))

        # ─── Get latest feature for each ticker ──────────────────────────────
        for ticker in all_stat_tickers:
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
