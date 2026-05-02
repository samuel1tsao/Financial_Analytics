from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from market_data import get_market_data
from database import SessionLocal
from market_models import StockPrice, CompanyInfo, FinancialStatement, CorporateAction, TickerLookupCache
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from sync_service import SYNC_STATE, run_global_sync_background
import asyncio

router = APIRouter(prefix="/api/v1", tags=["market"])
@router.get("/market/stats")
def market_stats():
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
    ticker: str = Query(..., description="Ticker symbol"),
    range: str = Query("1y", description="Time horizon")
):
    ticker = ticker.upper()
    db = SessionLocal()
    try:
        today = date.today()
        if range == "1y": start_date = today - relativedelta(years=1)
        elif range == "3y": start_date = today - relativedelta(years=3)
        elif range == "5y": start_date = today - relativedelta(years=5)
        elif range == "ytd": start_date = date(today.year, 1, 1)
        else: start_date = date(1900, 1, 1)
            
        prices = db.query(StockPrice).filter(
            StockPrice.ticker == ticker,
            StockPrice.date >= start_date
        ).order_by(StockPrice.date.asc()).all()
        
        if not prices:
            raise HTTPException(status_code=404, detail="Not found")
            
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

@router.get("/market/sync/status")
def sync_status():
    return SYNC_STATE

@router.post("/market/sync/trigger")
async def sync_trigger(background_tasks: BackgroundTasks):
    if SYNC_STATE["is_running"]:
        return {"status": "ignored", "message": "Global Sync is already running."}
    
    background_tasks.add_task(run_global_sync_background)
    return {"status": "started", "message": "Global Background Sync submitted."}


async def _yfinance_ticker_is_real(ticker: str) -> bool:
    """Check yfinance in a thread to avoid blocking the event loop."""
    import yfinance as yf
    def _check():
        try:
            # Create ticker object
            t_obj = yf.Ticker(ticker)
            # Use fast_info for minimal network overhead
            info = t_obj.fast_info
            
            # If fast_info works, check for price or currency
            if info and hasattr(info, 'last_price') and info.last_price is not None:
                return True
                
            # Fallback to history check (1 day) to confirm it exists
            hist = t_obj.history(period="1d")
            return not hist.empty
        except Exception:
            return False
            
    try:
        # Wrap the thread call in a timeout to prevent hanging the request
        return await asyncio.wait_for(asyncio.to_thread(_check), timeout=5.0)
    except asyncio.TimeoutError:
        return False


async def _sync_single_ticker_background(ticker: str):
    """Thin wrapper to sync a single newly discovered ticker."""
    from market_data import sync_market_data, sync_company_info, sync_historical_deep
    await asyncio.to_thread(sync_market_data, [ticker])
    await asyncio.to_thread(sync_company_info, [ticker])
    await asyncio.to_thread(sync_historical_deep, [ticker])


@router.get("/market/lookup/{ticker}")
async def market_lookup(ticker: str, background_tasks: BackgroundTasks):
    """
    Resolve an arbitrary ticker symbol:
    1. Already in CompanyInfo → return immediately (fully synced).
    2. Cached as invalid in TickerLookupCache → return invalid (no network hit).
    3. Try yfinance live lookup:
       a. Real → cache as valid, trigger background sync, return syncing.
       b. Fake → cache as invalid, return invalid.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty.")

    db = SessionLocal()
    try:
        # 1. Already fully synced?
        if db.query(CompanyInfo).filter(CompanyInfo.ticker == ticker).first():
            return {"ticker": ticker, "valid": True, "in_db": True, "status": "synced"}

        # 2. Known invalid?
        cached = db.query(TickerLookupCache).filter(TickerLookupCache.ticker == ticker).first()
        if cached:
            if cached.is_valid:
                return {"ticker": ticker, "valid": True, "in_db": False, "status": "syncing"}
            else:
                return {"ticker": ticker, "valid": False, "status": "known_invalid"}

        # 3. Live yfinance lookup (non-blocking)
        is_real = await _yfinance_ticker_is_real(ticker)

        now = datetime.utcnow()
        if is_real:
            # Cache as valid + kick off a focused background sync
            entry = TickerLookupCache(ticker=ticker, is_valid=True, source="yfinance", checked_at=now)
            db.merge(entry)
            db.commit()
            background_tasks.add_task(_sync_single_ticker_background, ticker)
            return {"ticker": ticker, "valid": True, "in_db": False, "status": "syncing"}
        else:
            entry = TickerLookupCache(ticker=ticker, is_valid=False, source="yfinance", checked_at=now)
            db.merge(entry)
            db.commit()
            return {"ticker": ticker, "valid": False, "status": "not_found"}
    finally:
        db.close()

@router.get("/market/asset/{ticker}")
def market_asset_details(ticker: str):
    """
    Returns fundamental data, recent statements, and corporate actions for an asset.
    """
    ticker = ticker.upper()
    db = SessionLocal()
    try:
        info = db.query(CompanyInfo).filter(CompanyInfo.ticker == ticker).first()
        statements = db.query(FinancialStatement).filter(FinancialStatement.ticker == ticker).order_by(FinancialStatement.report_date.desc()).limit(4).all()
        actions = db.query(CorporateAction).filter(CorporateAction.ticker == ticker).order_by(CorporateAction.date.desc()).limit(10).all()
        
        if not info and not statements:
            raise HTTPException(status_code=404, detail="Asset fundamentals not found.")
            
        # Convert SQLAlchemy objects to dicts
        info_dict = {c.name: getattr(info, c.name) for c in info.__table__.columns} if info else {}
        stmts_list = [{c.name: getattr(s, c.name) for c in s.__table__.columns if c.name != 'id'} for s in statements]
        acts_list = [{c.name: getattr(a, c.name) for c in a.__table__.columns if c.name != 'id'} for a in actions]
        
        return {
            "info": info_dict,
            "statements": stmts_list,
            "actions": acts_list
        }
    finally:
        db.close()

@router.get("/market/search")
def market_search(
    q: str = Query("", description="Search query for ticker or company name"),
    limit: int = Query(10, le=30)
):
    """
    Autocomplete search — returns ticker + name matches.
    """
    db = SessionLocal()
    try:
        query = db.query(CompanyInfo.ticker, CompanyInfo.name, CompanyInfo.asset_type, CompanyInfo.sector)
        if q:
            term = f"%{q.upper()}%"
            name_term = f"%{q}%"
            query = query.filter(
                (CompanyInfo.ticker.ilike(term)) |
                (CompanyInfo.name.ilike(name_term))
            )
        results = query.order_by(CompanyInfo.ticker).limit(limit).all()
        return [{"ticker": r.ticker, "name": r.name, "asset_type": r.asset_type, "sector": r.sector} for r in results]
    finally:
        db.close()

@router.get("/market/browse")
def market_browse(
    q: str = Query("", description="Search filter"),
    sector: str = Query("", description="Filter by sector"),
    asset_type: str = Query("", description="ETF or EQUITY"),
    sort_by: str = Query("market_cap", description="Sort field"),
    sort_dir: str = Query("desc", description="asc or desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(24, le=100),
):
    """
    Paginated, filterable browse of all synced assets.
    """
    db = SessionLocal()
    try:
        query = db.query(CompanyInfo)
        if q:
            term = f"%{q}%"
            query = query.filter(
                (CompanyInfo.ticker.ilike(term)) |
                (CompanyInfo.name.ilike(term))
            )
        if sector:
            query = query.filter(CompanyInfo.sector == sector)
        if asset_type:
            query = query.filter(CompanyInfo.asset_type == asset_type)

        total = query.count()

        sort_col = getattr(CompanyInfo, sort_by, CompanyInfo.market_cap)
        if sort_dir == "asc":
            query = query.order_by(sort_col.asc().nullslast())
        else:
            query = query.order_by(sort_col.desc().nullsfirst())

        items = query.offset((page - 1) * page_size).limit(page_size).all()
        data = []
        for info in items:
            data.append({
                "ticker": info.ticker,
                "name": info.name,
                "asset_type": info.asset_type,
                "sector": info.sector,
                "market_cap": info.market_cap,
                "forward_pe": info.forward_pe,
                "trailing_pe": info.trailing_pe,
                "dividend_yield": info.dividend_yield,
                "profit_margins": info.profit_margins,
                "beta": info.beta,
                "fifty_two_week_change": info.fifty_two_week_change,
                "five_year_average_return": info.five_year_average_return,
                "annual_report_expense_ratio": info.annual_report_expense_ratio,
                "revenue_growth": info.revenue_growth,
                "return_on_equity": info.return_on_equity,
            })

        # Distinct sectors for filter dropdown
        sectors = [r[0] for r in db.query(CompanyInfo.sector).filter(CompanyInfo.sector != None).distinct().order_by(CompanyInfo.sector).all()]

        return {"total": total, "page": page, "page_size": page_size, "items": data, "sectors": sectors}
    finally:
        db.close()


