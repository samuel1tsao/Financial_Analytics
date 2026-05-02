import asyncio
import logging
import pandas as pd
from datetime import date
from database import SessionLocal
from market_models import CompanyInfo
from congress_models import CongressTrade
from market_data import ALL_TICKERS, _fetch_and_store_prices, _compute_features_for_ticker, _get_latest_date_in_db, sync_market_data, sync_company_info, sync_historical_financials
import yfinance as yf
import time
import requests
import io
from datetime import timedelta

logger = logging.getLogger(__name__)

SYNC_STATE = {
    "is_running": False,
    "processed": 0,
    "total": 0,
    "progress": "Idle"
}

def get_global_tickers():
    """Scrapes Wikipedia for S&P 500, MidCap 400, and SmallCap 600 to get a deep, liquid 1,500 target universe."""
    tickers = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        # S&P 500
        html500 = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', headers=headers).text
        sp500 = pd.read_html(io.StringIO(html500))[0]
        tickers.extend(sp500['Symbol'].tolist())
        
        # S&P 400
        html400 = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', headers=headers).text
        sp400 = pd.read_html(io.StringIO(html400))[0]
        tickers.extend(sp400['Symbol'].tolist())
        
        # S&P 600
        html600 = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', headers=headers).text
        sp600 = pd.read_html(io.StringIO(html600))[0]
        tickers.extend(sp600['Symbol'].tolist())

        # Clean Wikipedia dots (BRK.B -> BRK-B for yfinance)
        tickers = [t.replace('.', '-') for t in tickers]
    except Exception as e:
        logger.error(f"Failed to fetch global tickers from Wikipedia: {e}")

    # Mix in custom ones
    db = SessionLocal()
    try:
        db_tickers = {t[0] for t in db.query(CongressTrade.ticker).distinct().all()}
        tickers.extend(db_tickers)
        tickers.extend(ALL_TICKERS)
    finally:
        db.close()
        
    return list(set(tickers))

async def run_global_sync_background():
    """Idempotent background loop catching up missing massive data quietly."""
    global SYNC_STATE
    if SYNC_STATE["is_running"]:
        return
        
    SYNC_STATE["is_running"] = True
    SYNC_STATE["progress"] = "Starting..."
    logger.info("Global Background Sync Starting...")
    
    try:
        tickers = get_global_tickers()
        SYNC_STATE["total"] = len(tickers)
        SYNC_STATE["processed"] = 0
        
        today = date.today()
        # For price history
        staleness_threshold = today - timedelta(days=3)
        
        db = SessionLocal()
        
        for t in tickers:
            if not SYNC_STATE["is_running"]: 
                break # Allow graceful termination
            
            # Check idempotency / skip today
            c_info = db.query(CompanyInfo).filter(CompanyInfo.ticker == t).first()
            already_ran_today = c_info is not None and c_info.updated_at and c_info.updated_at.date() >= today
            
            if already_ran_today:
                logger.info(f"Background Sync: Skipped {t} - already completely fresh today.")
                SYNC_STATE["processed"] += 1
                SYNC_STATE["progress"] = f"Skipping fresh assets ({SYNC_STATE['processed']}/{SYNC_STATE['total']})"
                continue

            SYNC_STATE["progress"] = f"Fetching prices for {t} ({SYNC_STATE['processed']}/{SYNC_STATE['total']})"
            await asyncio.to_thread(sync_market_data, override_tickers=[t])

            SYNC_STATE["progress"] = f"Fetching company info for {t} ({SYNC_STATE['processed']}/{SYNC_STATE['total']})"
            await asyncio.to_thread(sync_company_info, override_tickers=[t])

            SYNC_STATE["progress"] = f"Fetching deep financials for {t} ({SYNC_STATE['processed']}/{SYNC_STATE['total']})"
            await asyncio.to_thread(sync_historical_financials, override_tickers=[t])
            
            SYNC_STATE["processed"] += 1
            
            # Polite pause between tickers to preserve bandwidth and avoid rate limits
            # for real-time user requests (like ticker lookups).
            await asyncio.sleep(2.0)

    except Exception as e:
        logger.error(f"Global sync crashed: {e}")
    finally:
        SYNC_STATE["is_running"] = False
        SYNC_STATE["progress"] = "Idle"
        logger.info("Global Background Sync Terminated.")
