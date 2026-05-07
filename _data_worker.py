"""
_data_worker.py
───────────────
Member A: Data Ingestion, Caching, Preprocessing & Diagnostics.

Public API:
    fetch_macro_universe()                     → list[str]
    generate_dataset_member_a(tickers, config)  → (master_df, price_matrix, volume_matrix, daily_returns, drip_daily_returns)
    run_data_diagnostics(master_df, config)     → None  (prints diagnostic report)
"""

import os
import time
import io

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm.auto import tqdm
from pandas_datareader import data as web

from _constants import (
    MASTER_FILE, PRICE_FILE, VOLUME_FILE, DIV_CACHE,
    SP_CONSTITUENT_URLS, NASDAQ_FTP_URL,
    DataSyncMode,
)


# ═══════════════════════════════════════════════════════════════════════
# UNIVERSE FETCHER — Scrapes S&P 1500 + US ETF universe
# ═══════════════════════════════════════════════════════════════════════

def fetch_macro_universe():
    """
    Build the full investable universe by combining:
      1. S&P 1500 constituents (S&P 500 + 400 + 600) from Wikipedia
      2. All active US-listed ETFs from Nasdaq FTP directory

    Returns:
        list[str]: Deduplicated, cleaned ticker symbols.
    """
    import requests

    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Fetching latest S&P 1500 constituents from Wikipedia...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    tickers = []
    for u in SP_CONSTITUENT_URLS:
        html = requests.get(u, headers=headers).text
        df = pd.read_html(io.StringIO(html))[0]
        tickers.extend(df['Symbol'].tolist())

    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Fetching complete US ETF universe (~3500+ symbols) via Nasdaq FTP...")
    try:
        import urllib.request as ur
        data = ur.urlopen(NASDAQ_FTP_URL).read().decode('utf-8')
        lines = data.split('\n')
        etf_tickers = []
        for line in lines[1:]:
            parts = line.split('|')
            if len(parts) > 7 and parts[5] == 'Y' and parts[7] == 'N':
                etf_tickers.append(parts[1])
        tickers.extend(etf_tickers)
        print(f"[{time.strftime('%H:%M:%S')}] [Member A] Discovered {len(etf_tickers):,} active ETF symbols.")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [Member A] WARNING: Nasdaq FTP scrape failed: {e}")

    clean_tickers = [t.replace('.', '-') for t in set(tickers)]
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Total combined macro universe: {len(clean_tickers):,} symbols.")
    return clean_tickers


# ═══════════════════════════════════════════════════════════════════════
# DATA PIPELINE — Download, cache, and preprocess asset data
# ═══════════════════════════════════════════════════════════════════════

def generate_dataset_member_a(tickers, config):
    """
    Member A data pipeline: download, cache, and preprocess asset data.

    Steps:
        1. Load cached CSV data (if any)
        2. Sync historical prices + volume (based on DataSyncMode)
        3. Sync fundamentals (.info) for each ticker
        4. Persist caches to disk
        5. Compute daily returns (price-only and DRIP)
        6. Preprocess ML features (one-hot encode categoricals)
        7. Print diagnostics

    Returns:
        (master_df, price_matrix, volume_matrix, daily_returns, drip_daily_returns)
    """
    mode = config.get("data_source_mode", DataSyncMode.INCREMENTAL_SYNC)
    chunk_size = config.get("yf_chunk_size", 300)
    cooldown = config.get("scrape_delay", 0.3)
    drip_enabled = config.get("drip_reinvest", False)

    # ── 1. Load Cached Data ──────────────────────────────────────────
    master_df = _load_csv_index(MASTER_FILE, index_col='ticker')
    price_matrix = _load_timeseries_csv(PRICE_FILE)
    volume_matrix = _load_timeseries_csv(VOLUME_FILE)

    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Booting in mode [{mode}]")

    # ── Determine what to fetch based on sync mode ──────────────────
    tickers_to_pull_info, tickers_to_pull_price = _resolve_sync_targets(
        mode, tickers, master_df, price_matrix, volume_matrix
    )

    # ── 2. Sync Historical Prices + Volume ──────────────────────────
    if tickers_to_pull_price:
        price_matrix, volume_matrix = _sync_prices(
            tickers_to_pull_price, price_matrix, volume_matrix,
            mode, config, chunk_size, cooldown
        )

    # ── 3. Sync Fundamentals ────────────────────────────────────────
    if len(tickers_to_pull_info) > 0:
        master_df = _sync_fundamentals(tickers_to_pull_info, master_df, cooldown)

    # ── 4. Persist Caches ───────────────────────────────────────────
    if mode != DataSyncMode.OFFLINE_CSV_ONLY:
        _persist_caches(master_df, price_matrix, volume_matrix)

    # ── 5. Compute Returns ──────────────────────────────────────────
    #daily_returns = price_matrix.pct_change().dropna(how='all') if not price_matrix.empty else pd.DataFrame()
    #drip_daily_returns = _compute_drip_returns(drip_enabled, price_matrix, daily_returns)
    daily_returns = (
        price_matrix.pct_change().dropna(how='all')
        if not price_matrix.empty else pd.DataFrame()
    )
    drip_daily_returns = _compute_drip_returns(
        drip_enabled, price_matrix, daily_returns
    )
    # Infaltion Data
    annual_inflation = get_annual_inflation_series(
        start=config.get("fred_start_date", "2000-01-01")
    )
    inflation_daily_returns, inflation_index = get_daily_inflation_series(
        start=config.get("fred_start_date", "2000-01-01")
    )
    inflation_daily_returns = (
        inflation_daily_returns
        .reindex(daily_returns.index)
        .ffill()
        .fillna(0.0)
    )
    inflation_index = (
        inflation_index
        .reindex(daily_returns.index)
        .ffill()
    )

    # ── 6. ML Feature Preprocessing ─────────────────────────────────
    master_df = _preprocess_ml_features(master_df, config)

    if mode != DataSyncMode.OFFLINE_CSV_ONLY:
        master_df.to_csv(MASTER_FILE)

    # ── 7. Diagnostics ──────────────────────────────────────────────
    _print_pipeline_diagnostics(master_df, volume_matrix, drip_daily_returns, config)
    print(master_df.columns[master_df.columns.duplicated()])
    dupes = master_df.columns[master_df.columns.duplicated()]
    print("Duplicate columns:", dupes.tolist())
    print(master_df.shape)
    print(len(master_df.columns))
    print(len(set(master_df.columns)))
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    #master_df[numeric_cols] = master_df[numeric_cols].fillna(0.0)
    for col in numeric_cols:
        master_df[col] = master_df[col].fillna(0.0)
    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    return master_df, price_matrix, volume_matrix, daily_returns, drip_daily_returns, annual_inflation, inflation_daily_returns, inflation_index


# ═══════════════════════════════════════════════════════════════════════
# HELPERS — Called only by generate_dataset_member_a (flat, no nesting)
# ═══════════════════════════════════════════════════════════════════════

def _load_csv_index(filepath, index_col='ticker'):
    """Load a CSV with an index column, or return an empty DataFrame."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=index_col)
    return pd.DataFrame()


def _load_timeseries_csv(filepath):
    """Load a time-series CSV (dates as index), stripping timezone if present."""
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if not df.empty and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def _resolve_sync_targets(mode, tickers, master_df, price_matrix, volume_matrix):
    """Determine which tickers need info and price fetching based on sync mode."""
    if mode == DataSyncMode.OFFLINE_CSV_ONLY:
        if master_df.empty or price_matrix.empty:
            print(f"[{time.strftime('%H:%M:%S')}] [Member A] WARNING: OFFLINE selected but CSVs missing!")
        return [], []

    if len(tickers) == 0:
        tickers = fetch_macro_universe()

    if mode == DataSyncMode.FULL_REBUILD:
        return tickers, tickers

    if mode == DataSyncMode.REFRESH_FUNDAMENTALS:
        return tickers, [t for t in tickers if t not in price_matrix.columns]

    if mode == DataSyncMode.INCREMENTAL_SYNC:
        return [t for t in tickers if t not in master_df.index], tickers

    return [], []


def _sync_prices(tickers_to_pull_price, price_matrix, volume_matrix,
                 mode, config, chunk_size, cooldown):
    """Download price & volume data for new and existing tickers."""
    new_tickers = [t for t in tickers_to_pull_price if t not in price_matrix.columns]
    exist_tickers = [t for t in tickers_to_pull_price if t in price_matrix.columns]

    # Pull newly discovered tickers from inception
    if new_tickers:
        print(f"[{time.strftime('%H:%M:%S')}] [Member A] Pulling full history for {len(new_tickers)} new tickers from {config['data_start_date']}...")
        new_pm, new_vm = _download_price_chunks(new_tickers, config["data_start_date"], chunk_size, cooldown, desc="New Tickers")
        if not new_pm.empty:
            price_matrix = new_pm if price_matrix.empty else pd.concat([price_matrix, new_pm], axis=1)
            volume_matrix = new_vm if volume_matrix.empty else pd.concat([volume_matrix, new_vm], axis=1)

    # Pull existing tickers incrementally
    if exist_tickers and mode == DataSyncMode.INCREMENTAL_SYNC:
        last_dt = price_matrix.index.max()
        if pd.notna(last_dt):
            start_dt = (last_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
            if start_dt <= today_str:
                print(f"[{time.strftime('%H:%M:%S')}] [Member A] Incrementing {len(exist_tickers)} existing tickers starting {start_dt}...")
                inc_pm, inc_vm = _download_price_chunks(exist_tickers, start_dt, chunk_size, cooldown, desc="Inc Tickers")
                if not inc_pm.empty:
                    price_matrix = pd.concat([price_matrix, inc_pm], axis=0)
                    price_matrix = price_matrix[~price_matrix.index.duplicated(keep='last')].sort_index()
                    volume_matrix = pd.concat([volume_matrix, inc_vm], axis=0)
                    volume_matrix = volume_matrix[~volume_matrix.index.duplicated(keep='last')].sort_index()

    return price_matrix, volume_matrix


def _download_price_chunks(tickers, start_date, chunk_size, cooldown, desc="Download"):
    """Download price and volume data in chunks via yfinance."""
    price_chunks = []
    vol_chunks = []
    for i in tqdm(range(0, len(tickers), chunk_size), desc=desc):
        chunk_t = tickers[i:i+chunk_size]
        chunk_d = yf.download(chunk_t, ignore_tz=True, start=start_date,
                              group_by="ticker", auto_adjust=True,
                              progress=False, threads=True)
        if not chunk_d.empty:
            if isinstance(chunk_d.columns, pd.MultiIndex):
                pm = chunk_d.xs('Close', axis=1, level=1).ffill()
                vm = chunk_d.xs('Volume', axis=1, level=1).fillna(0)
            else:
                pm = pd.DataFrame({chunk_t[0]: chunk_d['Close']}).ffill()
                vm = pd.DataFrame({chunk_t[0]: chunk_d['Volume']}).fillna(0)
            price_chunks.append(pm)
            vol_chunks.append(vm)
        time.sleep(cooldown)

    if price_chunks:
        combined_pm = pd.concat(price_chunks, axis=1).dropna(axis=1, how='all')
        combined_vm = pd.concat(vol_chunks, axis=1).dropna(axis=1, how='all')
        if combined_pm.index.tz is not None:
            combined_pm.index = combined_pm.index.tz_localize(None)
            combined_vm.index = combined_vm.index.tz_localize(None)
        return combined_pm, combined_vm

    return pd.DataFrame(), pd.DataFrame()


def _sync_fundamentals(tickers_to_pull_info, master_df, cooldown):
    """Fetch yfinance .info fundamentals for each ticker."""
    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Fetching Info/Fundamentals for {len(tickers_to_pull_info)} tickers...")
    company_data = []
    for ticker in tqdm(tickers_to_pull_info, desc="Fundamentals"):
        if ticker == "^GSPC":
            company_data.append({"ticker": ticker, "sector": "Index/Bond", "industry": "Index",
                                 "quoteType": "ETF", "top_holdings": "['" + ticker + "']"})
            continue
        try:
            t_obj = yf.Ticker(ticker)
            info = t_obj.info
            time.sleep(cooldown)

            quote_type = info.get('quoteType', '')
            holdings = [ticker]
            if quote_type == 'ETF':
                funds = getattr(t_obj, 'funds_data', None)
                if funds and hasattr(funds, 'top_holdings') and funds.top_holdings is not None:
                    holdings = list(funds.top_holdings.index)

            company_record = info.copy()
            company_record['ticker'] = ticker
            company_record['quoteType'] = quote_type
            company_record['top_holdings'] = str(holdings)

            if 'sector' not in company_record:
                company_record['sector'] = 'Unknown'
            if 'industry' not in company_record:
                company_record['industry'] = 'Unknown'

            company_data.append(company_record)
        except Exception:
            pass

    if company_data:
        new_master_df = pd.DataFrame(company_data).set_index('ticker')
        if master_df.empty:
            master_df = new_master_df
        else:
            overlap = set(new_master_df.index).intersection(set(master_df.index))
            if overlap:
                print(f"[{time.strftime('%H:%M:%S')}] [Member A] WARNING: {len(overlap)} fundamental conflicts. Last write wins applied.")
            combined = pd.concat([master_df, new_master_df])
            master_df = combined[~combined.index.duplicated(keep='last')]

    return master_df


def _persist_caches(master_df, price_matrix, volume_matrix):
    """Write dataframes to CSV caches."""
    if not master_df.empty:
        master_df.to_csv(MASTER_FILE)
    if not price_matrix.empty:
        price_matrix.to_csv(PRICE_FILE)
    if not volume_matrix.empty:
        volume_matrix.to_csv(VOLUME_FILE)


def _compute_drip_returns(drip_enabled, price_matrix, daily_returns):
    """Compute DRIP-adjusted returns if enabled and dividend cache exists."""
    if not drip_enabled or price_matrix.empty:
        return None

    print(f"[{time.strftime('%H:%M:%S')}] [Member A] Computing dividend-adjusted (DRIP) returns...")
    try:
        div_df = pd.read_csv(DIV_CACHE, index_col=0, parse_dates=True) if os.path.exists(DIV_CACHE) else pd.DataFrame()
        if not div_df.empty:
            div_yield = div_df.reindex(index=price_matrix.index, columns=price_matrix.columns).fillna(0)
            div_yield = div_yield / price_matrix.shift(1)
            div_yield = div_yield.fillna(0).clip(0, 0.5)  # cap at 50% daily yield (data error guard)
            drip_daily_returns = daily_returns + div_yield
            print(f"[{time.strftime('%H:%M:%S')}] [Member A] DRIP returns computed for {len(drip_daily_returns.columns)} tickers.")
            return drip_daily_returns
        else:
            print(f"[{time.strftime('%H:%M:%S')}] [Member A] No dividend cache found. DRIP disabled (using price-only returns).")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] [Member A] DRIP computation failed: {e}. Using price-only returns.")

    return None


def _preprocess_ml_features(master_df, config):
    """One-hot encode categoricals and fill numeric NaNs for configured ML features."""
    print(f"\n[{time.strftime('%H:%M:%S')}] [Member A] Preprocessing Configured ML Features...")
    feature_config = config.get("ml_training_features", [])

    for f in feature_config:
        if f in ['hist_momentum', 'hist_volatility', 'hist_volume']:
            continue  # computed per-year in training loop

        if f in master_df.columns:
            if (master_df[f].dtype == 'object'
                    or pd.api.types.is_categorical_dtype(master_df[f])
                    or pd.api.types.is_string_dtype(master_df[f])):
                print(f"   -> Extracting Categorical Matrix: '{f}'")
                dummies = pd.get_dummies(master_df[f], prefix=f, dtype=float)
                # remove any duplicate columns
                dummies = dummies.loc[:, ~dummies.columns.isin(master_df.columns)]
                if len(dummies.columns) > 0:
                    master_df = pd.concat([master_df, dummies], axis=1)
                master_df.drop(columns=[f], inplace=True)
            else:
                print(f"   -> Processing Numeric Vector: '{f}'")
                master_df[f] = pd.to_numeric(master_df[f], errors='coerce').fillna(0.0)
        else:
            print(f"   -> WARNING: Requested config feature '{f}' not found in master_df.")
    master_df = master_df.loc[:, ~master_df.columns.duplicated()]
    return master_df


def _print_pipeline_diagnostics(master_df, volume_matrix, drip_daily_returns, config):
    """Print summary of the data pipeline output."""
    print("\n" + "=" * 50)
    print("=== DATAFRAME DIAGNOSTICS & SYSTEM MAPPING ===")
    print("=" * 50)
    print(f"Total Equities Tracked: {len(master_df)}")
    print(f"Total Columns Processed per Asset: {len(master_df.columns)}\n")
    print(f"All Features Extracted in Master DF:\n{list(master_df.columns)}\n")

    print("[PIPELINE DATA FLOW MAPPING]")
    print(f"-> Selected for ML Tensor Construction: {config.get('ml_training_features', [])}")
    print("-> Passed to Member B (Scoring Layer 2): embeddings, asset_predictions, daily_returns")
    print("-> Passed to Member C (Simulation Layer 3): daily_returns, price_matrix")
    print("-> Ignored by Logic / For Display Only: ALL OTHER COLUMNS (Data Poisoning Shield Active)")
    print(f"-> Volume Matrix: {'Loaded' if not volume_matrix.empty else 'Empty'} ({len(volume_matrix.columns)} tickers)")
    print(f"-> DRIP Returns: {'Enabled' if drip_daily_returns is not None else 'Disabled (price-only)'}")
    print("==============================================\n")


# ═══════════════════════════════════════════════════════════════════════
# DATA DICTIONARY & VALIDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════
# Classifies every column by data-poisoning risk for backtesting embeddings.
#
# Safety Classes:
#   SAFE_STATIC    — Time-invariant company identity (sector, country).
#   SLOW_CHANGING  — Changes gradually over years (employees, governance).
#   POINT_IN_TIME  — Current market/financial snapshot (price, PE, marketCap).
#                    HIGH RISK: using today's value to predict historical returns = data leakage.
#   DERIVED        — Computed per-year inside the training loop (hist_momentum, hist_volatility).
#   METADATA       — API/system plumbing (timestamps, IDs, booleans). Not useful for ML.
#   TEXT           — Free-text fields requiring NLP. Cannot be one-hot encoded.
#   IDENTIFIER     — Names, symbols, phone numbers, URLs. Not features.

COLUMN_SAFETY_REGISTRY = {
    # —— Company Identity (SAFE_STATIC) ——
    "sector":       ("GICS sector classification (e.g. Technology, Healthcare). ~12 unique values.", "SAFE_STATIC"),
    "sectorKey":    ("Sector key identifier (lowercase slug of sector).", "SAFE_STATIC"),
    "sectorDisp":   ("Sector display name (same as sector, alternate key).", "SAFE_STATIC"),
    "industry":     ("GICS industry sub-classification (e.g. Semiconductors). ~150 unique values.", "SAFE_STATIC"),
    "industryKey":  ("Industry key identifier (lowercase slug).", "SAFE_STATIC"),
    "industryDisp": ("Industry display name (alternate key).", "SAFE_STATIC"),
    "industrySymbol": ("Industry symbol code.", "SAFE_STATIC"),
    "quoteType":    ("Security type: EQUITY, ETF, MUTUALFUND. Defines asset class.", "SAFE_STATIC"),
    "state":        ("US state or province of company headquarters. ~50 unique values.", "SAFE_STATIC"),
    "country":      ("Country of domicile. Mostly 'United States'. ~5 unique values.", "SAFE_STATIC"),
    "city":         ("City of headquarters. HIGH cardinality (~800+ unique). One-hot will explode tensor width.", "SAFE_STATIC"),
    "address1":     ("Street address of HQ. EXTREME cardinality. Not useful for embeddings.", "IDENTIFIER"),
    "address2":     ("Street address line 2.", "IDENTIFIER"),
    "zip":          ("Postal/zip code. HIGH cardinality (~1000+ unique).", "SAFE_STATIC"),
    "exchange":     ("Exchange code (NMS, NYQ, etc.). ~5 unique values.", "SAFE_STATIC"),
    "fullExchangeName": ("Full exchange name (NASDAQ, NYSE, etc.).", "SAFE_STATIC"),
    "market":       ("Market identifier (e.g. us_market). Almost always the same.", "SAFE_STATIC"),
    "lastSplitFactor": ("Last stock split ratio (e.g. '4:1'). Historical event.", "SAFE_STATIC"),

    # —— Slowly Changing Fundamentals (SLOW_CHANGING) ——
    "fullTimeEmployees":    ("Number of full-time employees. Changes annually. Today's snapshot.", "SLOW_CHANGING"),
    "auditRisk":            ("ISS audit committee risk score (1-10, 10=highest). Updated ~annually.", "SLOW_CHANGING"),
    "boardRisk":            ("ISS board composition risk score (1-10). Updated ~annually.", "SLOW_CHANGING"),
    "compensationRisk":     ("ISS executive compensation risk score (1-10). Updated ~annually.", "SLOW_CHANGING"),
    "shareHolderRightsRisk": ("ISS shareholder rights risk score (1-10). Updated ~annually.", "SLOW_CHANGING"),
    "overallRisk":          ("ISS composite governance risk score (1-10). Updated ~annually.", "SLOW_CHANGING"),

    # —— Price-Derived & Market Snapshot (POINT_IN_TIME) ——
    "marketCap":                ("Current market cap (price x shares). CHANGES DAILY with stock price.", "POINT_IN_TIME"),
    "nonDilutedMarketCap":      ("Market cap excluding dilutive securities. Price-derived.", "POINT_IN_TIME"),
    "enterpriseValue":          ("Market cap + debt - cash. Price-derived.", "POINT_IN_TIME"),
    "currentPrice":             ("Current trading price.", "POINT_IN_TIME"),
    "previousClose":            ("Yesterday's closing price.", "POINT_IN_TIME"),
    "open":                     ("Today's opening price.", "POINT_IN_TIME"),
    "dayLow":                   ("Today's intraday low.", "POINT_IN_TIME"),
    "dayHigh":                  ("Today's intraday high.", "POINT_IN_TIME"),
    "regularMarketPreviousClose": ("Previous close (regular session). Duplicate of previousClose.", "POINT_IN_TIME"),
    "regularMarketOpen":        ("Opening price (regular session).", "POINT_IN_TIME"),
    "regularMarketDayLow":      ("Day low (regular session).", "POINT_IN_TIME"),
    "regularMarketDayHigh":     ("Day high (regular session).", "POINT_IN_TIME"),
    "regularMarketPrice":       ("Current regular market price.", "POINT_IN_TIME"),
    "regularMarketVolume":      ("Today's trading volume.", "POINT_IN_TIME"),
    "regularMarketChange":      ("Today's absolute price change.", "POINT_IN_TIME"),
    "regularMarketChangePercent": ("Today's price change %.", "POINT_IN_TIME"),
    "regularMarketDayRange":    ("Today's price range as string.", "POINT_IN_TIME"),
    "postMarketPrice":          ("After-hours trading price.", "POINT_IN_TIME"),
    "postMarketChange":         ("After-hours price change.", "POINT_IN_TIME"),
    "postMarketChangePercent":  ("After-hours price change %.", "POINT_IN_TIME"),
    "bid":                      ("Current best bid price.", "POINT_IN_TIME"),
    "ask":                      ("Current best ask price.", "POINT_IN_TIME"),
    "bidSize":                  ("Lot size at best bid.", "POINT_IN_TIME"),
    "askSize":                  ("Lot size at best ask.", "POINT_IN_TIME"),
    "volume":                   ("Today's trading volume.", "POINT_IN_TIME"),

    # —— Valuation Ratios (POINT_IN_TIME — numerator or denominator is current price) ——
    "beta":                         ("5-year monthly beta vs S&P 500. Computed from recent 5yr window.", "POINT_IN_TIME"),
    "trailingPE":                   ("Price / trailing 12-mo EPS. Numerator = current price.", "POINT_IN_TIME"),
    "forwardPE":                    ("Price / forward EPS estimate. Numerator = current price.", "POINT_IN_TIME"),
    "priceToBook":                  ("Price / book value per share. Numerator = current price.", "POINT_IN_TIME"),
    "priceToSalesTrailing12Months": ("Price / trailing 12-mo revenue/share. Price-derived.", "POINT_IN_TIME"),
    "enterpriseToRevenue":          ("EV / revenue. EV is price-derived.", "POINT_IN_TIME"),
    "enterpriseToEbitda":           ("EV / EBITDA. EV is price-derived.", "POINT_IN_TIME"),
    "trailingPegRatio":             ("PEG ratio (PE / growth). PE is price-derived.", "POINT_IN_TIME"),
    "priceEpsCurrentYear":          ("Price / current-year EPS estimate.", "POINT_IN_TIME"),

    # —— Dividend Metrics (POINT_IN_TIME) ——
    "dividendRate":                 ("Current annualized dividend ($/share).", "POINT_IN_TIME"),
    "dividendYield":                ("Annual dividend / current price. Price-derived.", "POINT_IN_TIME"),
    "payoutRatio":                  ("Dividends paid / net income. Last fiscal year snapshot.", "POINT_IN_TIME"),
    "fiveYearAvgDividendYield":     ("5-year average dividend yield.", "POINT_IN_TIME"),
    "trailingAnnualDividendRate":   ("Trailing 12-mo dividend ($/share).", "POINT_IN_TIME"),
    "trailingAnnualDividendYield":  ("Trailing dividend / current price. Price-derived.", "POINT_IN_TIME"),
    "lastDividendValue":            ("Most recent dividend payment amount.", "POINT_IN_TIME"),

    # —— Volume & Liquidity (POINT_IN_TIME) ——
    "averageVolume":            ("Average daily volume (3-month lookback).", "POINT_IN_TIME"),
    "averageVolume10days":      ("Average daily volume (10-day lookback).", "POINT_IN_TIME"),
    "averageDailyVolume10Day":  ("Average daily volume 10-day (duplicate key).", "POINT_IN_TIME"),
    "averageDailyVolume3Month": ("Average daily volume 3-month (duplicate key).", "POINT_IN_TIME"),

    # —— Technical Price Levels (POINT_IN_TIME) ——
    "fiftyDayAverage":                  ("50-day simple moving average price.", "POINT_IN_TIME"),
    "twoHundredDayAverage":             ("200-day simple moving average price.", "POINT_IN_TIME"),
    "fiftyDayAverageChange":            ("Absolute change from 50-day MA.", "POINT_IN_TIME"),
    "fiftyDayAverageChangePercent":     ("% change from 50-day MA.", "POINT_IN_TIME"),
    "twoHundredDayAverageChange":       ("Absolute change from 200-day MA.", "POINT_IN_TIME"),
    "twoHundredDayAverageChangePercent":("% change from 200-day MA.", "POINT_IN_TIME"),
    "fiftyTwoWeekLow":                  ("52-week low price.", "POINT_IN_TIME"),
    "fiftyTwoWeekHigh":                 ("52-week high price.", "POINT_IN_TIME"),
    "fiftyTwoWeekLowChange":            ("Price change from 52-week low.", "POINT_IN_TIME"),
    "fiftyTwoWeekLowChangePercent":     ("%% change from 52-week low.", "POINT_IN_TIME"),
    "fiftyTwoWeekHighChange":           ("Price change from 52-week high.", "POINT_IN_TIME"),
    "fiftyTwoWeekHighChangePercent":    ("% change from 52-week high.", "POINT_IN_TIME"),
    "fiftyTwoWeekChangePercent":        ("52-week price return %.", "POINT_IN_TIME"),
    "fiftyTwoWeekRange":                ("52-week price range as string.", "POINT_IN_TIME"),
    "52WeekChange":                     ("52-week price change ratio.", "POINT_IN_TIME"),
    "SandP52WeekChange":                ("S&P 500 52-week change (benchmark comparison).", "POINT_IN_TIME"),
    "allTimeHigh":                      ("All-time high price.", "POINT_IN_TIME"),
    "allTimeLow":                       ("All-time low price.", "POINT_IN_TIME"),

    # —— Income Statement & Margins (POINT_IN_TIME — last reported quarter/year) ——
    "profitMargins":        ("Net profit margin. Last reported quarter snapshot.", "POINT_IN_TIME"),
    "grossMargins":         ("Gross margin. Last reported quarter snapshot.", "POINT_IN_TIME"),
    "operatingMargins":     ("Operating margin. Last reported quarter snapshot.", "POINT_IN_TIME"),
    "ebitdaMargins":        ("EBITDA margin. Last reported quarter snapshot.", "POINT_IN_TIME"),
    "returnOnAssets":       ("Return on assets. Last reported snapshot.", "POINT_IN_TIME"),
    "returnOnEquity":       ("Return on equity. Last reported snapshot.", "POINT_IN_TIME"),
    "revenueGrowth":        ("Quarter-over-quarter revenue growth rate.", "POINT_IN_TIME"),
    "earningsGrowth":       ("Quarter-over-quarter earnings growth rate.", "POINT_IN_TIME"),
    "earningsQuarterlyGrowth": ("Quarterly earnings growth (duplicate key).", "POINT_IN_TIME"),
    "totalRevenue":         ("Total revenue in absolute $ (last reported).", "POINT_IN_TIME"),
    "revenuePerShare":      ("Revenue per share (last reported).", "POINT_IN_TIME"),
    "grossProfits":         ("Gross profit in absolute $ (last reported).", "POINT_IN_TIME"),
    "ebitda":               ("EBITDA in absolute $ (last reported).", "POINT_IN_TIME"),
    "netIncomeToCommon":    ("Net income to common shareholders in $ (last reported).", "POINT_IN_TIME"),

    # —— Balance Sheet (POINT_IN_TIME) ——
    "totalCash":            ("Total cash on hand in $.", "POINT_IN_TIME"),
    "totalCashPerShare":    ("Cash per share.", "POINT_IN_TIME"),
    "totalDebt":            ("Total debt in $.", "POINT_IN_TIME"),
    "debtToEquity":         ("Debt-to-equity ratio.", "POINT_IN_TIME"),
    "currentRatio":         ("Current assets / current liabilities.", "POINT_IN_TIME"),
    "quickRatio":           ("(Current assets - inventory) / current liabilities.", "POINT_IN_TIME"),
    "bookValue":            ("Book value per share.", "POINT_IN_TIME"),
    "freeCashflow":         ("Free cash flow in $.", "POINT_IN_TIME"),
    "operatingCashflow":    ("Operating cash flow in $.", "POINT_IN_TIME"),

    # —— EPS (POINT_IN_TIME) ——
    "trailingEps":              ("Trailing 12-month EPS.", "POINT_IN_TIME"),
    "forwardEps":               ("Forward EPS analyst estimate.", "POINT_IN_TIME"),
    "epsTrailingTwelveMonths":  ("EPS trailing 12 months (duplicate key).", "POINT_IN_TIME"),
    "epsForward":               ("Forward EPS (duplicate key).", "POINT_IN_TIME"),
    "epsCurrentYear":           ("Current fiscal year EPS estimate.", "POINT_IN_TIME"),

    # —— Ownership & Short Interest (POINT_IN_TIME) ——
    "floatShares":              ("Shares available for public trading.", "POINT_IN_TIME"),
    "sharesOutstanding":        ("Total shares outstanding.", "POINT_IN_TIME"),
    "impliedSharesOutstanding": ("Implied shares outstanding.", "POINT_IN_TIME"),
    "sharesShort":              ("Shares currently sold short.", "POINT_IN_TIME"),
    "sharesShortPriorMonth":    ("Shares short prior month.", "POINT_IN_TIME"),
    "sharesPercentSharesOut":   ("Short shares as % of outstanding.", "POINT_IN_TIME"),
    "shortRatio":               ("Days to cover short positions.", "POINT_IN_TIME"),
    "shortPercentOfFloat":      ("Short interest as % of float.", "POINT_IN_TIME"),
    "heldPercentInsiders":      ("% of shares held by insiders.", "POINT_IN_TIME"),
    "heldPercentInstitutions":  ("% of shares held by institutions.", "POINT_IN_TIME"),

    # —— Analyst Estimates (POINT_IN_TIME) ——
    "targetHighPrice":          ("Analyst highest price target.", "POINT_IN_TIME"),
    "targetLowPrice":           ("Analyst lowest price target.", "POINT_IN_TIME"),
    "targetMeanPrice":          ("Analyst mean price target.", "POINT_IN_TIME"),
    "targetMedianPrice":        ("Analyst median price target.", "POINT_IN_TIME"),
    "recommendationMean":       ("Analyst consensus (1=Strong Buy ... 5=Sell).", "POINT_IN_TIME"),
    "recommendationKey":        ("Analyst recommendation text (buy/hold/sell).", "POINT_IN_TIME"),
    "numberOfAnalystOpinions":  ("Number of analysts covering this stock.", "POINT_IN_TIME"),
    "averageAnalystRating":     ("Average analyst rating as string.", "POINT_IN_TIME"),

    # —— Pipeline-Derived (DERIVED — computed in training loop, time-aligned) ——
    "hist_momentum":        ("Annual return for the backtest year. Computed per-year, NO leakage.", "DERIVED"),
    "hist_volatility":      ("Annualized daily vol for the backtest year. Computed per-year, NO leakage.", "DERIVED"),
    "current_volatility":   ("Annualized vol from full daily returns. Used by Member B, not embedding.", "DERIVED"),
    "top_holdings":         ("ETF top holdings list (pipeline-generated string).", "METADATA"),

    # —— Identifiers (IDENTIFIER — not useful as ML features) ——
    "phone":        ("Company phone number.", "IDENTIFIER"),
    "fax":          ("Company fax number.", "IDENTIFIER"),
    "website":      ("Company website URL.", "IDENTIFIER"),
    "irWebsite":    ("Investor relations website URL.", "IDENTIFIER"),
    "shortName":    ("Short company name.", "IDENTIFIER"),
    "longName":     ("Full legal company name.", "IDENTIFIER"),
    "displayName":  ("Display name variant.", "IDENTIFIER"),
    "symbol":       ("Ticker symbol.", "IDENTIFIER"),
    "prevName":     ("Previous company name.", "IDENTIFIER"),

    # —— Free Text (TEXT — requires NLP, cannot one-hot encode) ——
    "longBusinessSummary":  ("Full company business description paragraph.", "TEXT"),
    "companyOfficers":      ("List of company officers (structured/nested).", "TEXT"),
    "executiveTeam":        ("Executive team information.", "TEXT"),

    # —— API/System Metadata (METADATA — no ML signal) ——
    "maxAge":                   ("Cache age in seconds (yfinance internal).", "METADATA"),
    "priceHint":                ("Decimal precision for price display.", "METADATA"),
    "currency":                 ("Trading currency (mostly USD).", "METADATA"),
    "financialCurrency":        ("Reporting currency.", "METADATA"),
    "language":                 ("Language code.", "METADATA"),
    "region":                   ("Region code.", "METADATA"),
    "typeDisp":                 ("Quote type display name.", "METADATA"),
    "quoteSourceName":          ("Data source name.", "METADATA"),
    "tradeable":                ("Whether tradeable on platform (boolean).", "METADATA"),
    "triggerable":              ("Whether alerts can be set (boolean).", "METADATA"),
    "cryptoTradeable":          ("Whether crypto trading available.", "METADATA"),
    "hasPrePostMarketData":     ("Whether pre/post market data exists.", "METADATA"),
    "customPriceAlertConfidence":("Alert confidence level.", "METADATA"),
    "corporateActions":         ("Corporate actions data.", "METADATA"),
    "messageBoardId":           ("Yahoo message board ID.", "METADATA"),
    "exchangeTimezoneName":     ("Exchange timezone name.", "METADATA"),
    "exchangeTimezoneShortName":("Exchange timezone abbreviation.", "METADATA"),
    "gmtOffSetMilliseconds":    ("GMT offset in milliseconds.", "METADATA"),
    "esgPopulated":             ("Whether ESG data is populated.", "METADATA"),
    "sourceInterval":           ("Data source polling interval.", "METADATA"),
    "exchangeDataDelayedBy":    ("Data delay in seconds.", "METADATA"),
    "firstTradeDateMilliseconds":("First trade date (epoch ms).", "METADATA"),
    "marketState":              ("Current market state (REGULAR, POST, PRE).", "METADATA"),
    "regularMarketTime":        ("Timestamp of last regular trade.", "METADATA"),
    "postMarketTime":           ("Timestamp of post-market data.", "METADATA"),
    "governanceEpochDate":      ("Date of governance assessment (epoch).", "METADATA"),
    "compensationAsOfEpochDate":("Date of compensation data (epoch).", "METADATA"),
    "exDividendDate":           ("Next ex-dividend date (epoch).", "METADATA"),
    "dividendDate":             ("Next dividend payment date (epoch).", "METADATA"),
    "lastDividendDate":         ("Date of most recent dividend (epoch).", "METADATA"),
    "lastSplitDate":            ("Date of last stock split (epoch).", "METADATA"),
    "sharesShortPreviousMonthDate": ("Date of prior month short data.", "METADATA"),
    "dateShortInterest":        ("Date of short interest data.", "METADATA"),
    "lastFiscalYearEnd":        ("Last fiscal year end (epoch).", "METADATA"),
    "nextFiscalYearEnd":        ("Next fiscal year end (epoch).", "METADATA"),
    "mostRecentQuarter":        ("Most recent quarter end (epoch).", "METADATA"),
    "earningsTimestamp":        ("Next earnings date (epoch).", "METADATA"),
    "earningsTimestampStart":   ("Earnings window start (epoch).", "METADATA"),
    "earningsTimestampEnd":     ("Earnings window end (epoch).", "METADATA"),
    "earningsCallTimestampStart":("Earnings call start (epoch).", "METADATA"),
    "earningsCallTimestampEnd": ("Earnings call end (epoch).", "METADATA"),
    "isEarningsDateEstimate":   ("Whether earnings date is estimated.", "METADATA"),
    "nameChangeDate":           ("Date of company name change.", "METADATA"),
    "ipoExpectedDate":          ("Expected IPO date.", "METADATA"),
    "prevExchange":             ("Previous exchange listing.", "METADATA"),
    "exchangeTransferDate":     ("Date of exchange transfer.", "METADATA"),
}

SAFETY_ICONS = {
    "SAFE_STATIC":   "\U0001F7E2",   # green circle
    "SLOW_CHANGING": "\U0001F7E1",   # yellow circle
    "POINT_IN_TIME": "\U0001F534",   # red circle
    "DERIVED":       "\u2699",       # gear
    "METADATA":      "\u274C",       # cross mark
    "TEXT":          "\U0001F4DD",    # memo
    "IDENTIFIER":   "\U0001F3F7\uFE0F",  # label
    "UNKNOWN":       "\u2753",       # question mark
}


def _resolve_column_info(col_name):
    """Look up a column in the registry, checking for one-hot prefixes if needed."""
    if col_name in COLUMN_SAFETY_REGISTRY:
        return COLUMN_SAFETY_REGISTRY[col_name]
    # Check if this is a one-hot encoded column (e.g. 'sector_Technology')
    for prefix in COLUMN_SAFETY_REGISTRY:
        if col_name.startswith(prefix + "_"):
            parent_desc, parent_safety = COLUMN_SAFETY_REGISTRY[prefix]
            suffix = col_name[len(prefix)+1:]
            return (f"One-hot from '{prefix}' = '{suffix}'", parent_safety)
    return ("(Not in registry — review manually)", "UNKNOWN")


def run_data_diagnostics(master_df, config):
    """Print comprehensive data dictionary, fill-rate analytics, and validate ML feature config."""
    feature_config = config.get("ml_training_features", [])

    # —— Section 1: Full Column Inventory ——
    print("\n" + "=" * 100)
    print("  DATA DICTIONARY — ALL COLUMNS IN master_df")
    print("  Total Columns:", len(master_df.columns), " | Total Assets:", len(master_df))
    print("=" * 100)
    print(f"{'#':<5} {'Column':<42} {'Dtype':<10} {'Non-Null':<9} {'Fill%':<7} {'Safety':<15} Description")
    print("-" * 150)

    safety_counts = {}
    for i, col in enumerate(master_df.columns):
        dtype_str = str(master_df[col].dtype)[:8]
        non_null = int(master_df[col].notna().sum())
        fill_pct = non_null / len(master_df) * 100 if len(master_df) > 0 else 0
        desc, safety = _resolve_column_info(col)
        icon = SAFETY_ICONS.get(safety, "\u2753")
        safety_counts[safety] = safety_counts.get(safety, 0) + 1

        in_model = " \u25C6 IN MODEL" if col in feature_config else ""
        fill_bar = "\u2588" * int(fill_pct // 10) + "\u2591" * (10 - int(fill_pct // 10))
        print(f"{i:<5} {col:<42} {dtype_str:<10} {non_null:<9} {fill_bar} {fill_pct:>5.1f}%  {icon} {safety:<13} {desc}{in_model}")

    # —— Section 2: Safety Class Summary ——
    print("\n" + "=" * 100)
    print("  SAFETY CLASS DISTRIBUTION")
    print("=" * 100)
    for cls in ["SAFE_STATIC", "SLOW_CHANGING", "POINT_IN_TIME", "DERIVED", "METADATA", "TEXT", "IDENTIFIER", "UNKNOWN"]:
        count = safety_counts.get(cls, 0)
        if count > 0:
            print(f"  {SAFETY_ICONS.get(cls, '')} {cls:<16} {count:>4} columns")

    # —— Section 3: ML Feature Validation ——
    print("\n" + "=" * 100)
    print("  ML FEATURE VALIDATION — Checking config['ml_training_features']")
    print("=" * 100)

    poisoning_warnings = []
    missing_warnings = []

    for f in feature_config:
        desc, safety = _resolve_column_info(f)
        icon = SAFETY_ICONS.get(safety, "\u2753")

        if safety == "DERIVED":
            print(f"  \u2705 '{f}' — {icon} DERIVED: Computed per-year in training loop. No data leakage.")
            continue

        if f in master_df.columns:
            fill = master_df[f].notna().sum() / len(master_df) * 100 if len(master_df) > 0 else 0
            uniq = master_df[f].nunique()
            is_cat = master_df[f].dtype == 'object' or pd.api.types.is_categorical_dtype(master_df[f])
            type_info = f"Categorical ({uniq} unique -> {uniq} one-hot cols)" if is_cat else "Numeric"

            if safety == "POINT_IN_TIME":
                print(f"  \u26A0\uFE0F  '{f}' — {icon} POINT_IN_TIME | Fill: {fill:.1f}% | {type_info}")
                print(f"       \u2514\u2500 WARNING: Uses TODAY's value for all backtest years -> data leakage risk!")
                poisoning_warnings.append(f)
            elif safety == "SAFE_STATIC":
                print(f"  \u2705 '{f}' — {icon} SAFE_STATIC | Fill: {fill:.1f}% | {type_info}")
            elif safety == "SLOW_CHANGING":
                print(f"  \U0001F7E1 '{f}' — {icon} SLOW_CHANGING | Fill: {fill:.1f}% | {type_info}")
                print(f"       \u2514\u2500 Note: Today's snapshot used for all years. Low risk but not historically accurate.")
            elif safety in ["METADATA", "IDENTIFIER", "TEXT"]:
                print(f"  \u274C '{f}' — {icon} {safety} | This column type is not suitable for ML training.")
            else:
                print(f"  \u2753 '{f}' — {icon} {safety} | Fill: {fill:.1f}% | {type_info} — Review manually.")
        else:
            # Check if it was already one-hot expanded (e.g. 'sector' -> 'sector_Technology', ...)
            dummy_cols = [c for c in master_df.columns if c.startswith(f + "_")]
            if dummy_cols:
                print(f"  \u2705 '{f}' — {icon} {safety} | Already one-hot expanded -> {len(dummy_cols)} columns")
            else:
                print(f"  \u274C '{f}' — NOT FOUND in master_df and no one-hot expansion detected!")
                missing_warnings.append(f)

    # —— Section 3.5: Predicted Input Dimensionality ——
    print("\n" + "=" * 100)
    print("  PREDICTED EMBEDDING INPUT VECTOR DIMENSIONALITY")
    print("=" * 100)
    print(f"  {'Feature':<35} {'Type':<15} {'Dims':>6}   Details")
    print("  " + "-" * 95)

    _total_pred = 0
    _horizons = len(config.get('ml_target_horizons', [1,3,5,10,15]))
    for f in feature_config:
        if f in ['hist_momentum', 'hist_volatility']:
            print(f"  {f:<35} {'DERIVED':<15} {_horizons:>6}   Multi-horizon trailing metrics")
            _total_pred += _horizons
        elif f in master_df.columns and pd.api.types.is_numeric_dtype(master_df[f]):
            print(f"  {f:<35} {'Numeric':<15} {1:>6}")
            _total_pred += 1
        else:
            _dc = [c for c in master_df.columns if c.startswith(f + '_')]
            if _dc:
                _sample = ', '.join(_dc[:3])
                _more = f" ... +{len(_dc)-3} more" if len(_dc) > 3 else ""
                print(f"  {f:<35} {'One-Hot':<15} {len(_dc):>6}   ({_sample}{_more})")
                _total_pred += len(_dc)
            elif f in master_df.columns:
                print(f"  {f:<35} {'Numeric':<15} {1:>6}")
                _total_pred += 1
            else:
                print(f"  {f:<35} {'NOT FOUND':<15} {'--':>6}   Will be skipped by training loop")

    print("  " + "-" * 95)
    print(f"  {'TOTAL PREDICTED INPUT DIMS':<35} {'':<15} {_total_pred:>6}")

    _hl = config.get('ml_hidden_layers', [32])
    _ed = config.get('ml_embedding_dim', 8)
    _oh = len(config.get('ml_target_horizons', [])) * 2

    print(f"\n  Network Architecture Preview:")
    _enc_p = [str(_total_pred)] + [f"{h}->ReLU" for h in _hl] + [f"{_ed}(embed)"]
    _dec_p = [f"{_ed}(embed)"] + [f"{h}->ReLU" for h in reversed(_hl)] + [f"{_oh}(output)"]
    print(f"  Encoder: {' -> '.join(_enc_p)}")
    print(f"  Decoder: {' -> '.join(_dec_p)}")
    _cr = _total_pred // _ed if _ed else 0
    print(f"  Compression Ratio: {_total_pred} -> {_ed} ({_cr}:1)")

    if _total_pred > 0 and _hl and _total_pred > _hl[0] * 3:
        print(f"\n  WARNING: Input dims ({_total_pred}) significantly larger than first hidden layer ({_hl[0]}).")
        print(f"      Consider increasing ml_hidden_layers[0] for better gradient flow.")

    # —— Section 4: Fill Rate Summary ——
    print("\n" + "=" * 100)
    print("  DATA QUALITY — FILL RATE ANALYTICS")
    print("=" * 100)

    fill_rates = master_df.notna().mean() * 100

    print(f"  Columns with 100% fill:   {(fill_rates == 100).sum():>4}")
    print(f"  Columns with 80-99% fill: {((fill_rates >= 80) & (fill_rates < 100)).sum():>4}")
    print(f"  Columns with 50-79% fill: {((fill_rates >= 50) & (fill_rates < 80)).sum():>4}")
    print(f"  Columns with 20-49% fill: {((fill_rates >= 20) & (fill_rates < 50)).sum():>4}")
    print(f"  Columns with  1-19% fill: {((fill_rates > 0) & (fill_rates < 20)).sum():>4}")
    print(f"  Columns with    0% fill:  {(fill_rates == 0).sum():>4}")

    empty_cols = fill_rates[fill_rates == 0].index.tolist()
    if empty_cols:
        print(f"\n  \u26A0\uFE0F  COMPLETELY EMPTY COLUMNS ({len(empty_cols)} total):")
        for c in empty_cols[:25]:
            print(f"      \u2022 {c}")
        if len(empty_cols) > 25:
            print(f"      ... and {len(empty_cols) - 25} more")

    sparse_cols = fill_rates[(fill_rates > 0) & (fill_rates < 20)].sort_values()
    if len(sparse_cols) > 0:
        print(f"\n  \u26A0\uFE0F  VERY SPARSE COLUMNS (<20% fill, {len(sparse_cols)} total):")
        for c, pct in sparse_cols.head(15).items():
            print(f"      \u2022 {c}: {pct:.1f}%")
        if len(sparse_cols) > 15:
            print(f"      ... and {len(sparse_cols) - 15} more")

    # —— Section 5: Final Verdict ——
    print("\n" + "=" * 100)
    print("  VALIDATION VERDICT")
    print("=" * 100)

    if poisoning_warnings:
        print(f"  \U0001F534 DATA POISONING RISK: {len(poisoning_warnings)} feature(s) are POINT_IN_TIME:")
        for f in poisoning_warnings:
            print(f"     -> '{f}' — today's value will be used for ALL historical backtest years")
        print(f"     Consider removing these or accepting the leakage trade-off.")
    else:
        print(f"  \u2705 No POINT_IN_TIME features in config. Clean backtest.")

    if missing_warnings:
        print(f"  \u274C MISSING FEATURES: {len(missing_warnings)} feature(s) not found in master_df:")
        for f in missing_warnings:
            print(f"     -> '{f}'")
    else:
        print(f"  \u2705 All configured features exist in master_df.")

    print("=" * 100 + "\n")

def load_cpi(start="2000-01-01"):
    """Pull CPIAUCSL dataset from FRED and preprocess"""
    cpi = web.DataReader("CPIAUCSL", "fred", start)
    cpi = cpi.dropna()
    cpi = cpi.sort_index()
    return cpi

def cpi_to_inflation_rate(cpi_df):
    """Calculates inflation rate from CPI data"""
    inflation = cpi_df.pct_change().dropna()
    inflation = inflation.rename(columns=lambda x: "inflation_rate")
    return inflation

def to_annual_inflation(monthly_inflation):
    """Converts to an annual inflation rate"""
    annual = (1 + monthly_inflation).resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    )
    annual.index = annual.index.year
    return annual

def align_inflation_series(annual_inflation, start_year, horizon):
    """Processes the inflation rates to align with the simulation"""
    years = list(range(start_year, start_year + horizon))

    aligned = []
    last_val = annual_inflation.iloc[-1].values[0]

    for y in years:
        if y in annual_inflation.index:
            aligned.append(annual_inflation.loc[y].values[0])
        else:
            aligned.append(last_val)  # fallback

    return np.array(aligned)

def get_annual_inflation_series(start="2000-01-01"):
    """Pipeline function"""
    cpi = load_cpi(start)
    monthly_infl = cpi_to_inflation_rate(cpi)
    annual_infl = to_annual_inflation(monthly_infl)

    return annual_infl

def get_daily_inflation_series(start = "2000-01-01"):
    cpi = load_cpi(start)
    cpi_daily = (
        cpi 
        .resample("D")
        .ffill()
    )
    daily_inflation_returns = (
        cpi_daily
        .pct_change()
        .fillna(0.0)
    )
    return (
        daily_inflation_returns.iloc[:,0],
        cpi_daily.iloc[:,0]
    )