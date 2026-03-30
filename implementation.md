# Implementation Plan & Progress: Integrating Real Financial Data

This document tracks the plan and progress for transitioning the Financial Analytics platform from hardcoded, simulated financial data to a functional application utilizing real-world data (yfinance API and Congressional stock trading disclosures).

## Overview of Changes

### Component 1: Market Data Service (yfinance → SQLite)
Database-backed market data service:
- **Fetches 15+ years** of daily adjusted close prices for all tickers in the asset universe (max history available from Yahoo Finance)
- Stores raw daily prices in a new `StockPrice` table (ticker, date, adj_close)
- **On startup:** checks the latest date in `StockPrice` for each ticker; only fetches missing days from yfinance (incremental sync)
- **Rolling 1-year annualized returns:** For each ticker and each date, computes the return over the trailing 252 trading days. These rolling returns are stored in a `TickerFeature` table and serve as **feature vectors for training the recommendation model**
- Computes **annualized volatility** over the full history
- Computes a **full pairwise correlation matrix** from daily returns (replacing the 3-entry category approximation)
- Builds a **real variance-covariance matrix** from the above
- Exposes `get_market_data()` function that reads from DB (fast, no network call)
- Falls back to existing hardcoded values if a ticker has no DB data (first run with no internet, etc.)

### Component 2: Congressional Data (Direct DB Scraping & Profile Rebuilding)
Instead of using rate-limited APIs or intermediate CSV files, we are implementing a robust BeautifulSoup scraper targeting `capitoltrades.com`.
- **`scrape_capitol_trades.py`**: A CLI scraper that pulls trades. It features two modes:
  - *Normal Sync*: Queries the local database for the newest `transaction_date` and stops scraping immediately when it detects older data. This saves bandwidth and acts responsibly towards the host. It includes a `time.sleep(2)` polite delay.
  - *Full Sync*: Scrapes to a set depth (e.g. `pages=10`) and selectively replaces stale data in our database within that date range.
- **Enhanced Demographics Extraction**: The scraper actively parses HTML classes (e.g. `q-field party party--republican`) to correctly capture Politician Party, Chamber, and State during the scrape.
- **`profile_builder.py`**: A new automated routine (called on startup, via API, and on intervals) that iterates over the raw trades and builds daily Historical Equity Curves. It calculates net shares over time and stores `CongressPortfolioHistory` so the UI can draw progression charts.

### Component 3: Market Data API Endpoint
- Updated `GET /api/v1/market/stats` to an endpoint that supports historical ETF charts: `GET /api/v1/market/history?ticker=VOO&range=1y`. This returns actual closing price and return vectors for UI visualizations, with ending % change.

### Component 4: Frontend Updates
- Updated the Officials Directory to support chamber filters (House/Senate), sorting options, and displaying a "Data Source" badge (Live DB vs. Curated fallback).

---

## Progress Checklist

### Component 1: Market Data Service
- [x] Create `market_models.py` — StockPrice, TickerFeature tables
- [x] Create `market_data.py` — sync service, rolling returns, correlation matrix
- [x] Modify `vector_encoder.py` — use real data from DB, hardcoded as fallback
- [x] Modify `main.py` — startup sync + import market models

### Component 2: Congressional Trading Data
- [x] Create `congress_models.py` — CongressMember, CongressTrade tables
- [x] Create `congress_scraper.py` — AI-assisted scraper → CSV output
- [x] Create `congress_loader.py` — CSV → DB loader (CLI + API)
- [x] Refactor `officials_data.py` → `officials_service.py`
- [x] Modify `officials_routes.py` — scrape/load endpoints + filters
- [x] Modify `models.py` — import congress models

### Component 3: Market Data API
- [x] Create `market_routes.py` — GET /market/stats
- [x] Modify `api/__init__.py` — register market_routes

### Component 4: Frontend Updates
- [x] Modify `Directory.jsx` — chamber/party filters, data source indicator

### Remaining / Verification Left To Do
- [x] Complete Testing: Backend startup DB population and market stats endpoints.
- [x] Complete Testing: Questionnaire → dashboard flow works with real data.
- [x] Scraper Verification: Test congressional scraper outputs valid CSV/DB entries.
- [x] Scraper Verification: Test CSV loader/Profile builder logic against database.
- [x] Update root `README.md` with new setup/data instructions (yfinance sync and CSV load process).
