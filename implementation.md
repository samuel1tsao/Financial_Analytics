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

### Component 2: Congressional Trading Data (AI-Assisted Scrape → CSV → User Verification)
Instead of auto-populating the database from fragile scrapers, we use an **AI-assisted pipeline** that produces a CSV file for human review. The user verifies accuracy, then pushes the approved data into the system.
- **House & Senate Scraper:** Navigates the search forms and downloads PTR results, outputting to a CSV file (`data/congress_trades_raw.csv`).
- **CSV Loader Toolkit:** Reads a user-verified CSV and upserts records into the Database.

### Component 3: Market Data API Endpoint
- Added `GET /api/v1/market/stats` to return current market data from DB (returns, volatilities, correlation matrix) for the asset universe.

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
- [ ] Complete Testing: Backend startup DB population and market stats endpoints.
- [ ] Complete Testing: Questionnaire → dashboard flow works with real data.
- [ ] Scraper Verification: Test congressional scraper outputs valid CSV.
- [ ] Scraper Verification: Test CSV loader logic against database.
- [ ] Update root `README.md` with new setup/data instructions (yfinance sync and CSV load process).
