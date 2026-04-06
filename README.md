# Financial Analytics — Stock Recommender

A full-stack web application that generates personalized ETF/stock portfolio recommendations based on a user's financial goals, risk tolerance, and behavioral tendencies (FOMO scoring). It also lets users browse the disclosed stock portfolios of U.S. public officials and "mimic" their allocations.

## Features

- **Investment Questionnaire** — 5-step guided flow capturing financial goals, risk tolerance (1–100 slider), FOMO tendency (situational quiz), hard allocation constraints, and existing holdings.
- **Portfolio Recommendation Engine** — Encodes questionnaire answers into a weighted ETF allocation across a 12-asset universe (8 equity, 4 bond) using risk-based equity/bond split, FOMO-adjusted speculative tilt, and short-term goal safeguards.
- **Growth Projection Simulator** — Runs a variance-covariance simulation on any saved portfolio, producing a 30-year expected path with ±2σ confidence bands and cash-out event modeling for short-term goals.
- **Public Officials Tracker** — Browse portfolios of 8 tracked U.S. officials (Pelosi, Tuberville, Ossoff, etc.) with holdings, recent trades, and 1yr/5yr performance. Users can copy ("mimic") any official's portfolio as a saved profile.
- **User Accounts & Auth** — Email/password registration with JWT-based authentication (bcrypt hashing, 24hr token expiry).
- **Data Science Research Environment** — Standalone Jupyter Notebooks (`research_modeling.ipynb` and Colab version) built for quants. Dynamically scrapes the full S&P 1500 universe and uses pure Pandas vectorization to rapidly compute historical correlations, technical indicators (MACD, RSI), and Options Greeks natively skipping the web backend.
- **Portfolio Management** — Save multiple named portfolio profiles, mark one as "current", bulk delete.

## Tech Stack

| Layer     | Technology |
|-----------|------------|
| Frontend  | React 19, Vite 8, Zustand (state), Recharts (charts), React Router 7, Axios, TailwindCSS 4 |
| Backend   | Python, FastAPI, SQLAlchemy, SQLite |
| Auth      | JWT (python-jose), bcrypt |
| Analytics | NumPy, scikit-learn, PyPortfolioOpt |
| Data      | yfinance, BeautifulSoup4, pandas |

## Project Structure

```
Financial_Analytics/
├── backend/
│   ├── main.py                  # FastAPI entrypoint (uvicorn, CORS, router setup)
│   ├── database.py              # SQLAlchemy engine + session (SQLite)
│   ├── models.py                # ORM models: User, Questionnaire, Portfolio, Favorite
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── auth.py                  # Password hashing, JWT creation, get_current_user dependency
│   ├── vector_encoder.py        # Questionnaire → portfolio weights + simulation engine
│   ├── market_data.py           # yFinance integration for historical prices/returns
│   ├── congress_scraper.py      # AI-assisted scraper for congressional trades
│   ├── congress_loader.py       # Loads verified CSV trades into SQLite
│   ├── officials_service.py     # Serves public officials' portfolio data (live + fallback)
│   ├── requirements.txt
│   └── api/
│       ├── auth_routes.py       # POST /register, /login
│       ├── user_routes.py       # GET /user/data, POST /user/favorites
│       ├── questionnaire_routes.py  # POST /questionnaire/save, GET /questionnaire/current
│       ├── portfolio_routes.py  # POST /profile/save, DELETE /profile
│       ├── recommend_routes.py  # POST /recommend, POST /simulate
│       └── officials_routes.py  # GET /officials, GET /officials/:id, POST /officials/:id/mimic
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Router + protected layout shell
│   │   ├── store.js             # Zustand global state (auth, user data, questionnaire)
│   │   ├── api/client.js        # Axios instance with JWT interceptor
│   │   ├── views/
│   │   │   ├── Auth/Login.jsx           # Login / Register page
│   │   │   ├── Dashboard/Main.jsx       # Portfolio dashboard + simulation chart
│   │   │   ├── Questionnaire/Stepper.jsx  # 5-step questionnaire wizard
│   │   │   └── Officials/Directory.jsx  # Public officials browser
│   │   └── components/
│   │       ├── TopNavigation.jsx
│   │       └── LeftSidebar.jsx
│   ├── package.json
│   └── vite.config.js
└── .gitignore
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **npm**

## Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd Financial_Analytics
```

### 2. Backend setup

```bash
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Start the API server (runs on http://localhost:8000)
python main.py
```

The SQLite database (`stock_recommender.db`) is created automatically on first startup. 

**Note on Market Data:** On startup, the backend automatically fetches up to 15 years of daily historical data for the 12-ticker asset universe using `yfinance`. This data is stored locally and used to compute real rolling returns and variance-covariance matrices.

### 3. Congressional Data
Because government disclosure sites (House/Senate PTRs) are notoriously difficult to scrape cleanly, this project uses an AI-assisted pipeline for official trades:
```bash
# 1. Scrape latest disclosures to a raw CSV
python congress_scraper.py

# 2. Review the CSV manually to ensure data integrity
# 3. Load the verified CSV into the SQLite database
python congress_loader.py data/congress_trades_verified.csv
```

### 3. Frontend setup

Open a **second terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server (runs on http://localhost:5173)
npm run dev
```

### 4. Use the app

1. Open **http://localhost:5173** in your browser.
2. Register a new account.
3. Complete the investment questionnaire — a portfolio recommendation is generated automatically on submission.
4. View your portfolio dashboard with growth projection charts and allocation breakdown.
5. Browse the **Officials** tab in the sidebar to explore public officials' portfolios and optionally mimic them.

## API Endpoints

All routes are prefixed with `/api/v1`.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/user/register` | No | Create account |
| `POST` | `/user/login` | No | Get JWT token |
| `GET`  | `/user/data` | Yes | Full user profile (portfolios, favorites) |
| `POST` | `/user/favorites?official_id=` | Yes | Toggle favorite official |
| `POST` | `/questionnaire/save` | Yes | Save questionnaire answers |
| `GET`  | `/questionnaire/current` | Yes | Get latest questionnaire |
| `POST` | `/recommend` | Yes | Generate portfolio from questionnaire |
| `POST` | `/simulate` | Yes | Run growth simulation on a portfolio |
| `POST` | `/profile/save` | Yes | Save a portfolio profile |
| `DELETE`| `/profile` | Yes | Bulk delete portfolio profiles |
| `GET`  | `/officials` | No | List all tracked officials |
| `GET`  | `/officials/:id` | No | Get single official detail |
| `POST` | `/officials/:id/mimic` | Yes | Copy official's portfolio to user |

## Environment Notes

- The backend uses a hardcoded JWT secret (`dev-secret-key-change-in-production`). For production, move this to an environment variable.
- CORS is configured to allow `localhost:5173` and `localhost:3000`.
- Market data is pulled via `yfinance`. Please be mindful of Yahoo Finance rate limits if modifying `market_data.py` to pull hundreds of tickers simultaneously.
- The officials dataset uses a curated fallback list if the database is completely empty. To populate it with live data, run the `congress_scraper.py` and stringently review the CSV before using `congress_loader.py`.
