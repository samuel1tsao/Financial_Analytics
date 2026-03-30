# Financial Analytics — Stock Recommender

A full-stack web application that generates personalized ETF/stock portfolio recommendations based on a user's financial goals, risk tolerance, and behavioral tendencies (FOMO scoring). It also lets users browse the disclosed stock portfolios of U.S. public officials and "mimic" their allocations.

## Features

- **Investment Questionnaire** — 5-step guided flow capturing financial goals, risk tolerance (1–100 slider), FOMO tendency (situational quiz), hard allocation constraints, and existing holdings.
- **Portfolio Recommendation Engine** — Encodes questionnaire answers into a weighted ETF allocation across a 12-asset universe (8 equity, 4 bond) using risk-based equity/bond split, FOMO-adjusted speculative tilt, and short-term goal safeguards. Driven by actual market data downloaded via `yfinance`.
- **Growth Projection Simulator** — Runs a variance-covariance Monte Carlo simulation on any saved portfolio, producing a 30-year expected path with ±2σ confidence bands and cash-out event modeling for short-term goals. *Interactive sandbox mode available on the main dashboard!*
- **Public Officials Tracker** — Browse portfolios of tracked U.S. officials pulling directly from a `capitoltrades.com` data pipeline. See their live stock holdings, timeline of trades, and Recharts historical performance progression mapped against true stock market data.
- **User Accounts & Auth** — Email/password registration with JWT-based authentication (bcrypt hashing, 24hr token expiry).
- **Portfolio Management** — Save multiple named portfolio profiles, mark one as "current", bulk delete.

## Tech Stack

| Layer     | Technology |
|-----------|------------|
| Frontend  | React 19, Vite 8, Zustand (state), Recharts (charts), React Router 7, Axios, TailwindCSS 4 |
| Backend   | Python, FastAPI, SQLAlchemy, SQLite |
| Auth      | JWT (python-jose), bcrypt |
| Analytics | NumPy, scikit-learn, PyPortfolioOpt |

## Project Structure

```
Financial_Analytics/
├── backend/
│   ├── main.py                  # FastAPI entrypoint (uvicorn, CORS, router setup)
│   ├── database.py              # SQLAlchemy engine + session (SQLite)
│   ├── models.py                # Core App ORM models: User, Questionnaire, Portfolio, Favorite
│   ├── congress_models.py       # DB Models: CongressMember, CongressTrade, CongressPortfolioHistory
│   ├── market_models.py         # DB Models: StockPrice, TickerFeature
│   ├── schemas.py               # Pydantic request/response schemas
│   ├── auth.py                  # Password hashing, JWT creation, get_current_user dependency
│   ├── vector_encoder.py        # Questionnaire → portfolio weights + simulation engine
│   ├── market_data.py           # yFinance background downloader + Covariance matrix compute
│   ├── officials_service.py     # Aggregates Congress trades + acts as API mediator
│   ├── scrape_capitol_trades.py # The CLI Web Scraper for tracking public officials
│   ├── profile_builder.py       # Historical portfolio performance calculator script
│   ├── requirements.txt
│   └── api/
│       ├── auth_routes.py       # POST /register, /login
│       ├── market_routes.py     # GET /market/stats, /market/history
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

### 3. Frontend setup

Open a **second terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Start the dev server (runs on http://localhost:5173)
npm run dev
```

### 4. Data Ingestion (Optional)
On startup, the backend will automatically sync foundational Market Data from Yahoo Finance and run a cached portfolio build. 
However, to pull down **real-time Congressional trades**, you should run the web scraper manually (we advise running this on a weekly CRON job to avoid getting blacklisted by CapitolTrades):

```bash
cd backend
python scrape_capitol_trades.py --full --pages 10
python profile_builder.py
```

### 5. Use the app

1. Open **http://localhost:5173** in your browser.
2. Register a new account.
3. Complete the investment questionnaire — a portfolio recommendation is generated automatically on submission.
4. View your portfolio dashboard to interact with the **Dynamic Interactive Sandbox** growth projection charts. You can click on the underlying allocation ETFs to view their historical progression.
5. Browse the **Officials** tab in the sidebar to explore public officials' live portfolios, interactive tracking charts, and optionally mimic them.

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
- Fast API integrates with `yfinance` to automatically backfill historical stock data on boot, which may delay server boot by 15-30 seconds.
