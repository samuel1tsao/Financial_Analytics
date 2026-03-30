from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from api import auth_routes, user_routes, questionnaire_routes, portfolio_routes, recommend_routes, officials_routes, market_routes
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Recommender API", version="0.2.0")

# Allow React dev server to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route modules
app.include_router(auth_routes.router)
app.include_router(user_routes.router)
app.include_router(questionnaire_routes.router)
app.include_router(portfolio_routes.router)
app.include_router(recommend_routes.router)
app.include_router(officials_routes.router)
app.include_router(market_routes.router)


@app.on_event("startup")
def on_startup():
    # Import models so tables are registered with Base before create_all
    import models  # noqa: F401
    import market_models  # noqa: F401
    import congress_models  # noqa: F401

    # Create all tables
    init_db()
    logger.info("Database tables initialized")

    # Sync market data from yfinance (incremental — only fetches missing days)
    try:
        from market_data import sync_market_data
        summary = sync_market_data()
        logger.info(f"Market data sync: {summary}")
    except Exception as e:
        logger.error(f"Market data sync failed: {e}")

    # Automatically rebuild congress profile equity histories on startup
    try:
        from profile_builder import build_all_profiles
        build_all_profiles()
    except Exception as e:
        logger.error(f"Congress profile build failed: {e}")


@app.get("/")
def health():
    return {"status": "ok", "message": "Stock Recommender API Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
