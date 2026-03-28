from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import init_db
from api import auth_routes, user_routes, questionnaire_routes, portfolio_routes, recommend_routes, officials_routes
import uvicorn

app = FastAPI(title="Stock Recommender API", version="0.1.0")

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


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def health():
    return {"status": "ok", "message": "Stock Recommender API Running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
