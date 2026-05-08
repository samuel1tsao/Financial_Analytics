from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


# ─── Auth ────────────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: int
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Questionnaire ───────────────────────────────────────────────────────────
class GoalItem(BaseModel):
    name: str
    amount: float
    years: int


class HardConstraint(BaseModel):
    ticker: str
    pct: float


class QuestionnaireV1(BaseModel):
    goals: list[GoalItem] = []
    drawdown_answers: list[Optional[int]] = []
    volatility_answers: list[Optional[int]] = []
    drawdown_sensitivity: int = Field(ge=1, le=10, default=5)
    volatility_sensitivity: int = Field(ge=1, le=10, default=5)
    goal_flexibility: int = Field(ge=1, le=10, default=5)
    concentration_pref: int = Field(ge=1, le=10, default=5)
    start_cap: float = 100000.0
    monthly_contrib: float = 500.0
    hard_constraints: list[HardConstraint] = []
    current_portfolio: list[HardConstraint] = []
    # Reserved logic (e.g. forced 10% AAPL)
    reserved_asset: str = "AAPL"
    reserved_ratio: float = 0.1


class QuestionnaireSave(BaseModel):
    schema_version: str = "v1"
    answers: QuestionnaireV1


# ─── Portfolio ───────────────────────────────────────────────────────────────
class PortfolioSave(BaseModel):
    profile_name: str
    is_current: bool = False
    weights: dict  # e.g. {"VOO": 0.45, "BND": 0.30, "AAPL": 0.10, ...}


class PortfolioOut(BaseModel):
    id: int
    profile_name: str
    is_current: bool
    weight_json: str
    created_at: datetime

    class Config:
        from_attributes = True


class SimulationRequest(BaseModel):
    portfolio_id: Optional[int] = None
    initial_investment: float = 100000
    projection_years: int = 30
    custom_goals_json: str = "[]"


class PortfolioDeleteRequest(BaseModel):
    ids: list[int]


class SimulationBenchmarkRequest(BaseModel):
    benchmark_type: str  # "sp500" or "conservative_60_40"
    initial_investment: float = 100000
    projection_years: int = 30
