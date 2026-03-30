"""
Officials Service — Serves public officials' portfolio data.

Primary path: reads from CongressMember + CongressTrade tables (real scraped data).
Fallback path: returns the curated static OFFICIALS list when DB has no data.
"""
import json
import logging
from collections import defaultdict
from datetime import date, timedelta

from sqlalchemy.orm import Session
from database import SessionLocal
from congress_models import CongressMember, CongressTrade

logger = logging.getLogger(__name__)


# ─── Curated Fallback Dataset ────────────────────────────────────────────────
# Kept for offline development and as a guaranteed baseline.
FALLBACK_OFFICIALS = [
    {
        "id": "pelosi",
        "name": "Nancy Pelosi",
        "title": "Former Speaker of the House",
        "party": "Democrat",
        "state": "California",
        "total_value": 65_000_000,
        "portfolio": {
            "AAPL": 0.18, "MSFT": 0.14, "GOOGL": 0.12, "AMZN": 0.10,
            "NVDA": 0.11, "CRM": 0.08, "RBLX": 0.05, "DIS": 0.06,
            "TSLA": 0.09, "PYPL": 0.07,
        },
        "top_trades": [
            {"ticker": "NVDA", "action": "BUY", "amount": 5_000_000, "date": "2024-06-20"},
            {"ticker": "AAPL", "action": "BUY", "amount": 3_000_000, "date": "2024-03-15"},
            {"ticker": "RBLX", "action": "SELL", "amount": 1_500_000, "date": "2024-09-01"},
        ],
        "performance_1y": 0.32,
        "performance_5y": 1.45,
        "data_source": "curated",
    },
    {
        "id": "tuberville",
        "name": "Tommy Tuberville",
        "title": "U.S. Senator",
        "party": "Republican",
        "state": "Alabama",
        "total_value": 8_500_000,
        "portfolio": {
            "XOM": 0.15, "CVX": 0.12, "LMT": 0.10, "RTX": 0.09,
            "GD": 0.08, "DVN": 0.07, "HAL": 0.06, "SLB": 0.05,
            "NOC": 0.08, "BA": 0.07, "JPM": 0.06, "GS": 0.07,
        },
        "top_trades": [
            {"ticker": "LMT", "action": "BUY", "amount": 500_000, "date": "2024-07-10"},
            {"ticker": "DVN", "action": "SELL", "amount": 250_000, "date": "2024-05-22"},
            {"ticker": "RTX", "action": "BUY", "amount": 400_000, "date": "2024-08-14"},
        ],
        "performance_1y": 0.18,
        "performance_5y": 0.72,
        "data_source": "curated",
    },
    {
        "id": "ossoff",
        "name": "Jon Ossoff",
        "title": "U.S. Senator",
        "party": "Democrat",
        "state": "Georgia",
        "total_value": 4_200_000,
        "portfolio": {
            "MSFT": 0.20, "GOOGL": 0.15, "AMZN": 0.12, "V": 0.10,
            "MA": 0.08, "UNH": 0.10, "JNJ": 0.08, "PG": 0.07,
            "BRK.B": 0.05, "HD": 0.05,
        },
        "top_trades": [
            {"ticker": "MSFT", "action": "BUY", "amount": 300_000, "date": "2024-04-10"},
            {"ticker": "UNH", "action": "BUY", "amount": 200_000, "date": "2024-06-05"},
        ],
        "performance_1y": 0.25,
        "performance_5y": 1.10,
        "data_source": "curated",
    },
    {
        "id": "scott_rick",
        "name": "Rick Scott",
        "title": "U.S. Senator",
        "party": "Republican",
        "state": "Florida",
        "total_value": 250_000_000,
        "portfolio": {
            "GLD": 0.12, "TLT": 0.10, "VNQ": 0.08, "SPY": 0.15,
            "QQQ": 0.10, "IWM": 0.05, "XLF": 0.08, "XLV": 0.07,
            "XLE": 0.10, "BND": 0.08, "TIPS": 0.07,
        },
        "top_trades": [
            {"ticker": "GLD", "action": "BUY", "amount": 10_000_000, "date": "2024-01-20"},
            {"ticker": "XLE", "action": "BUY", "amount": 8_000_000, "date": "2024-05-15"},
        ],
        "performance_1y": 0.14,
        "performance_5y": 0.55,
        "data_source": "curated",
    },
    {
        "id": "crenshaw",
        "name": "Dan Crenshaw",
        "title": "U.S. Representative",
        "party": "Republican",
        "state": "Texas",
        "total_value": 1_800_000,
        "portfolio": {
            "MSFT": 0.18, "AAPL": 0.15, "GOOGL": 0.10, "META": 0.12,
            "AMZN": 0.10, "JPM": 0.08, "BAC": 0.07, "XOM": 0.08,
            "CVX": 0.06, "PFE": 0.06,
        },
        "top_trades": [
            {"ticker": "META", "action": "BUY", "amount": 100_000, "date": "2024-02-28"},
            {"ticker": "AAPL", "action": "BUY", "amount": 80_000, "date": "2024-07-01"},
        ],
        "performance_1y": 0.28,
        "performance_5y": 1.05,
        "data_source": "curated",
    },
    {
        "id": "warren",
        "name": "Elizabeth Warren",
        "title": "U.S. Senator",
        "party": "Democrat",
        "state": "Massachusetts",
        "total_value": 12_000_000,
        "portfolio": {
            "VTI": 0.30, "VXUS": 0.15, "BND": 0.20, "TIPS": 0.10,
            "VNQ": 0.05, "SCHD": 0.10, "VTIP": 0.05, "MUB": 0.05,
        },
        "top_trades": [
            {"ticker": "VTI", "action": "BUY", "amount": 500_000, "date": "2024-03-01"},
        ],
        "performance_1y": 0.12,
        "performance_5y": 0.48,
        "data_source": "curated",
    },
    {
        "id": "cruz",
        "name": "Ted Cruz",
        "title": "U.S. Senator",
        "party": "Republican",
        "state": "Texas",
        "total_value": 5_500_000,
        "portfolio": {
            "XOM": 0.12, "CVX": 0.10, "OXY": 0.08, "DVN": 0.06,
            "GLD": 0.10, "BTC-ETF": 0.05, "SPY": 0.15, "QQQ": 0.08,
            "LMT": 0.06, "RTX": 0.05, "AMZN": 0.08, "GOOGL": 0.07,
        },
        "top_trades": [
            {"ticker": "OXY", "action": "BUY", "amount": 250_000, "date": "2024-04-22"},
            {"ticker": "BTC-ETF", "action": "BUY", "amount": 150_000, "date": "2024-08-10"},
        ],
        "performance_1y": 0.20,
        "performance_5y": 0.75,
        "data_source": "curated",
    },
    {
        "id": "schumer",
        "name": "Chuck Schumer",
        "title": "Senate Majority Leader",
        "party": "Democrat",
        "state": "New York",
        "total_value": 1_200_000,
        "portfolio": {
            "VOO": 0.25, "VTI": 0.20, "BND": 0.20, "VNQ": 0.10,
            "SCHD": 0.10, "VXUS": 0.10, "VTIP": 0.05,
        },
        "top_trades": [
            {"ticker": "VOO", "action": "BUY", "amount": 100_000, "date": "2024-06-15"},
        ],
        "performance_1y": 0.15,
        "performance_5y": 0.52,
        "data_source": "curated",
    },
]




def _build_top_trades(trades: list[CongressTrade], limit: int = 5) -> list[dict]:
    """Return the most recent trades formatted for the API response."""
    recent = sorted(
        [t for t in trades if t.transaction_date],
        key=lambda t: t.transaction_date,
        reverse=True,
    )[:limit]

    return [
        {
            "ticker": t.ticker,
            "action": t.transaction_type.upper() if t.transaction_type else "UNKNOWN",
            "amount": (t.amount_low + t.amount_high) / 2 if t.amount_low and t.amount_high else 0,
            "date": str(t.transaction_date) if t.transaction_date else "",
        }
        for t in recent
    ]


def _member_to_official(member: CongressMember) -> dict:
    """Convert a CongressMember + trades into the official API response format."""
    top_trades = _build_top_trades(member.trades)

    # Transform historical equity into a simple time-series list for charting
    history = [
        {"date": str(h.date), "value": h.total_value}
        for h in member.portfolio_history
    ] if member.portfolio_history else []

    return {
        "id": str(member.id),
        "name": member.name,
        "title": f"U.S. {'Senator' if member.chamber == 'senate' else 'Representative'}",
        "party": member.party or "Unknown",
        "state": member.state or "Unknown",
        "total_value": round(member.total_value),
        "portfolio": member.portfolio_weights or {},
        "top_trades": top_trades,
        "historical_equity": history,
        "performance_1y": member.performance_1y,
        "performance_5y": member.performance_5y,
        "data_source": "database",
        "last_updated": str(member.last_updated) if member.last_updated else None,
    }


def get_all_officials(
    chamber: str | None = None,
    party: str | None = None,
) -> list[dict]:
    """
    Return all tracked public officials.
    Tries the database first; falls back to the curated list.
    """
    db = SessionLocal()
    try:
        query = db.query(CongressMember)
        if chamber:
            query = query.filter(CongressMember.chamber == chamber.lower())
        if party:
            query = query.filter(CongressMember.party.ilike(f"%{party}%"))

        members = query.all()

        if members:
            return [_member_to_official(m) for m in members]

        logger.info("No congress data in DB, returning curated fallback")

        # Apply filters to fallback data too
        officials = FALLBACK_OFFICIALS
        if chamber:
            chamber_map = {"house": "Representative", "senate": "Senator"}
            keyword = chamber_map.get(chamber.lower(), "")
            if keyword:
                officials = [o for o in officials if keyword in o.get("title", "")]
        if party:
            officials = [o for o in officials if party.lower() in o.get("party", "").lower()]

        return officials

    finally:
        db.close()


def get_official_by_id(official_id: str) -> dict | None:
    """Look up a single official by their ID."""
    db = SessionLocal()
    try:
        # Try DB first (ID is the integer member ID as string)
        try:
            member_id = int(official_id)
            member = db.query(CongressMember).filter(CongressMember.id == member_id).first()
            if member:
                return _member_to_official(member)
        except ValueError:
            pass  # Not a numeric ID, try fallback

        # Fallback: search curated list
        for o in FALLBACK_OFFICIALS:
            if o["id"] == official_id:
                return o

        return None

    finally:
        db.close()
