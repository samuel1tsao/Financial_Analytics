"""
Static dataset of public officials' disclosed stock portfolios.
In production this would pull from SEC EDGAR / Senate Financial Disclosures API,
but for MVP we use a curated representative dataset.
"""

OFFICIALS = [
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
    },
]


def get_all_officials() -> list[dict]:
    """Return the full list of tracked public officials."""
    return OFFICIALS


def get_official_by_id(official_id: str) -> dict | None:
    """Look up a single official by their internal ID."""
    for o in OFFICIALS:
        if o["id"] == official_id:
            return o
    return None
