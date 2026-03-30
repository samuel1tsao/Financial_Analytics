"""
Congressional Trading Scraper — AI-Assisted Pipeline

Scrapes congressional stock trading disclosures from official government portals
(House and Senate) and outputs structured data as a CSV file for human review.

The CSV is NOT auto-loaded into the database. The user must review and verify
accuracy before pushing through congress_loader.py.

Usage:
    # As module
    from congress_scraper import scrape_disclosures
    csv_path = scrape_disclosures()

    # As CLI
    python congress_scraper.py [--output data/congress_trades_raw.csv]
"""
import csv
import logging
import os
import re
import time
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT_FILE = "congress_trades_raw.csv"
REQUEST_DELAY = 1.0  # seconds between requests
USER_AGENT = "FinancialAnalytics/1.0 (Educational Research Project)"

CSV_COLUMNS = [
    "member_name", "chamber", "party", "state", "ticker",
    "transaction_type", "amount_low", "amount_high",
    "transaction_date", "disclosure_date", "source_url",
]

# ─── Amount range parsing ────────────────────────────────────────────────────
AMOUNT_RANGES = {
    "$1,001 - $15,000": (1001, 15000),
    "$15,001 - $50,000": (15001, 50000),
    "$50,001 - $100,000": (50001, 100000),
    "$100,001 - $250,000": (100001, 250000),
    "$250,001 - $500,000": (250001, 500000),
    "$500,001 - $1,000,000": (500001, 1000000),
    "$1,000,001 - $5,000,000": (1000001, 5000000),
    "$5,000,001 - $25,000,000": (5000001, 25000000),
    "$25,000,001 - $50,000,000": (25000001, 50000000),
    "$50,000,001 - ": (50000001, None),
}


def _parse_amount_range(amount_str: str) -> tuple[float | None, float | None]:
    """Parse a STOCK Act amount range string into (low, high)."""
    if not amount_str:
        return (None, None)

    amount_str = amount_str.strip()
    for pattern, (low, high) in AMOUNT_RANGES.items():
        if pattern in amount_str:
            return (low, high)

    # Try to extract numbers directly
    numbers = re.findall(r'[\d,]+', amount_str.replace('$', ''))
    if len(numbers) >= 2:
        return (float(numbers[0].replace(',', '')), float(numbers[1].replace(',', '')))
    elif len(numbers) == 1:
        val = float(numbers[0].replace(',', ''))
        return (val, val)
    return (None, None)


def _extract_ticker(description: str) -> str | None:
    """Try to extract a stock ticker from an asset description."""
    if not description:
        return None

    # Common patterns: "AAPL - Apple Inc", "MSFT", "(TSLA)", "[NVDA]"
    # Look for uppercase letter sequences that look like tickers
    patterns = [
        r'\(([A-Z]{1,5})\)',            # (AAPL)
        r'\[([A-Z]{1,5})\]',            # [AAPL]
        r'^([A-Z]{1,5})\s*[-–—]',       # AAPL - Apple Inc
        r'^([A-Z]{1,5})$',              # AAPL
        r'\b([A-Z]{1,5})\b',            # any uppercase word <= 5 chars
    ]

    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            ticker = match.group(1)
            # Filter out common non-ticker words
            if ticker not in {"THE", "AND", "FOR", "INC", "LLC", "ETF", "LTD",
                              "USD", "USA", "NEW", "SEC", "ACT", "ALL", "CEO"}:
                return ticker
    return None


def _parse_date(date_str: str) -> str | None:
    """Try to parse various date formats into YYYY-MM-DD."""
    if not date_str or not date_str.strip():
        return None

    date_str = date_str.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _scrape_house_disclosures(year: int | None = None) -> list[dict]:
    """
    Scrape Periodic Transaction Reports from the House Clerk's website.
    The site uses a search form at:
    https://disclosures-clerk.house.gov/FinancialDisclosure#Search

    Returns list of trade dicts matching CSV_COLUMNS.
    """
    trades = []
    if year is None:
        year = date.today().year

    # The House disclosure search uses a POST-based ASPX form.
    # We attempt to search and parse the results table.
    search_url = "https://disclosures-clerk.house.gov/FinancialDisclosure/Search"

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        # Load the search page first to get any CSRF/form tokens
        time.sleep(REQUEST_DELAY)
        page = session.get(
            "https://disclosures-clerk.house.gov/FinancialDisclosure",
            timeout=30,
        )

        if page.status_code != 200:
            logger.warning(f"House disclosure page returned {page.status_code}")
            return trades

        # Attempt the search
        time.sleep(REQUEST_DELAY)
        search_data = {
            "FilingYear": str(year),
            "State": "",
            "District": "",
            "LastName": "",
            "ReportTypes": "PTR",  # Periodic Transaction Reports
        }
        result = session.post(search_url, data=search_data, timeout=30)

        if result.status_code != 200:
            logger.warning(f"House search returned {result.status_code}")
            return trades

        soup = BeautifulSoup(result.text, "html.parser")

        # Look for results table
        table = soup.find("table", class_="library-table") or soup.find("table")
        if not table:
            logger.info("No results table found on House disclosures page")
            return trades

        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip header row
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            name = cells[0].get_text(strip=True)
            # Additional fields depend on table structure
            filing_date = _parse_date(cells[-1].get_text(strip=True)) if len(cells) > 3 else None

            # Look for a PDF link to the actual filing
            link = row.find("a", href=True)
            source_url = ""
            if link:
                href = link["href"]
                if not href.startswith("http"):
                    href = f"https://disclosures-clerk.house.gov{href}"
                source_url = href

            # The main search page lists filings, not individual trades.
            # Each filing would need to be opened to get trade details.
            # For now, record the filing as a single entry.
            trades.append({
                "member_name": name,
                "chamber": "house",
                "party": "",
                "state": "",
                "ticker": "",
                "transaction_type": "",
                "amount_low": "",
                "amount_high": "",
                "transaction_date": "",
                "disclosure_date": filing_date or "",
                "source_url": source_url,
            })

            time.sleep(REQUEST_DELAY)

        logger.info(f"Found {len(trades)} House PTR filings for {year}")

    except requests.RequestException as e:
        logger.error(f"Error scraping House disclosures: {e}")
    except Exception as e:
        logger.error(f"Unexpected error scraping House disclosures: {e}")

    return trades


def _scrape_senate_disclosures(year: int | None = None) -> list[dict]:
    """
    Scrape Periodic Transaction Reports from the Senate eFD system.
    https://efdsearch.senate.gov/search/

    Returns list of trade dicts matching CSV_COLUMNS.
    """
    trades = []
    if year is None:
        year = date.today().year

    search_url = "https://efdsearch.senate.gov/search/"

    try:
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        # Accept the agreement page (required)
        time.sleep(REQUEST_DELAY)
        agreement_url = "https://efdsearch.senate.gov/search/home/"
        session.get(agreement_url, timeout=30)

        # The Senate eFD uses JavaScript-heavy search.
        # Attempt a direct search API call.
        time.sleep(REQUEST_DELAY)
        api_url = "https://efdsearch.senate.gov/search/report/data/"
        search_payload = {
            "start": "0",
            "length": "100",
            "report_type_id": "11",  # Periodic Transaction Report
            "filer_type_id": "1",    # Senator
            "submitted_start_date": f"01/01/{year}",
            "submitted_end_date": f"12/31/{year}",
        }

        result = session.post(api_url, data=search_payload, timeout=30)

        if result.status_code == 200:
            try:
                data = result.json()
                records = data.get("data", [])

                for record in records:
                    # Senate API returns array of arrays
                    if isinstance(record, list) and len(record) >= 5:
                        # Extract name from HTML in first column
                        name_html = record[0]
                        name_soup = BeautifulSoup(name_html, "html.parser")
                        name = name_soup.get_text(strip=True)

                        # Extract link
                        link_tag = name_soup.find("a", href=True)
                        source_url = ""
                        if link_tag:
                            href = link_tag["href"]
                            if not href.startswith("http"):
                                href = f"https://efdsearch.senate.gov{href}"
                            source_url = href

                        # Filing date is usually in one of the later columns
                        filing_date = _parse_date(record[4]) if len(record) > 4 else None

                        trades.append({
                            "member_name": name,
                            "chamber": "senate",
                            "party": "",
                            "state": "",
                            "ticker": "",
                            "transaction_type": "",
                            "amount_low": "",
                            "amount_high": "",
                            "transaction_date": "",
                            "disclosure_date": filing_date or "",
                            "source_url": source_url,
                        })

                logger.info(f"Found {len(records)} Senate PTR filings for {year}")

            except (ValueError, KeyError) as e:
                logger.error(f"Failed to parse Senate API response: {e}")
        else:
            logger.warning(f"Senate search API returned {result.status_code}")

    except requests.RequestException as e:
        logger.error(f"Error scraping Senate disclosures: {e}")
    except Exception as e:
        logger.error(f"Unexpected error scraping Senate disclosures: {e}")

    return trades


def scrape_disclosures(
    year: int | None = None,
    output_dir: str | None = None,
    output_file: str | None = None,
) -> str:
    """
    Run the full scraping pipeline for both House and Senate disclosures.

    Outputs a CSV file at {output_dir}/{output_file} for user review.
    Returns the absolute path to the generated CSV.
    """
    out_dir = output_dir or DEFAULT_OUTPUT_DIR
    out_file = output_file or DEFAULT_OUTPUT_FILE
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, out_file)

    logger.info(f"Starting congressional disclosure scrape for year={year or 'current'}")

    all_trades = []

    # Scrape House
    house_trades = _scrape_house_disclosures(year=year)
    all_trades.extend(house_trades)

    # Scrape Senate
    senate_trades = _scrape_senate_disclosures(year=year)
    all_trades.extend(senate_trades)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_trades)

    logger.info(
        f"Wrote {len(all_trades)} rows to {csv_path} "
        f"(House: {len(house_trades)}, Senate: {len(senate_trades)})"
    )
    return csv_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Scrape congressional trading disclosures")
    parser.add_argument("--year", type=int, default=None, help="Filing year (default: current)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    out_dir = None
    out_file = None
    if args.output:
        out_dir = os.path.dirname(args.output) or DEFAULT_OUTPUT_DIR
        out_file = os.path.basename(args.output)

    path = scrape_disclosures(year=args.year, output_dir=out_dir, output_file=out_file)
    print(f"CSV written to: {path}")
