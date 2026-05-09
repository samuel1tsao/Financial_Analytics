"""
Congressional Trade CSV Loader

Reads a user-verified CSV file of congressional trading disclosures and
upserts the data into the CongressMember and CongressTrade database tables.

Usage:
    # CLI
    python congress_loader.py data/congress_trades_verified.csv

    # As module
    from congress_loader import load_csv_to_db
    stats = load_csv_to_db("data/congress_trades_verified.csv")
"""
import csv
import logging
import os
import sys
from datetime import datetime

from sqlalchemy.orm import Session
from database import SessionLocal
from congress_models import CongressMember, CongressTrade

logger = logging.getLogger(__name__)


def _parse_date(date_str: str):
    """Parse a date string into a date object, or None."""
    if not date_str or not date_str.strip():
        return None
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def _parse_float(val: str) -> float | None:
    """Parse a string to float, or None."""
    if not val or not val.strip():
        return None
    try:
        return float(val.strip().replace(",", "").replace("$", ""))
    except ValueError:
        return None


def _get_or_create_member(
    db: Session,
    name: str,
    chamber: str,
    party: str = "",
    state: str = "",
) -> CongressMember:
    """Find an existing member by name+chamber, or create a new one."""
    member = (
        db.query(CongressMember)
        .filter(
            CongressMember.name == name,
            CongressMember.chamber == chamber,
        )
        .first()
    )
    if member:
        # Update fields if provided
        if party and not member.party:
            member.party = party
        if state and not member.state:
            member.state = state
        member.last_updated = datetime.utcnow()
        return member

    member = CongressMember(
        name=name,
        chamber=chamber,
        party=party or None,
        state=state or None,
    )
    db.add(member)
    db.flush()  # Assign ID
    return member


def load_csv_to_db(csv_path: str) -> dict:
    """
    Read a verified CSV and upsert records into the database.

    CSV must have columns:
        member_name, chamber, party, state, ticker, transaction_type,
        amount_low, amount_high, transaction_date, disclosure_date, source_url

    Returns a summary dict: {"members_created", "members_updated", "trades_inserted", "rows_skipped"}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    db = SessionLocal()
    stats = {
        "members_created": 0,
        "members_updated": 0,
        "trades_inserted": 0,
        "rows_skipped": 0,
    }

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                name = row.get("member_name", "").strip()
                chamber = row.get("chamber", "").strip().lower()
                ticker = row.get("ticker", "").strip().upper()

                # Skip rows with no name or no ticker
                if not name or not ticker:
                    stats["rows_skipped"] += 1
                    continue

                if chamber not in ("house", "senate"):
                    logger.warning(f"Row {row_num}: invalid chamber '{chamber}', skipping")
                    stats["rows_skipped"] += 1
                    continue

                # Get or create the member
                is_new = (
                    db.query(CongressMember)
                    .filter(CongressMember.name == name, CongressMember.chamber == chamber)
                    .first()
                ) is None

                member = _get_or_create_member(
                    db,
                    name=name,
                    chamber=chamber,
                    party=row.get("party", "").strip(),
                    state=row.get("state", "").strip(),
                )

                if is_new:
                    stats["members_created"] += 1
                else:
                    stats["members_updated"] += 1

                # Check for duplicate trade
                transaction_date = _parse_date(row.get("transaction_date", ""))
                existing_trade = (
                    db.query(CongressTrade)
                    .filter(
                        CongressTrade.member_id == member.id,
                        CongressTrade.ticker == ticker,
                        CongressTrade.transaction_date == transaction_date,
                        CongressTrade.transaction_type == row.get("transaction_type", "").strip().lower(),
                    )
                    .first()
                )

                if existing_trade:
                    stats["rows_skipped"] += 1
                    continue

                trade = CongressTrade(
                    member_id=member.id,
                    ticker=ticker,
                    transaction_type=row.get("transaction_type", "").strip().lower(),
                    amount_low=_parse_float(row.get("amount_low", "")),
                    amount_high=_parse_float(row.get("amount_high", "")),
                    transaction_date=transaction_date,
                    disclosure_date=_parse_date(row.get("disclosure_date", "")),
                    source_url=row.get("source_url", "").strip() or None,
                )
                db.add(trade)
                stats["trades_inserted"] += 1

        db.commit()
        logger.info(f"CSV load complete: {stats}")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to load CSV: {e}")
        raise
    finally:
        db.close()

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python congress_loader.py <path_to_verified_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    result = load_csv_to_db(csv_path)
    print(f"Load complete: {result}")
