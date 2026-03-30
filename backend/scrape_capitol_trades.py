import os
import sys
import time
import argparse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dateutil import parser as date_parser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import SessionLocal
from congress_models import CongressMember, CongressTrade


def get_or_create_member(db, name: str, chamber: str, party: str, state: str):
    member = db.query(CongressMember).filter(CongressMember.name == name).first()
    if not member:
        member = CongressMember(name=name, chamber=chamber, party=party, state=state)
        db.add(member)
        db.commit()
        db.refresh(member)
    else:
        # Update demographics if they were 'Unknown' before
        updated = False
        if member.chamber == "unknown" and chamber != "unknown":
            member.chamber = chamber
            updated = True
        if (not member.party or member.party == "Unknown") and party != "Unknown":
            member.party = party
            updated = True
        if (not member.state or member.state == "Unknown") and state != "Unknown":
            member.state = state
            updated = True
        
        if updated:
            db.commit()
            
    return member

def parse_amount_range(size_str: str):
    """Parses ranges like '1K–15K' into low/high floats."""
    if not size_str or size_str == "N/A":
        return None, None
    
    size_str = size_str.replace("$", "").replace(",", "")
    parts = size_str.split("–")
    if len(parts) == 1:
        parts = size_str.split("-")
        
    def _to_float(val):
        val = val.strip().upper()
        multiplier = 1
        if "M" in val:
            multiplier = 1000000
            val = val.replace("M", "")
        elif "K" in val:
            multiplier = 1000
            val = val.replace("K", "")
        try:
            return float(val) * multiplier
        except ValueError:
            return None

    if len(parts) == 2:
        return _to_float(parts[0]), _to_float(parts[1])
    return _to_float(parts[0]), None


def scrape_capitol_trades(pages=5, full_sync=False):
    db = SessionLocal()
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,webp,image/apng,*/*;q=0.8",
        "Referer": "https://www.capitoltrades.com/trades"
    }

    # Find the latest trade date we have in out DB (for Normal Sync)
    latest_trade_date_in_db = None
    if not full_sync:
        latest_trade = db.query(CongressTrade).order_by(CongressTrade.transaction_date.desc()).first()
        if latest_trade and latest_trade.transaction_date:
            latest_trade_date_in_db = latest_trade.transaction_date
            print(f"Normal Sync Mode: Will stop when encountering trades on or before {latest_trade_date_in_db}.")
    else:
        print("Full Sync Mode: Replacing/upserting trades without early stopping.")

    base_url = "https://www.capitoltrades.com/trades?page={}"
    
    total_new_trades = 0
    stop_scraping = False

    for p in range(1, pages + 1):
        if stop_scraping:
            break
            
        print(f"📡 Scraping page {p}...", end=" ")
        try:
            response = requests.get(base_url.format(p), headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('tr')
            
            page_trades_added = 0
            
            # The first row is usually the table header
            for row in rows:
                ticker_el = row.select_one('.issuer-ticker')
                if not ticker_el or ":US" not in ticker_el.text:
                    continue
                
                ticker = ticker_el.text.replace(":US", "").strip()
                
                # Politician Name
                name_el = row.select_one('.politician-name')
                politician = name_el.text.strip() if name_el else "Unknown"
                
                # Demographics Parse (from user-provided CSS classes)
                party_el = row.select_one('.party')
                chamber_el = row.select_one('.chamber')
                state_el = row.select_one('.us-state-compact')
                
                party = party_el.text.strip() if party_el else "Unknown"
                chamber = chamber_el.text.strip().lower() if chamber_el else "unknown"
                state = state_el.text.strip() if state_el else "Unknown"

                # Trade Date & Published Date (Columns 3 & 4 usually)
                tds = row.find_all('td')
                if len(tds) < 5:
                    continue
                
                # Date format is often stacked div e.g. "26 Mar\n2026"
                traded_text = tds[2].get_text(" ").strip()
                published_text = tds[3].get_text(" ").strip()
                
                try:
                    traded_date = date_parser.parse(traded_text).date()
                except Exception:
                    continue
                
                try:
                    published_date = date_parser.parse(published_text).date()
                except Exception:
                    published_date = None

                # Normal Sync Stop Condition
                if not full_sync and latest_trade_date_in_db:
                    if traded_date <= latest_trade_date_in_db:
                        print(f"🛑 Reached known date {traded_date}. Stopping sync.")
                        stop_scraping = True
                        break

                # Extract the rest
                type_el = row.select_one('.tx-type')
                tx_type = type_el.text.strip().lower() if type_el else "unknown"
                if "buy" in tx_type or "purchase" in tx_type:
                    tx_type = "purchase"
                elif "sell" in tx_type or "sale" in tx_type:
                    tx_type = "sale"

                size_el = row.select_one('.trade-size')
                size_str = size_el.get_text(" ").strip() if size_el else "N/A"
                amount_low, amount_high = parse_amount_range(size_str)
                
                # Get Member
                member = get_or_create_member(db, name=politician, chamber=chamber, party=party, state=state)

                # Check if trade already exists
                existing_trade = db.query(CongressTrade).filter(
                    CongressTrade.member_id == member.id,
                    CongressTrade.ticker == ticker,
                    CongressTrade.transaction_date == traded_date,
                    CongressTrade.transaction_type == tx_type
                ).first()

                if not existing_trade:
                    # Insert New Trade
                    trade = CongressTrade(
                        member_id=member.id,
                        ticker=ticker,
                        transaction_type=tx_type,
                        amount_low=amount_low,
                        amount_high=amount_high,
                        transaction_date=traded_date,
                        disclosure_date=published_date,
                        source_url=f"https://www.capitoltrades.com/trades?politician={member.id}"
                    )
                    db.add(trade)
                    page_trades_added += 1

            db.commit()
            total_new_trades += page_trades_added
            print(f"✅ Found {page_trades_added} new trades.")
            
            # If a page returns 0 new trades and we're not stopping early, it means page was blank
            if page_trades_added == 0 and not stop_scraping and len(rows) <= 1:
                print("🛑 No more data on website. Stopping.")
                break
                
        except Exception as e:
            print(f"❌ Error on page {p}: {e}")
        
        # POLITE DELAY to prevent overwhelming capitoltrades.com
        time.sleep(2)

    db.close()
    return total_new_trades


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Politician scraper from CapitolTrades.")
    parser.add_argument("--pages", type=int, default=5, help="Number of pages to scrape.")
    parser.add_argument("--full", action="store_true", help="Full sync mode (ignore last DB date).")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("🏛️ Scraping CapitolTrades...")
    print("="*50)
    
    total = scrape_capitol_trades(pages=args.pages, full_sync=args.full)
    print(f"\n🎉 Sync complete! Total new trades added: {total}")
