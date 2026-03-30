import os
import sys
import logging
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import SessionLocal
from congress_models import CongressMember, CongressTrade, CongressPortfolioHistory
from market_models import StockPrice

logger = logging.getLogger(__name__)

def load_price_matrix(db):
    """
    Loads all StockPrice data into a single Pandas DataFrame.
    Index: Date, Columns: Ticker, Values: adj_close.
    Forward fills missing weekend/holiday data.
    """
    prices = db.query(StockPrice.date, StockPrice.ticker, StockPrice.adj_close).all()
    if not prices:
        return pd.DataFrame()
    
    df = pd.DataFrame(prices, columns=["date", "ticker", "adj_close"])
    matrix = df.pivot(index="date", columns="ticker", values="adj_close")
    
    # Resample to daily, forward filling missing prices up to today
    today = pd.Timestamp.today().normalize()
    matrix.index = pd.to_datetime(matrix.index)
    idx = pd.date_range(start=matrix.index.min(), end=max(matrix.index.max(), today), freq='D')
    matrix = matrix.reindex(idx).ffill()
    return matrix


def build_all_profiles():
    logger.info("Starting Profile Builder...")
    db = SessionLocal()
    
    try:
        price_matrix = load_price_matrix(db)
        if price_matrix.empty:
            logger.warning("No market data available to build profiles. Run sync_market_data first.")
            return
            
        members = db.query(CongressMember).all()
        print(f"Aggregating portfolios for {len(members)} representatives...")

        history_inserts = []
        today = pd.Timestamp.today().normalize()
        
        # Clear existing history to replace it freshly (or we could selectively update)
        db.query(CongressPortfolioHistory).delete()

        for member in members:
            trades = sorted(member.trades, key=lambda x: x.transaction_date)
            if not trades:
                continue
                
            first_trade_date = pd.Timestamp(trades[0].transaction_date)
            
            # Pre-pass: Determine minimum starting balances to prevent sells dropping positions below 0
            starting_positions = {}
            temp_pos = {}
            for t in trades:
                ticker = t.ticker
                t_date = pd.Timestamp(t.transaction_date)
                
                # Get price on transaction date
                price = None
                if ticker in price_matrix.columns:
                    try:
                        price_val = price_matrix.loc[t_date, ticker]
                        if not pd.isna(price_val):
                            price = price_val
                        else:
                            valid_prices = price_matrix.loc[:t_date, ticker].dropna()
                            if not valid_prices.empty:
                                price = valid_prices.iloc[-1]
                    except KeyError:
                        pass
                if not price:
                    price = 1.0
                
                amount = (t.amount_low + t.amount_high) / 2 if (t.amount_low and t.amount_high) else (t.amount_low or 1000.0)
                
                if t.transaction_type in ("purchase", "buy"):
                    shares = amount / price
                    temp_pos[ticker] = temp_pos.get(ticker, 0.0) + shares
                elif t.transaction_type in ("sale", "sell"):
                    shares_sold = amount / price
                    current = temp_pos.get(ticker, 0.0)
                    if shares_sold > current:
                        deficit = shares_sold - current
                        starting_positions[ticker] = starting_positions.get(ticker, 0.0) + deficit
                        temp_pos[ticker] = 0.0
                    else:
                        temp_pos[ticker] -= shares_sold
            
            # Initialize positions with the calculated retro-active starting balance
            positions = dict(starting_positions)
            daily_history = []
            
            trade_idx = 0
            
            # Ensure the simulation starts at least 30 days prior to their first trade
            # This allows the retroactive starting balance to be plotted before the first 'sell' nuke.
            current_date = first_trade_date - relativedelta(days=30)
            while current_date <= today:
                # 1. Process any trades on this day
                while trade_idx < len(trades) and pd.Timestamp(trades[trade_idx].transaction_date) <= current_date:
                    t = trades[trade_idx]
                    ticker = t.ticker
                    
                    # Find closest previous price if exact day doesn't exist
                    price = None
                    if ticker in price_matrix.columns:
                        try:
                            price_val = price_matrix.loc[current_date, ticker]
                            if not pd.isna(price_val):
                                price = price_val
                            else:
                                valid_prices = price_matrix.loc[:current_date, ticker].dropna()
                                if not valid_prices.empty:
                                    price = valid_prices.iloc[-1]
                        except KeyError:
                            pass
                            
                    # If we don't track this ticker's price, fallback to 1.0 (cash equivalent tracking)
                    if not price:
                        price = 1.0

                    # Use the estimated average trade size to match the frontend calculation
                    if t.amount_low and t.amount_high:
                        amount = (t.amount_low + t.amount_high) / 2
                    else:
                        amount = t.amount_low or 1000.0
                    
                    if t.transaction_type in ("purchase", "buy"):
                        shares = amount / price
                        positions[ticker] = positions.get(ticker, 0.0) + shares
                    elif t.transaction_type in ("sale", "sell"):
                        # If selling, calculate shares sold at standard low amount. 
                        # To avoid negative positions going underwater, we max(0).
                        shares_sold = amount / price
                        current_shares = positions.get(ticker, 0.0)
                        positions[ticker] = max(0.0, current_shares - shares_sold)
                        
                        # Cleanup empty positions
                        if positions[ticker] == 0.0:
                            del positions[ticker]
                            
                    trade_idx += 1

                # 2. Evaluate portfolio on this day
                daily_value = 0.0
                daily_weights = {}
                for tick, shares in list(positions.items()):
                    price = 1.0
                    if tick in price_matrix.columns:
                        try:
                            price_val = price_matrix.loc[current_date, tick]
                            if not pd.isna(price_val):
                                price = price_val
                        except KeyError:
                            pass
                            
                    position_value = shares * price
                    daily_value += position_value
                    daily_weights[tick] = position_value
                
                if daily_value > 0:
                    history_inserts.append(
                        CongressPortfolioHistory(
                            member_id=member.id,
                            date=current_date.date(),
                            total_value=daily_value
                        )
                    )
                    
                    # If this is today (or the last loop), update Member snapshot
                    if current_date == today:
                        member.total_value = daily_value
                        
                        # Calculate weights %
                        if daily_value > 0:
                            for k in daily_weights:
                                daily_weights[k] = daily_weights[k] / daily_value
                        member.portfolio_weights = daily_weights
                
                current_date += relativedelta(days=1)

            # Calculate historical returns
            # Extract the member's daily values specifically
            member_ts = {h.date: h.total_value for h in history_inserts if h.member_id == member.id}
            
            if member_ts and member.total_value > 0:
                one_yr_ago = (today - relativedelta(years=1)).date()
                five_yr_ago = (today - relativedelta(years=5)).date()
                
                val_1y = member_ts.get(one_yr_ago)
                # If exact date missing, find closest after
                if not val_1y:
                    dates_1y = [d for d in member_ts.keys() if d >= one_yr_ago]
                    val_1y = member_ts[min(dates_1y)] if dates_1y else None

                val_5y = member_ts.get(five_yr_ago)
                if not val_5y:
                    dates_5y = [d for d in member_ts.keys() if d >= five_yr_ago]
                    val_5y = member_ts[min(dates_5y)] if dates_5y else None

                member.performance_1y = (member.total_value / val_1y - 1.0) if val_1y and val_1y > 0 else None
                member.performance_5y = (member.total_value / val_5y - 1.0) if val_5y and val_5y > 0 else None

        # Bulk Insert History
        db.bulk_save_objects(history_inserts)
        db.commit()
        logger.info(f"Generated {len(history_inserts)} historical data points across all members.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during profile building: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    build_all_profiles()
