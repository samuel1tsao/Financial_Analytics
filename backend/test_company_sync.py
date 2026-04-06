from main import init_db
from market_data import sync_company_info, sync_historical_financials

if __name__ == "__main__":
    # Ensure company_info table is re-created
    init_db()
    # Trigger scraping
    sync_company_info()
    sync_historical_financials()
