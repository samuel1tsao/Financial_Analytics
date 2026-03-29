from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.orm import Session
import json
import os
import shutil
import models
from database import get_db
from auth import get_current_user
from officials_service import get_all_officials, get_official_by_id

router = APIRouter(prefix="/api/v1", tags=["officials"])


@router.get("/officials")
def list_officials(
    chamber: str | None = Query(None, description="Filter by chamber: house or senate"),
    party: str | None = Query(None, description="Filter by party: democrat or republican"),
):
    """Return all tracked public officials and their portfolio data."""
    officials = get_all_officials(chamber=chamber, party=party)
    return {"officials": officials}


@router.get("/officials/{official_id}")
def get_official(official_id: str):
    """Return a single official's full profile."""
    official = get_official_by_id(official_id)
    if not official:
        raise HTTPException(status_code=404, detail="Official not found")
    return official


@router.post("/officials/{official_id}/mimic")
def mimic_official(
    official_id: str,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Copy an official's portfolio as a new saved profile for the user.
    Creates a portfolio named 'Mimic: {official_name}'.
    """
    official = get_official_by_id(official_id)
    if not official:
        raise HTTPException(status_code=404, detail="Official not found")

    portfolio = models.Portfolio(
        user_id=current_user.id,
        profile_name=f"Mimic: {official['name']}",
        is_current=False,
        weight_json=json.dumps(official["portfolio"]),
    )
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return {
        "id": portfolio.id,
        "profile_name": portfolio.profile_name,
        "weights": official["portfolio"],
    }


@router.post("/officials/scrape")
def trigger_scrape(
    year: int | None = Query(None, description="Filing year (default: current)"),
    current_user: models.User = Depends(get_current_user),
):
    """
    Trigger the AI-assisted congressional disclosure scraper.
    Returns the path to the generated CSV for user review.
    The CSV is NOT auto-loaded into the database.
    """
    from congress_scraper import scrape_disclosures

    try:
        csv_path = scrape_disclosures(year=year)
        return {
            "status": "complete",
            "csv_path": csv_path,
            "message": "CSV generated. Review the file and upload via POST /officials/load when verified.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")


@router.post("/officials/load")
def load_verified_csv(
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
):
    """
    Upload a user-verified CSV of congressional trades and load into the database.
    The CSV should have columns:
    member_name, chamber, party, state, ticker, transaction_type,
    amount_low, amount_high, transaction_date, disclosure_date, source_url
    """
    from congress_loader import load_csv_to_db

    # Save uploaded file temporarily
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    temp_path = os.path.join(data_dir, "upload_verified.csv")

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        stats = load_csv_to_db(temp_path)
        return {
            "status": "loaded",
            "stats": stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV load failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
