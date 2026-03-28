from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import json
import models
import schemas
from database import get_db
from auth import get_current_user

router = APIRouter(prefix="/api/v1/questionnaire", tags=["questionnaire"])


@router.post("/save")
def save_questionnaire(
    payload: schemas.QuestionnaireSave,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save or overwrite the user's questionnaire answers (versioned)."""
    # Upsert: find existing for this user+version, or create new
    existing = (
        db.query(models.Questionnaire)
        .filter(
            models.Questionnaire.user_id == current_user.id,
            models.Questionnaire.schema_version == payload.schema_version,
        )
        .first()
    )

    raw = payload.answers.model_dump_json()

    if existing:
        existing.raw_json = raw
    else:
        q = models.Questionnaire(
            user_id=current_user.id,
            schema_version=payload.schema_version,
            raw_json=raw,
        )
        db.add(q)

    db.commit()
    return {"status": "saved", "schema_version": payload.schema_version}


@router.get("/current")
def get_current_questionnaire(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve the latest questionnaire answers so the frontend can pre-fill."""
    q = (
        db.query(models.Questionnaire)
        .filter(models.Questionnaire.user_id == current_user.id)
        .order_by(models.Questionnaire.created_at.desc())
        .first()
    )

    if not q:
        return {"exists": False, "answers": None}

    return {
        "exists": True,
        "schema_version": q.schema_version,
        "answers": json.loads(q.raw_json),
    }
