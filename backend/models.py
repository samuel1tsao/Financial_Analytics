from sqlalchemy import Column, Integer, String, Float, Boolean, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    questionnaires = relationship("Questionnaire", back_populates="user")
    portfolios = relationship("Portfolio", back_populates="user")
    favorites = relationship("Favorite", back_populates="user")


class Questionnaire(Base):
    __tablename__ = "questionnaires"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    schema_version = Column(String, default="v1")
    raw_json = Column(Text, nullable=False)  # Full JSON blob of structured answers
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="questionnaires")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    profile_name = Column(String, nullable=False, default="Default")
    is_current = Column(Boolean, default=False)
    weight_json = Column(Text, nullable=False)  # JSON blob of asset weights
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="portfolios")


class Favorite(Base):
    __tablename__ = "favorites"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    official_id = Column(String, nullable=False)  # Reference to public official

    user = relationship("User", back_populates="favorites")
