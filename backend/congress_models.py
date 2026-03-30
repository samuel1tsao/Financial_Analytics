"""
SQLAlchemy models for congressional trading data.

- CongressMember: basic information about a tracked member of congress
- CongressTrade: individual stock transaction disclosure
"""
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime


class CongressMember(Base):
    """A tracked member of the U.S. Congress."""
    __tablename__ = "congress_members"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    chamber = Column(String, nullable=False)  # "house" or "senate"
    party = Column(String, nullable=True)
    state = Column(String, nullable=True)
    
    # Cached portfolio stats built by profile_builder.py
    total_value = Column(Float, default=0.0)
    performance_1y = Column(Float, nullable=True)
    performance_5y = Column(Float, nullable=True)
    portfolio_weights = Column(JSON, default=dict)

    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trades = relationship("CongressTrade", back_populates="member", cascade="all, delete-orphan")
    portfolio_history = relationship("CongressPortfolioHistory", back_populates="member", cascade="all, delete-orphan", order_by="CongressPortfolioHistory.date")


class CongressTrade(Base):
    """A single stock transaction disclosed by a member of congress."""
    __tablename__ = "congress_trades"

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("congress_members.id"), nullable=False, index=True)
    ticker = Column(String, nullable=False, index=True)
    transaction_type = Column(String, nullable=False)  # "purchase", "sale", "exchange"
    amount_low = Column(Float, nullable=True)   # Lower bound of reported range
    amount_high = Column(Float, nullable=True)  # Upper bound of reported range
    transaction_date = Column(Date, nullable=True)
    disclosure_date = Column(Date, nullable=True)
    source_url = Column(String, nullable=True)

    member = relationship("CongressMember", back_populates="trades")

    __table_args__ = (
        Index("ix_congress_trades_member_ticker", "member_id", "ticker"),
    )


class CongressPortfolioHistory(Base):
    """Daily historical equity snapshot of a member's portfolio for charting."""
    __tablename__ = "congress_portfolio_history"

    id = Column(Integer, primary_key=True, index=True)
    member_id = Column(Integer, ForeignKey("congress_members.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    total_value = Column(Float, nullable=False)

    member = relationship("CongressMember", back_populates="portfolio_history")

    __table_args__ = (
        Index("ix_congress_portfolio_history_member_date", "member_id", "date", unique=True),
    )
