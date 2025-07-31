"""
Database configuration for the Agent Team Builder Service.

This module provides SQLAlchemy session and engine setup.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import the shared declarative Base with naming conventions
from .models.base import Base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get database connection parameters from environment variables
PG_DSN = os.getenv("PG_DSN", "postgresql://postgres_user:postgres_pass@localhost:5432/agentteambuilder")

# Create SQLAlchemy engine
engine = create_engine(PG_DSN)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)




def get_db():
    """
    Return a database session.
    
    This function is designed to be used as a FastAPI dependency.
    It yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
