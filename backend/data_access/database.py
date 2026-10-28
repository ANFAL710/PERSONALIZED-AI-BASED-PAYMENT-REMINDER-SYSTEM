"""
Database Configuration & Session Management Module
AI Smart Vehicle Finance Reminder & Payment Monitoring System

Configures SQLite connection via SQLAlchemy and provides session lifecycles.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from backend.data_access.models import Base

# Calculate project paths safely
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "finance.db")

_ENGINE: Optional[Engine] = None
_SESSION_FACTORY: Optional[sessionmaker] = None


def get_db_path() -> str:
    """Returns the default SQLite database path, ensuring data directory exists."""
    data_dir = os.path.dirname(DEFAULT_DB_PATH)
    os.makedirs(data_dir, exist_ok=True)
    return DEFAULT_DB_PATH


def get_engine(db_path: Optional[str] = None, echo: bool = False) -> Engine:
    """
    Returns or initializes the SQLAlchemy engine for SQLite.
    """
    global _ENGINE
    if db_path is not None:
        target_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        return create_engine(f"sqlite:///{target_path}", echo=echo)

    if _ENGINE is None:
        path = get_db_path()
        _ENGINE = create_engine(f"sqlite:///{path}", echo=echo)
    return _ENGINE


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """
    Returns a configured sessionmaker bound to the given or default engine.
    """
    global _SESSION_FACTORY
    if engine is not None:
        return sessionmaker(bind=engine, autoflush=False, autocommit=False)
    if _SESSION_FACTORY is None:
        _SESSION_FACTORY = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SESSION_FACTORY


def get_session(engine: Optional[Engine] = None) -> Session:
    """
    Instantiates a new database session.
    """
    factory = get_session_factory(engine)
    return factory()


@contextmanager
def session_scope(engine: Optional[Engine] = None) -> Generator[Session, None, None]:
    """
    Transactional session scope context manager.
    Automatically commits on success or rolls back on exception.
    """
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine: Optional[Engine] = None) -> None:
    """
    Creates all tables in the SQLite database if they do not exist.
    """
    target_engine = engine or get_engine()
    Base.metadata.create_all(bind=target_engine)
