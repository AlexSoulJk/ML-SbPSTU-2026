from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from ..config import FOREST_STATE_DB_PATH
from .models import Base


def sqlite_url() -> str:
    return f"sqlite:///{FOREST_STATE_DB_PATH.as_posix()}"


engine = create_engine(
    sqlite_url(),
    connect_args={"check_same_thread": False},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_forest_database() -> None:
    FOREST_STATE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    migrate_forest_database()


def migrate_forest_database() -> None:
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    if "forest_samples" not in table_names:
        return

    sample_columns = {column["name"] for column in inspector.get_columns("forest_samples")}
    sentinel_download_columns = (
        {column["name"] for column in inspector.get_columns("forest_sentinel_downloads")}
        if "forest_sentinel_downloads" in table_names
        else set()
    )
    with engine.begin() as connection:
        if "display_name" not in sample_columns:
            connection.execute(text("ALTER TABLE forest_samples ADD COLUMN display_name VARCHAR(255)"))
        if "has_multiple_events" not in sample_columns:
            connection.execute(
                text("ALTER TABLE forest_samples ADD COLUMN has_multiple_events BOOLEAN NOT NULL DEFAULT 0")
            )
        if "nodata_fraction" not in sentinel_download_columns:
            connection.execute(text("ALTER TABLE forest_sentinel_downloads ADD COLUMN nodata_fraction FLOAT"))
        if "dark_fraction" not in sentinel_download_columns:
            connection.execute(text("ALTER TABLE forest_sentinel_downloads ADD COLUMN dark_fraction FLOAT"))
        if "is_bad_quality" not in sentinel_download_columns:
            connection.execute(
                text("ALTER TABLE forest_sentinel_downloads ADD COLUMN is_bad_quality BOOLEAN NOT NULL DEFAULT 0")
            )
        if "quality_flags_json" not in sentinel_download_columns:
            connection.execute(text("ALTER TABLE forest_sentinel_downloads ADD COLUMN quality_flags_json TEXT"))


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
