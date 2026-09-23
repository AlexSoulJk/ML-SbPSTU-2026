from __future__ import annotations

import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "Scripts" / "ResearchTool" / "Data" / "forest" / "state.sqlite"


def main() -> None:
    print(f"DB path: {DB_PATH}")
    print(f"Exists: {DB_PATH.exists()}")
    if not DB_PATH.exists():
        return

    size = DB_PATH.stat().st_size
    header = DB_PATH.read_bytes()[:128]
    print(f"Size: {size} bytes")
    print(f"Header bytes: {header[:32]!r}")

    if header.startswith(b"SQLite format 3\x00"):
        print("Header: SQLite format 3")
    elif header.startswith(b"version https://git-lfs.github.com/spec"):
        print("Header: Git LFS pointer, not a real database file")
    elif header.lstrip().startswith((b"<html", b"<!DOCTYPE", b"<?xml")):
        print("Header: looks like markup, not a SQLite database")
    else:
        print("Header: unknown/non-SQLite")

    try:
        with sqlite3.connect(DB_PATH) as connection:
            quick_check = connection.execute("pragma quick_check").fetchone()[0]
            tables = connection.execute(
                "select count(*) from sqlite_master where type = 'table'"
            ).fetchone()[0]
            print(f"SQLite quick_check: {quick_check}")
            print(f"SQLite tables: {tables}")
    except sqlite3.DatabaseError as exc:
        print(f"SQLite error: {exc}")


if __name__ == "__main__":
    main()
