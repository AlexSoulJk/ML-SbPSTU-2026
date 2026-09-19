from __future__ import annotations

import json
from sqlalchemy import inspect, text
from backend.forest.database import engine


TABLES = [
    "forest_import_batches",
    "forest_samples",
    "forest_sample_assignments",
    "forest_sample_statuses",
    "forest_manual_validations",
    "forest_hansen_analyses",
    "forest_hansen_tiles",
    "forest_hansen_masks",
    "forest_sentinel_searches",
    "forest_sentinel_downloads",
    "forest_derived_previews",
    "forest_exports",
]

LIMIT = 3


def preview():
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())

    with engine.begin() as conn:
        for table in TABLES:
            if table not in existing:
                print(f"\n=== {table} === [НЕТ В БД]")
                continue

            total = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"\n=== {table} === всего строк: {total}")

            if total == 0:
                continue

            rows = conn.execute(
                text(f"SELECT * FROM {table} LIMIT {LIMIT}")
            ).mappings().all()

            for i, row in enumerate(rows, 1):
                print(f"  --- строка {i} ---")
                for key, value in row.items():
                    # Обрезаем длинные строки
                    if isinstance(value, str) and len(value) > 200:
                        value = value[:200] + "..."
                    print(f"    {key}: {value}")


if __name__ == "__main__":
    preview()