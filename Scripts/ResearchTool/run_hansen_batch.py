from __future__ import annotations

import sys
import time
from pathlib import Path

import os
from pathlib import Path

# Указываем PROJ на папку rasterio (у него свой proj.db)
_RASTERIO_PROJ = Path(r"C:\Users\Asus\ML-SbPSTU-2026\Lib\site-packages\rasterio\proj_data")
if _RASTERIO_PROJ.joinpath("proj.db").exists():
    os.environ["PROJ_LIB"] = str(_RASTERIO_PROJ)
    os.environ["PROJ_DATA"] = str(_RASTERIO_PROJ)
    # Убираем чужой GDAL_DATA от PostgreSQL, он мешает
    os.environ.pop("GDAL_DATA", None)
    print(f"PROJ_LIB → {_RASTERIO_PROJ}")

from sqlalchemy import func, select

from backend.forest.database import SessionLocal
from backend.forest.hansen_service import analyze_sample_hansen
from backend.forest.models import HansenAnalysis, Sample


def main() -> None:
    db = SessionLocal()

    done_ids = set(db.scalars(select(HansenAnalysis.sample_id).distinct()))
    pending = list(
        db.scalars(select(Sample).where(Sample.sample_id.notin_(done_ids)))
    )

    total = db.scalar(select(func.count()).select_from(Sample))
    print(f"Всего точек: {total}")
    print(f"Уже с Hansen: {len(done_ids)}")
    print(f"Осталось: {len(pending)}")
    print()

    if not pending:
        print("Нечего обрабатывать.")
        db.close()
        return

    start = time.monotonic()
    ok = failed = 0

    for i, sample in enumerate(pending, start=1):
        try:
            analyze_sample_hansen(db, sample, include_treecover=True)
            ok += 1
            status = "ok"
        except Exception as exc:
            db.rollback()
            failed += 1
            status = f"FAIL: {str(exc)[:120]}"

        elapsed = time.monotonic() - start
        rate = i / max(elapsed, 1e-6)
        eta_min = (len(pending) - i) / max(rate, 1e-6) / 60
        print(
            f"[{i}/{len(pending)}] "
            f"{sample.sample_id[:8]} {sample.driver_primary:25s} {status}  "
            f"({rate:.2f} т/сек, ETA {eta_min:.1f} мин)"
        )

    db.close()
    print()
    print(f"Готово. Успешно: {ok}, ошибок: {failed}")


if __name__ == "__main__":
    main()