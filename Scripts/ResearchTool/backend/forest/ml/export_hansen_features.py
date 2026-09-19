from __future__ import annotations

import json
import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _parent in _here.parents:
    if (_parent / "backend").exists():
        sys.path.insert(0, str(_parent))
        break

import pandas as pd
from sqlalchemy import select

from backend.config import FOREST_FEATURES_DIR
from backend.forest.database import SessionLocal
from backend.forest.models import HansenAnalysis, Sample

def export() -> pd.DataFrame:
    db = SessionLocal()
    samples = {s.sample_id: s for s in db.scalars(select(Sample))}

    by_sample = {}
    for a in db.scalars(select(HansenAnalysis).order_by(HansenAnalysis.created_at)):
        by_sample[a.sample_id] = a

    rows = []
    for sample_id, a in by_sample.items():
        s = samples.get(sample_id)
        if s is None:
            continue

        hist = json.loads(a.histogram_json or "[]")
        plot_area_ha = (a.plot_size_m ** 2) / 10_000.0

        loss_fraction = (
            (a.total_loss_area_ha or 0.0) / plot_area_ha if plot_area_ha > 0 else 0.0
        )

        max_year = max((h["year"] for h in hist), default=0)
        cutoff = max_year - 5
        recent_ha = sum(h["loss_area_ha"] for h in hist if h["year"] > cutoff)
        recent_fraction = recent_ha / plot_area_ha if plot_area_ha > 0 else 0.0

        rows.append({
            "sample_id": s.sample_id,
            "lat": s.lat,
            "lon": s.lon,
            "driver_primary": s.driver_primary,
            "confidence_primary": s.confidence_primary,
            "region": s.region,
            "source_file": s.source_file,
            "hansen_event_year": a.event_year,
            "hansen_total_loss_ha": a.total_loss_area_ha,
            "hansen_dominant_share": a.dominant_year_share,
            "hansen_loss_fraction": round(loss_fraction, 4),
            "hansen_recent_loss_5y": round(recent_fraction, 4),
            "hansen_status": a.status,
        })

    db.close()
    df = pd.DataFrame(rows)

    out = FOREST_FEATURES_DIR / "hansen_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"[ok] {out} — {len(df)} строк")
    print()
    print("=== Классы ===")
    print(df["driver_primary"].value_counts())
    print()
    print("=== Статусы ===")
    print(df["hansen_status"].value_counts())
    print()
    print("=== Средние по драйверам ===")
    print(df.groupby("driver_primary")[[
        "hansen_loss_fraction",
        "hansen_recent_loss_5y",
        "hansen_dominant_share",
    ]].mean().round(3))
    return df


if __name__ == "__main__":
    export()