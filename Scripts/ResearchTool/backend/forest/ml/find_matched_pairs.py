from __future__ import annotations

import sys
from pathlib import Path

_here = Path(__file__).resolve()
for _p in _here.parents:
    if (_p / "backend").exists():
        sys.path.insert(0, str(_p))
        break

import numpy as np
import pandas as pd
from sqlalchemy import select, text

from backend.forest.database import SessionLocal
from backend.forest.models import HansenAnalysis, Sample


VALID_LABEL = "Valid"
DISTANCES_KM = (100, 250, 500, 1000, 2000)
YEAR_DIFFS = (2, 5)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_data() -> pd.DataFrame:
    with SessionLocal() as db:
        valid_ids = {
            r[0] for r in db.execute(
                text("SELECT DISTINCT sample_id FROM forest_manual_validations WHERE validation = :v"),
                {"v": VALID_LABEL},
            ).all()
        }

        latest: dict[str, HansenAnalysis] = {}
        for a in db.scalars(select(HansenAnalysis).order_by(HansenAnalysis.created_at)):
            if a.sample_id in valid_ids:
                latest[a.sample_id] = a

        rows = []
        for s in db.scalars(select(Sample)):
            if s.sample_id not in valid_ids:
                continue
            if s.driver_primary not in ("Wildfire", "Logging"):
                continue
            a = latest.get(s.sample_id)
            if a is None or not a.event_year:
                continue
            rows.append({
                "sample_id": s.sample_id,
                "driver": s.driver_primary,
                "lat": s.lat,
                "lon": s.lon,
                "year": int(a.event_year),
            })
    return pd.DataFrame(rows)


def count_pairs(wildfires: pd.DataFrame, loggings: pd.DataFrame,
                max_distance_km: float, max_year_diff: int) -> tuple[int, int, list[dict]]:
    """Возвращает (общее число пар, число уникальных пар после жадного отбора, список пар)."""
    pairs = []
    for _, w in wildfires.iterrows():
        dists = haversine_km(w["lat"], w["lon"], loggings["lat"], loggings["lon"])
        for idx, d in zip(loggings.index, dists):
            if d > max_distance_km:
                continue
            l = loggings.loc[idx]
            if abs(w["year"] - l["year"]) > max_year_diff:
                continue
            pairs.append({
                "wildfire_id": w["sample_id"],
                "logging_id": l["sample_id"],
                "distance_km": float(d),
                "wildfire_year": int(w["year"]),
                "logging_year": int(l["year"]),
                "year_diff": abs(int(w["year"]) - int(l["year"])),
            })

    if not pairs:
        return 0, 0, []

    pdf = pd.DataFrame(pairs).sort_values("distance_km").reset_index(drop=True)
    used_w, used_l = set(), set()
    unique_pairs = []
    for _, row in pdf.iterrows():
        if row["wildfire_id"] in used_w or row["logging_id"] in used_l:
            continue
        unique_pairs.append(row.to_dict())
        used_w.add(row["wildfire_id"])
        used_l.add(row["logging_id"])

    return len(pairs), len(unique_pairs), unique_pairs


def main() -> None:
    df = load_data()
    print(f"Всего Valid Wildfire/Logging с годом: {len(df)}")
    print(df["driver"].value_counts().to_string())
    print()

    wildfires = df[df["driver"] == "Wildfire"].reset_index(drop=True)
    loggings = df[df["driver"] == "Logging"].reset_index(drop=True)
    print(f"Wildfire: {len(wildfires)}, Logging: {len(loggings)}\n")

    print("=== Число пар при разных порогах ===")
    print(f"{'distance':>10s}  {'Δгод':>5s}  {'всего пар':>10s}  {'уникальных':>11s}")
    print("-" * 45)

    results = []
    for dist in DISTANCES_KM:
        for dy in YEAR_DIFFS:
            n_all, n_unique, uniq_pairs = count_pairs(wildfires, loggings, dist, dy)
            print(f"{dist:>10d}  {dy:>5d}  {n_all:>10d}  {n_unique:>11d}")
            results.append({
                "distance_km": dist,
                "year_diff": dy,
                "pairs_total": n_all,
                "pairs_unique": n_unique,
                "unique_pair_list": uniq_pairs,
            })

    # Сохранить сводку
    from backend.config import FOREST_MODELS_DIR
    FOREST_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "unique_pair_list"} for r in results])
    summary_path = FOREST_MODELS_DIR / "matched_pairs_thresholds.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[ok] сводка по порогам: {summary_path}")

    # Самый большой набор уникальных пар — показать его подробно
    best = max(results, key=lambda r: r["pairs_unique"])
    print(f"\n=== Наибольший набор уникальных пар ===")
    print(f"Порог: distance ≤ {best['distance_km']} км, Δгод ≤ {best['year_diff']}")
    print(f"Уникальных пар: {best['pairs_unique']}")

    if best["unique_pair_list"]:
        best_df = pd.DataFrame(best["unique_pair_list"])
        best_path = FOREST_MODELS_DIR / "matched_pairs.csv"
        best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
        print(f"[ok] список пар: {best_path}")
        print()
        print(best_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()