import sys
from pathlib import Path

from collections import Counter
from sqlalchemy import select
from backend.forest.database import SessionLocal
from backend.forest.models import HansenAnalysis, Sample

db = SessionLocal()
samples = {s.sample_id: s for s in db.scalars(select(Sample))}
analyses = list(db.scalars(select(HansenAnalysis)))

statuses = Counter(a.status for a in analyses)
print("Статусы Hansen:")
for s, c in statuses.most_common():
    print(f"  {s}: {c}")
print()

years = Counter(a.event_year for a in analyses if a.event_year)
print("Топ-15 годов событий:")
for y, c in sorted(years.items(), reverse=True)[:15]:
    print(f"  {y}: {c}")
print()

by_driver = {}
for a in analyses:
    s = samples.get(a.sample_id)
    if s is None or a.dominant_year_share is None:
        continue
    by_driver.setdefault(s.driver_primary, []).append(a.dominant_year_share)

print("Средний dominant_share по драйверам:")
for d, vals in sorted(by_driver.items(), key=lambda x: -len(x[1])):
    print(f"  {d:30s} {sum(vals)/len(vals):.3f}  (n={len(vals)})")

db.close()