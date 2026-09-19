import rasterio
from pathlib import Path

rasterio_dir = Path(rasterio.__file__).parent
print("rasterio package:", rasterio_dir)

# Ищем proj.db внутри rasterio
found = list(rasterio_dir.rglob("proj.db"))
print(f"Найдено proj.db внутри rasterio: {len(found)}")
for p in found:
    print(" ", p)

# Ищем proj.db рядом с GDAL
gdal_dir = rasterio_dir / "gdal_data"
if gdal_dir.exists():
    print("gdal_data exists")
    for p in gdal_dir.rglob("proj.db"):
        print("  gdal proj.db:", p)

# Проверяем переменные окружения
import os
print()
print("PROJ_LIB:", os.environ.get("PROJ_LIB"))
print("PROJ_DATA:", os.environ.get("PROJ_DATA"))
print("GDAL_DATA:", os.environ.get("GDAL_DATA"))