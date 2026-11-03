from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from Scripts.ResearchTool.backend.geospatial_env import configure_geospatial_environment


class GeospatialEnvironmentTest(unittest.TestCase):
    def test_rasterio_data_overrides_incompatible_system_paths(self) -> None:
        postgis_root = r"C:\Program Files\PostgreSQL\16\share"
        with patch.dict(
            os.environ,
            {
                "PROJ_LIB": str(Path(postgis_root) / "contrib" / "postgis-3.4" / "proj"),
                "PROJ_DATA": "",
                "GDAL_DATA": str(Path(postgis_root) / "gdal-data"),
            },
            clear=False,
        ):
            configured = configure_geospatial_environment()

            self.assertTrue(Path(configured["PROJ_DATA"], "proj.db").is_file())
            self.assertEqual(os.environ["PROJ_LIB"], configured["PROJ_DATA"])
            self.assertTrue(Path(configured["GDAL_DATA"]).is_dir())

            from rasterio.crs import CRS

            self.assertEqual(CRS.from_epsg(4326).to_epsg(), 4326)


if __name__ == "__main__":
    unittest.main()
