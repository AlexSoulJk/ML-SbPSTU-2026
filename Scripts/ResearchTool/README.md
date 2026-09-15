# Satellite Pipeline Explorer

Local research viewer for pipeline protection-zone KML and Sentinel-2 L2A
observations from Copernicus Data Space.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r Scripts\ResearchTool\requirements.txt
.\.venv\Scripts\python.exe Scripts\ResearchTool\run_local.py
```
.\.venv\Scripts\python.exe Scripts\ResearchTool\run_local.py --port 8010
Open:

```text
http://127.0.0.1:8000
```

If port 8000 is already busy, the runner automatically picks the next free
port and prints the actual URL.

If port 8000 is busy:

```powershell
.\.venv\Scripts\python.exe Scripts\ResearchTool\run_local.py --port 8010
```

For frontend/backend editing with auto-reload:

```powershell
.\.venv\Scripts\python.exe Scripts\ResearchTool\run_local.py --reload
```

CDSE credentials are read from the project root `.env`:

```text
CDSE_SH_CLIENT_ID=...
CDSE_SH_CLIENT_SECRET=...
CDSE_USE_ENV_PROXY=0
```

`CDSE_USE_ENV_PROXY=0` keeps CDSE requests from inheriting broken
`HTTP_PROXY`/`HTTPS_PROXY` values such as `127.0.0.1:9`. Set it to `1` only
when you intentionally use a working system proxy.

## MVP Scope

- KML boundary display.
- Closed LineString can be treated as a polygon.
- Approximate centerline via raster medial axis.
- 2 km centerline segments.
- Square AOI around a selected segment with 200 m or 500 m context.
- Sentinel-2 L2A search through CDSE STAC.
- Sentinel-2 raster fetch through Sentinel Hub Process API.
- RGB, false color, NDVI, NDMI previews.

The generated centerline is an approximation for exploration, not a real
pipeline geometry.
