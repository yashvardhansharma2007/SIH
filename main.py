"""
FastAPI backend for the Earth Observation Change & Risk Platform.

Endpoints:
  GET /api/regions                 -> list of monitored regions + latest reading + risk
  GET /api/regions/{region_id}      -> single region with full history + prediction
  GET /api/health                   -> liveness check

Run locally:
  pip install fastapi uvicorn python-dateutil --break-system-packages
  python data_gen.py          # (re)generate regions.json if needed
  uvicorn main:app --reload --port 8000

Then open frontend/index.html in a browser (it fetches from
http://localhost:8000 by default — edit API_BASE at the top of index.html
if you deploy the backend elsewhere).
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import forecast

DATA_PATH = Path(__file__).parent / "regions.json"
FALLBACK_PATH = Path(__file__).parent / "regions_fallback.json"

app = FastAPI(title="Earth Observation Change & Risk Platform")

# Wide-open CORS: fine for a hackathon demo where the frontend is a static
# HTML file opened locally. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_regions():
    """Load real GEE-sourced regions.json if present and healthy; otherwise
    fall back to the synthetic dataset so a bad pull never breaks the demo."""
    try:
        with open(DATA_PATH) as f:
            regions = json.load(f)["regions"]
        # A region with too many null months (cloud cover, failed reduceRegion
        # calls) is unusable for a forecast — treat that as "unhealthy".
        for r in regions:
            nulls = sum(1 for h in r["history"] if h["value"] is None)
            if nulls > len(r["history"]) * 0.3:
                raise ValueError(f"{r['id']} has too many null months ({nulls})")
        return regions
    except Exception as exc:
        if FALLBACK_PATH.exists():
            print(f"[warn] falling back to synthetic data: {exc}")
            with open(FALLBACK_PATH) as f:
                return json.load(f)["regions"]
        raise


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/regions")
def list_regions():
    regions = load_regions()
    out = []
    for r in regions:
        pred = forecast(r["history"])
        out.append({
            "id": r["id"],
            "name": r["name"],
            "lat": r["lat"],
            "lon": r["lon"],
            "kind": r["kind"],
            "index": r["index"],
            "current_value": pred["current_value"],
            "risk_tier": pred["risk_tier"],
        })
    return {"regions": out}


@app.get("/api/regions/{region_id}")
def get_region(region_id: str):
    regions = load_regions()
    region = next((r for r in regions if r["id"] == region_id), None)
    if region is None:
        raise HTTPException(status_code=404, detail="Region not found")

    pred = forecast(region["history"])
    return {
        "id": region["id"],
        "name": region["name"],
        "lat": region["lat"],
        "lon": region["lon"],
        "kind": region["kind"],
        "index": region["index"],
        "history": region["history"],
        "prediction": pred,
    }
