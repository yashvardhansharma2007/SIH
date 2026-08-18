"""
REAL satellite data fetch, using Google Earth Engine + Sentinel-2 SR Harmonized.

This is a drop-in replacement for data_gen.py — it writes regions.json in
the exact same {id, name, lat, lon, kind, index, history:[{date,value}]}
shape, so backend/main.py and backend/model.py need ZERO changes.

SETUP (do this once):
  1. Register a Cloud project for Earth Engine at:
     https://code.earthengine.google.com  (pick Noncommercial, any Gmail).
  2. Easiest run environment: Google Colab (earthengine-api preinstalled).
     Locally instead: pip install earthengine-api
  3. Authenticate once:
        import ee
        ee.Authenticate()
        ee.Initialize(project="YOUR-PROJECT-ID")
     (ee.Authenticate() opens a browser login; do it once, it's cached.)

RUN:
    python gee_fetch.py --project YOUR-PROJECT-ID

This takes a few minutes — it's doing 16 regions x 24 monthly composites,
each requiring a cloud-masked Sentinel-2 mosaic + a reduceRegion call.
"""

import argparse
import json
import time
from datetime import date

try:
    import ee
except ImportError:
    raise SystemExit(
        "earthengine-api not installed. Run: pip install earthengine-api"
    )

MONTHS = 24
START = date(2024, 8, 1)
BUFFER_METERS = 4000  # radius around each point — widen if you get null months

# Same regions as data_gen.py — id, display name, lat, lon, kind, index.
# kind/index decide whether we compute vegetation (NDVI) or water (NDWI).
REGIONS = [
    ("western-ghats-nilgiri", "Western Ghats — Nilgiri", 11.4064, 76.6932, "forest", "ndvi"),
    ("western-ghats-agasthyamalai", "Western Ghats — Agasthyamalai", 8.6167, 77.2333, "forest", "ndvi"),
    ("eastern-ghats-andhra", "Eastern Ghats — Andhra Belt", 18.1124, 83.4091, "forest", "ndvi"),
    ("central-india-ranchi", "Central India — Ranchi Belt", 23.3441, 85.3096, "forest", "ndvi"),
    ("central-india-bastar", "Central India — Bastar Belt", 19.1071, 81.9550, "forest", "ndvi"),
    ("northeast-meghalaya", "Northeast — Meghalaya Hills", 25.4670, 91.3662, "forest", "ndvi"),
    ("northeast-arunachal", "Northeast — Arunachal Forest Belt", 27.7050, 93.6167, "forest", "ndvi"),
    ("himalayan-uttarakhand", "Himalayan Foothills — Uttarakhand", 30.0668, 79.0193, "forest", "ndvi"),
    ("sundarbans-mangroves", "Sundarbans Mangroves", 21.9497, 88.9468, "forest", "ndvi"),
    ("aravalli-range", "Aravalli Range", 24.5854, 73.7125, "forest", "ndvi"),
    ("kosi-basin-floodplain", "Kosi Basin Floodplain (Bihar)", 25.9358, 86.5910, "water", "ndwi"),
    ("brahmaputra-floodplain", "Brahmaputra Floodplain (Assam)", 26.1445, 91.7362, "water", "ndwi"),
    ("ganga-floodplain-bihar", "Ganga Floodplain (Bihar/UP)", 25.5941, 85.1376, "water", "ndwi"),
    ("godavari-delta", "Godavari Delta (Andhra Pradesh)", 16.9891, 81.7800, "water", "ndwi"),
    ("krishna-delta", "Krishna Delta (Andhra Pradesh)", 16.1730, 81.1350, "water", "ndwi"),
    ("mahanadi-delta", "Mahanadi Delta (Odisha)", 20.2600, 86.7200, "water", "ndwi"),
]


def mask_clouds(img):
    """Cloud-mask Sentinel-2 SR using the Scene Classification (SCL) band.
    Keeps: 4=vegetation, 5=bare soil, 6=water, 7=unclassified, 11=snow.
    Drops: clouds, cloud shadow, cirrus, saturated/defective pixels."""
    scl = img.select("SCL")
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return img.updateMask(good)


def month_ranges(start, n_months):
    y, m = start.year, start.month
    out = []
    for _ in range(n_months):
        s = ee.Date.fromYMD(y, m, 1)
        e = s.advance(1, "month")
        out.append((s, e, date(y, m, 1)))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def fetch_region_history(lat, lon, index_name):
    point = ee.Geometry.Point(lon, lat)
    geom = point.buffer(BUFFER_METERS)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
        .map(mask_clouds)
    )

    history = []
    for start_d, end_d, py_date in month_ranges(START, MONTHS):
        monthly = collection.filterDate(start_d, end_d)
        composite = monthly.median()

        if index_name == "ndvi":
            index_img = composite.normalizedDifference(["B8", "B4"])
        else:  # ndwi
            index_img = composite.normalizedDifference(["B3", "B8"])

        try:
            stats = index_img.reduceRegion(
                reducer=ee.Reducer.mean(), geometry=geom, scale=20, maxPixels=1e9
            ).getInfo()
            value = stats.get("nd")
        except Exception as exc:
            print(f"    ! failed for {py_date}: {exc}")
            value = None

        history.append({"date": py_date.isoformat(), "value": round(value, 3) if value is not None else None})

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Your registered GEE project ID")
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    regions_out = []
    for i, (rid, name, lat, lon, kind, index_name) in enumerate(REGIONS, 1):
        print(f"[{i}/{len(REGIONS)}] Fetching {name} ({index_name.upper()})...")
        t0 = time.time()
        history = fetch_region_history(lat, lon, index_name)
        n_null = sum(1 for h in history if h["value"] is None)
        print(f"    done in {time.time()-t0:.1f}s, {n_null} null months")

        regions_out.append({
            "id": rid, "name": name, "lat": lat, "lon": lon,
            "kind": kind, "index": index_name, "history": history,
        })

    with open("regions.json", "w") as f:
        json.dump({"regions": regions_out}, f, indent=2)

    print(f"\nWrote regions.json with {len(regions_out)} regions from real Sentinel-2 data.")


if __name__ == "__main__":
    main()
