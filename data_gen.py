"""
Generates synthetic NDVI / NDWI monthly time series that mimic what you'd
pull from Sentinel-2 via Google Earth Engine for a set of demo regions.

WHY SYNTHETIC DATA: this lets the whole platform run offline, instantly,
with zero API keys, which is exactly what you want for a hackathon demo
(no risk of GEE auth issues / rate limits / bad wifi on stage).

SWAPPING IN REAL DATA (do this before the actual SIH round if you have
GEE access working):
  1. Sign up at https://earthengine.google.com/ and get a service account.
  2. For each region's (lat, lon, bbox), pull a Sentinel-2 SR image
     collection filtered by date, cloud-mask it, and compute:
         NDVI = (NIR - RED) / (NIR + RED)   -> bands B8, B4
         NDWI = (GREEN - NIR) / (GREEN + NIR) -> bands B3, B8
  3. Reduce each monthly composite to a mean value over the region
     geometry, and append {date, ndvi, ndwi} to that region's history
     in regions.json — the rest of the pipeline (model.py, main.py,
     frontend) needs no changes because it only depends on that shape.

Run: python data_gen.py   -> writes regions.json
"""

import json
import random
from datetime import date
from dateutil.relativedelta import relativedelta

random.seed(42)

MONTHS = 24  # 2 years of monthly "satellite passes"


def month_series(start, months):
    return [start + relativedelta(months=i) for i in range(months)]


def gen_forest_decline(base=0.78, monthly_drop=0.014, noise=0.015):
    """Steady deforestation trend, e.g. encroachment/logging."""
    vals = []
    v = base
    for _ in range(MONTHS):
        v -= monthly_drop
        vals.append(round(max(0.05, v + random.uniform(-noise, noise)), 3))
    return vals


def gen_forest_stable(base=0.81, noise=0.02):
    """Healthy, roughly stable vegetation cover."""
    return [round(max(0.05, base + random.uniform(-noise, noise)), 3) for _ in range(MONTHS)]


def gen_flood_event(base=0.18, event_month=15, spike=0.55, recover_rate=0.06, noise=0.02):
    """NDWI series: mostly-dry baseline, sudden flood spike, partial recovery."""
    vals = []
    v = base
    for i in range(MONTHS):
        if i == event_month:
            v = base + spike
        elif i > event_month:
            v = max(base, v - recover_rate)
        vals.append(round(max(0.02, v + random.uniform(-noise, noise)), 3))
    return vals


def gen_early_warning_decline(base=0.70, monthly_drop=0.006, noise=0.02):
    """Slow, early-stage decline — the kind you WANT to catch with prediction
    before it becomes obvious in raw before/after imagery."""
    vals = []
    v = base
    for _ in range(MONTHS):
        v -= monthly_drop
        vals.append(round(max(0.05, v + random.uniform(-noise, noise)), 3))
    return vals


def build_region(region_id, name, lat, lon, kind, index_name, values, months):
    return {
        "id": region_id,
        "name": name,
        "lat": lat,
        "lon": lon,
        "kind": kind,  # "forest" | "water"
        "index": index_name,  # "ndvi" | "ndwi"
        "history": [
            {"date": m.isoformat(), "value": v} for m, v in zip(months, values)
        ],
    }


def main():
    start = date(2024, 8, 1)
    months = month_series(start, MONTHS)

    # Coordinates are representative points within each named belt/basin,
    # not precise boundaries — good enough for a demo map pin. Swap in
    # real polygon geometries (GEE FeatureCollections) when you wire up
    # actual Sentinel-2 pulls.
    regions = [
        # ---- Major forest chains ----
        build_region(
            "western-ghats-nilgiri", "Western Ghats — Nilgiri", 11.4064, 76.6932,
            "forest", "ndvi", gen_forest_stable(base=0.82), months,
        ),
        build_region(
            "western-ghats-agasthyamalai", "Western Ghats — Agasthyamalai", 8.6167, 77.2333,
            "forest", "ndvi", gen_early_warning_decline(base=0.75, monthly_drop=0.005), months,
        ),
        build_region(
            "eastern-ghats-andhra", "Eastern Ghats — Andhra Belt", 18.1124, 83.4091,
            "forest", "ndvi", gen_forest_decline(base=0.70, monthly_drop=0.010), months,
        ),
        build_region(
            "central-india-ranchi", "Central India — Ranchi Belt", 23.3441, 85.3096,
            "forest", "ndvi", gen_forest_decline(base=0.78, monthly_drop=0.014), months,
        ),
        build_region(
            "central-india-bastar", "Central India — Bastar Belt", 19.1071, 81.9550,
            "forest", "ndvi", gen_forest_decline(base=0.80, monthly_drop=0.011), months,
        ),
        build_region(
            "northeast-meghalaya", "Northeast — Meghalaya Hills", 25.4670, 91.3662,
            "forest", "ndvi", gen_forest_decline(base=0.83, monthly_drop=0.009), months,
        ),
        build_region(
            "northeast-arunachal", "Northeast — Arunachal Forest Belt", 27.7050, 93.6167,
            "forest", "ndvi", gen_forest_stable(base=0.85), months,
        ),
        build_region(
            "himalayan-uttarakhand", "Himalayan Foothills — Uttarakhand", 30.0668, 79.0193,
            "forest", "ndvi", gen_early_warning_decline(base=0.72, monthly_drop=0.006), months,
        ),
        build_region(
            "sundarbans-mangroves", "Sundarbans Mangroves", 21.9497, 88.9468,
            "forest", "ndvi", gen_early_warning_decline(base=0.70, monthly_drop=0.006), months,
        ),
        build_region(
            "aravalli-range", "Aravalli Range", 24.5854, 73.7125,
            "forest", "ndvi", gen_forest_decline(base=0.55, monthly_drop=0.009), months,
        ),

        # ---- Major flood-prone basins / deltas ----
        build_region(
            "kosi-basin-floodplain", "Kosi Basin Floodplain (Bihar)", 25.9358, 86.5910,
            "water", "ndwi", gen_flood_event(event_month=15, spike=0.55), months,
        ),
        build_region(
            "brahmaputra-floodplain", "Brahmaputra Floodplain (Assam)", 26.1445, 91.7362,
            "water", "ndwi", gen_flood_event(event_month=10, spike=0.60, recover_rate=0.05), months,
        ),
        build_region(
            "ganga-floodplain-bihar", "Ganga Floodplain (Bihar/UP)", 25.5941, 85.1376,
            "water", "ndwi", gen_flood_event(event_month=13, spike=0.45), months,
        ),
        build_region(
            "godavari-delta", "Godavari Delta (Andhra Pradesh)", 16.9891, 81.7800,
            "water", "ndwi", gen_flood_event(event_month=17, spike=0.40, recover_rate=0.08), months,
        ),
        build_region(
            "krishna-delta", "Krishna Delta (Andhra Pradesh)", 16.1730, 81.1350,
            "water", "ndwi", gen_flood_event(event_month=9, spike=0.42, recover_rate=0.07), months,
        ),
        build_region(
            "mahanadi-delta", "Mahanadi Delta (Odisha)", 20.2600, 86.7200,
            "water", "ndwi", gen_flood_event(event_month=12, spike=0.50, recover_rate=0.06), months,
        ),
    ]

    with open("regions.json", "w") as f:
        json.dump({"regions": regions}, f, indent=2)

    print(f"Wrote regions.json with {len(regions)} regions x {MONTHS} months each.")


if __name__ == "__main__":
    main()
