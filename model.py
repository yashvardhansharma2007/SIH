"""
The AI layer: trend forecasting + risk classification on top of the
NDVI/NDWI time series for a region.

Kept deliberately lightweight (numpy linear regression, no heavy deps)
so it trains/runs instantly on CPU and is easy to explain to judges:
"we fit a trend to the last N satellite passes and classify the region
into a risk tier based on the rate and direction of change."

If you want a fancier model before the real event, this is the one
function to swap out: forecast(history) -> keep the same return shape
and everything upstream (API, frontend) keeps working. Good drop-in
upgrades: Prophet (facebook/prophet) for seasonality-aware forecasting,
or a small sklearn GradientBoostingRegressor on lag features.
"""

import numpy as np


def forecast(history, months_ahead=6, lookback=12):
    """
    history: list of {"date": iso_str, "value": float}, chronological.
    Returns dict with slope (per month), current value, predicted value
    N months ahead, percent change, and a risk tier.
    """
    # Filter out any null months (e.g. cloud-covered passes in real data)
    # before fitting the trend line.
    clean = [h for h in history[-lookback:] if h["value"] is not None]
    values = np.array([h["value"] for h in clean], dtype=float)
    x = np.arange(len(values))

    # Simple linear regression (least squares) — the trend line.
    slope, intercept = np.polyfit(x, values, 1)

    current_value = float(values[-1])
    predicted_value = float(slope * (len(values) - 1 + months_ahead) + intercept)
    predicted_value = max(0.0, predicted_value)

    pct_change = (
        ((predicted_value - current_value) / current_value) * 100
        if current_value != 0
        else 0.0
    )

    risk_tier, message = _classify_risk(slope, pct_change)

    return {
        "slope_per_month": round(float(slope), 5),
        "current_value": round(current_value, 3),
        "predicted_value": round(predicted_value, 3),
        "months_ahead": months_ahead,
        "pct_change": round(pct_change, 1),
        "risk_tier": risk_tier,
        "message": message,
    }


def _classify_risk(slope, pct_change):
    """Thresholds are intentionally simple + explainable for a judge Q&A."""
    if slope <= -0.010 or pct_change <= -25:
        return "High", (
            f"Rapid decline detected. Projected {abs(pct_change):.0f}% drop "
            f"if the current trend continues — recommend flagging for "
            f"field verification."
        )
    if slope <= -0.004 or pct_change <= -10:
        return "Medium", (
            f"Gradual decline detected. Projected {abs(pct_change):.0f}% drop "
            f"over the forecast window — worth monitoring closely."
        )
    if slope >= 0.004 or pct_change >= 10:
        return "Recovering", (
            f"Positive trend detected. Index projected to improve by "
            f"{pct_change:.0f}% — consistent with recovery or seasonal regrowth."
        )
    return "Low", "Stable. No significant change trend detected in recent passes."
