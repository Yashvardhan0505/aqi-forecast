"""Score forecasts against what was measured, the same way for backtest and live."""

from __future__ import annotations

import pandas as pd

from .aqi_india import pm25_category

METHODS = ("model", "persistence", "last_week")
METHOD_LABELS = {
    "model": "Model",
    "persistence": "Same as last day",
    "last_week": "Same as last week",
}


def summarize(pairs: pd.DataFrame) -> list[dict]:
    """Per-horizon error stats.

    `pairs` has one row per (base_date, horizon) with columns actual, model, lo, hi,
    persistence, last_week. Only rows where every method has a forecast are scored,
    so all methods are compared on exactly the same days.
    """
    if pairs is None or pairs.empty:
        return []
    out = []
    for horizon, g in pairs.groupby("horizon"):
        g = g.dropna(subset=["actual", *METHODS])
        if g.empty:
            continue
        actual_cat = g["actual"].map(pm25_category)
        row: dict = {"horizon": int(horizon), "n": int(len(g))}
        for m in METHODS:
            err = g[m] - g["actual"]
            row[f"mae_{m}"] = float(err.abs().mean())
            row[f"rmse_{m}"] = float((err**2).mean() ** 0.5)
            row[f"bias_{m}"] = float(err.mean())
            row[f"category_hit_{m}"] = float((g[m].map(pm25_category) == actual_cat).mean())
        for base in ("persistence", "last_week"):
            denom = row[f"mae_{base}"]
            row[f"skill_vs_{base}"] = float(1 - row["mae_model"] / denom) if denom > 0 else None
        if {"lo", "hi"} <= set(g.columns) and g["lo"].notna().all():
            row["interval_coverage"] = float(((g["lo"] <= g["actual"]) & (g["actual"] <= g["hi"])).mean())
        out.append(row)
    return out
