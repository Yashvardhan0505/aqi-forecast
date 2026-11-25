"""The forecast log: every forecast is written once, before its day is measured.

data/forecast_log.csv is append-only. A forecast for a given (base date, target
date, method) is never rewritten, so the live error numbers on the page are what
the model actually predicted at the time, and git history proves it.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data import read_csv, write_csv

LOG_FILE = "forecast_log.csv"
LOG_COLUMNS = ["base_date", "target_date", "horizon", "method", "pred", "lo", "hi", "issued_at"]
KEY = ["base_date", "target_date", "method"]


def load_log(cfg: Config) -> pd.DataFrame:
    df = read_csv(cfg.data_dir / LOG_FILE, LOG_COLUMNS)
    for col in ("base_date", "target_date", "method", "issued_at"):
        df[col] = df[col].astype(str) if not df.empty else df[col]
    return df


def save_log(cfg: Config, log_df: pd.DataFrame) -> None:
    write_csv(log_df.sort_values(["base_date", "horizon", "method"]), cfg.data_dir / LOG_FILE)


def append(log_df: pd.DataFrame, new_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Add rows whose key isn't in the log yet. Existing rows are never changed."""
    if new_rows.empty:
        return log_df, 0
    if log_df.empty:
        return new_rows[LOG_COLUMNS].reset_index(drop=True), len(new_rows)
    existing = pd.MultiIndex.from_frame(log_df[KEY])
    fresh = new_rows[~pd.MultiIndex.from_frame(new_rows[KEY]).isin(existing)]
    if fresh.empty:
        return log_df, 0
    return pd.concat([log_df, fresh[LOG_COLUMNS]], ignore_index=True), len(fresh)


def pairs(log_df: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    """One row per (base date, horizon) with each method's forecast and the measured value."""
    cols = ["base_date", "target_date", "horizon", "actual", "model", "lo", "hi", "persistence", "last_week"]
    if log_df.empty:
        return pd.DataFrame(columns=cols)
    key = ["base_date", "target_date", "horizon"]
    wide = log_df.pivot_table(index=key, columns="method", values="pred", aggfunc="first")
    bounds = (log_df[log_df["method"] == "model"].set_index(key)[["lo", "hi"]])
    wide = wide.join(bounds, how="left").reset_index()
    for m in ("model", "persistence", "last_week"):
        if m not in wide:
            wide[m] = float("nan")
    actual = dict(zip(city["date"].astype(str), pd.to_numeric(city["pm25"], errors="coerce")))
    wide["actual"] = wide["target_date"].map(actual)
    wide["horizon"] = wide["horizon"].astype(int)
    return wide[cols]
