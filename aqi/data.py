"""Download, store and clean the data: station PM2.5, the city series, weather.

Everything is kept as small CSV files in data/ and committed by the GitHub Action,
so the repo itself is the database and every run is reproducible from it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import openaq, weather
from .config import Config
from .http import Http, HttpError

log = logging.getLogger(__name__)

STATIONS_FILE = "stations.csv"
STATION_DAILY_FILE = "pm25_station_daily.csv"
CITY_DAILY_FILE = "pm25_city_daily.csv"
WEATHER_FILE = "weather_daily.csv"
STATE_FILE = "state.json"

STATION_COLUMNS = ["sensor_id", "location_id", "name", "provider", "latitude", "longitude",
                   "is_monitor", "last_seen_utc"]
STATION_DAILY_COLUMNS = ["date", "sensor_id", "value", "coverage"]
CITY_COLUMNS = ["date", "pm25", "n_stations"]


# --------------------------------------------------------------------------- files

def read_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype={"date": str})


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g", lineterminator="\n")


def load_state(cfg: Config) -> dict:
    path = cfg.data_dir / STATE_FILE
    return json.loads(path.read_text()) if path.exists() else {}


def save_state(cfg: Config, state: dict) -> None:
    path = cfg.data_dir / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _days_since(date_str: str | None, today: dt.date) -> int | None:
    if not date_str:
        return None
    return (today - dt.date.fromisoformat(date_str[:10])).days


# --------------------------------------------------------------------------- stations

def update_stations(cfg: Config, http: Http, today: dt.date, state: dict) -> pd.DataFrame:
    """Refresh the list of PM2.5 monitors in the city every few days."""
    path = cfg.data_dir / STATIONS_FILE
    existing = read_csv(path, STATION_COLUMNS)
    age = _days_since(state.get("stations_refreshed"), today)
    if not existing.empty and age is not None and age < cfg.station_refresh_days:
        return existing

    found = pd.DataFrame(openaq.find_pm25_sensors(http, cfg.city.bbox), columns=STATION_COLUMNS)
    if cfg.reference_monitors_only and found["is_monitor"].astype(bool).any():
        found = found[found["is_monitor"].astype(bool)]
    if found.empty:
        if existing.empty:
            raise RuntimeError(
                f"No PM2.5 monitors found inside bbox {cfg.city.bbox}. "
                "Check city.bbox in config.toml (order: west, south, east, north)."
            )
        log.warning("Station search came back empty; keeping the previous list")
        return existing

    found = found.drop_duplicates("sensor_id").sort_values("sensor_id")
    write_csv(found, path)
    state["stations_refreshed"] = today.isoformat()
    log.info("Using %d monitors: %s", len(found), ", ".join(found["name"].astype(str)))
    return found


# --------------------------------------------------------------------------- PM2.5

def update_pm25(cfg: Config, http: Http, stations: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Backfill new stations from history_start; re-download recent days for the rest."""
    path = cfg.data_dir / STATION_DAILY_FILE
    existing = read_csv(path, STATION_DAILY_COLUMNS)
    yesterday = today - dt.timedelta(days=1)
    frames = [existing] if not existing.empty else []
    attempted, failures = 0, []

    for st in stations.itertuples(index=False):
        sid = int(st.sensor_id)
        have = existing.loc[existing["sensor_id"] == sid, "date"]
        last_seen = str(st.last_seen_utc)[:10] if pd.notna(st.last_seen_utc) and st.last_seen_utc else None

        if have.empty:
            if last_seen and last_seen < cfg.history_start.isoformat():
                continue  # retired before our window starts
            date_from = cfg.history_start
        else:
            if last_seen and _days_since(last_seen, today) > 30:
                continue  # offline for a month; its history stays in the file
            resume = dt.date.fromisoformat(have.max()) + dt.timedelta(days=1)
            date_from = max(cfg.history_start, min(resume, today - dt.timedelta(days=cfg.refetch_days)))
        if date_from > yesterday:
            continue

        attempted += 1
        try:
            rows = openaq.daily_means(http, sid, date_from, yesterday)
        except HttpError as exc:
            if exc.status in (401, 403):
                raise  # a key problem affects every sensor; stop with a clear error
            failures.append(f"{st.name}: {exc}")
            log.warning("  Skipping %s for now: %s", st.name, exc)
            continue
        log.info("  %-45s %s..%s: %d days", str(st.name)[:45], date_from, yesterday, len(rows))
        if rows:
            df = pd.DataFrame(rows)
            df["sensor_id"] = sid
            frames.append(df[STATION_DAILY_COLUMNS])

    if attempted and len(failures) == attempted:
        raise RuntimeError("OpenAQ returned errors for every monitor: " + "; ".join(failures[:3]))
    if not frames:
        return pd.DataFrame(columns=STATION_DAILY_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined["sensor_id"] = combined["sensor_id"].astype(int)
    combined = combined.drop_duplicates(["date", "sensor_id"], keep="last")  # newest download wins
    combined = combined[combined["date"] < today.isoformat()]  # today isn't over yet
    combined = combined.sort_values(["date", "sensor_id"]).reset_index(drop=True)
    write_csv(combined, path)
    return combined


def build_city_series(cfg: Config, station_daily: pd.DataFrame) -> pd.DataFrame:
    """One PM2.5 value per day for the whole city.

    Stations differ systematically (a roadside monitor reads higher than a park), so
    a plain average jumps whenever a station goes offline. Each station is first
    scaled by its typical ratio to the city mean, then the scaled values are averaged.
    """
    df = station_daily.copy()
    if df.empty:
        city = pd.DataFrame(columns=CITY_COLUMNS)
        write_csv(city, cfg.data_dir / CITY_DAILY_FILE)
        return city

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    valid = (df["value"] > 0) & (df["value"] < cfg.max_valid_pm25)
    valid &= df["coverage"].isna() | (df["coverage"] >= cfg.min_daily_coverage)
    dropped = int((~valid).sum())
    if dropped:
        log.info("Dropped %d station-days (low coverage or invalid values)", dropped)
    df = df[valid]

    wide = df.pivot_table(index="date", columns="sensor_id", values="value", aggfunc="mean").sort_index()
    plain_mean = wide.mean(axis=1)
    n_reporting = wide.notna().sum(axis=1)
    min_overlap = min(3, wide.shape[1])

    ratios = {}
    for sid in wide.columns:
        r = (wide[sid] / plain_mean)[n_reporting >= min_overlap].dropna()
        ratios[sid] = float(np.clip(r.median(), 0.5, 2.0)) if len(r) >= 30 and min_overlap > 1 else 1.0

    scaled = wide.div(pd.Series(ratios))
    city = pd.DataFrame({
        "date": wide.index,
        "pm25": scaled.mean(axis=1).round(1).to_numpy(),
        "n_stations": n_reporting.to_numpy(),
    })
    city = city[city["n_stations"] > 0].reset_index(drop=True)
    write_csv(city, cfg.data_dir / CITY_DAILY_FILE)
    return city


# --------------------------------------------------------------------------- weather

def update_weather(cfg: Config, http: Http, today: dt.date) -> pd.DataFrame:
    """Archive of past daily weather plus the latest forecast for the next few days.

    Past days stay as they were first stored (from the forecast archive); only the
    recent window and the days ahead are refreshed on each run.
    """
    path = cfg.data_dir / WEATHER_FILE
    existing = read_csv(path, ["date", *weather.COLUMNS])
    frames = []
    recent_window = 14

    if existing.empty:
        frames.append(weather.history(http, cfg.city, cfg.history_start, today - dt.timedelta(days=1)))
    else:
        existing = existing.set_index("date")
        frames.append(existing)
        last = dt.date.fromisoformat(existing.index.max())
        if last < today - dt.timedelta(days=recent_window):
            frames.append(weather.history(http, cfg.city, last + dt.timedelta(days=1),
                                          today - dt.timedelta(days=1)))

    frames.append(weather.recent_and_forecast(http, cfg.city, past_days=recent_window, forecast_days=5))
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No weather data returned by Open-Meteo")
    combined = pd.concat(frames)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = "date"
    out = combined.reset_index()[["date", *weather.COLUMNS]]
    write_csv(out, path)
    return out
