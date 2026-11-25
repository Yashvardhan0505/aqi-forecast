"""Weather from Open-Meteo (free, no key): hourly values rolled up to IST days.

History comes from the Historical Forecast API, an archive of past forecasts in the
same format as the live Forecast API. Training on archived forecasts instead of
reanalysis keeps training data closer to what the model sees when it forecasts.
Data: Open-Meteo.com, CC BY 4.0.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from .config import City
from .http import Http, HttpError

log = logging.getLogger(__name__)

HISTORY_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CORE_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)
# Boundary layer height matters a lot for PM2.5 (a shallow layer traps smoke) but
# isn't served by every model, so it's requested separately from the core variables.
OPTIONAL_VARS = ("boundary_layer_height",)

COLUMNS = ["temp", "rh", "precip", "wind", "wind_u", "wind_v", "pressure", "blh", "blh_min"]


def make_http(session=None, sleep=None) -> Http:
    kwargs = {"sleep": sleep} if sleep else {}
    return Http(min_interval=0.5, session=session, **kwargs)


def history(http: Http, city: City, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Daily weather for [start, end], fetched in yearly chunks."""
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=365))
        hourly = _hourly(http, HISTORY_URL, city, {
            "start_date": cursor.isoformat(),
            "end_date": chunk_end.isoformat(),
        })
        frames.append(to_daily(hourly))
        cursor = chunk_end + dt.timedelta(days=1)
    return _concat(frames)


def recent_and_forecast(http: Http, city: City, past_days: int = 14, forecast_days: int = 5) -> pd.DataFrame:
    """The last `past_days` days plus `forecast_days` days ahead (today counts as one)."""
    hourly = _hourly(http, FORECAST_URL, city, {"past_days": past_days, "forecast_days": forecast_days})
    return to_daily(hourly)


def _hourly(http: Http, url: str, city: City, extra: dict) -> pd.DataFrame:
    base ={"latitude": city.latitude, "longitude": city.longitude, "timezone": city.timezone, **extra}
    core = pd.DataFrame(_get_hourly(http, url, base, CORE_VARS))
    try:
        optional = pd.DataFrame(_get_hourly(http, url, base, OPTIONAL_VARS))
    except HttpError as exc:
        if exc.status != 400:
            raise
        log.warning("Open-Meteo doesn't serve %s here; continuing without it", ", ".join(OPTIONAL_VARS))
        return core
    if optional.empty or "time" not in optional:
        return core
    return core.merge(optional, on="time", how="left")


def _get_hourly(http: Http, url: str, base: dict, variables: tuple[str, ...]) -> dict:
    data = http.get_json(url, {**base, "hourly": ",".join(variables)})
    return data.get("hourly") or {}


def to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly frame with a local `time` column -> one row per local date."""
    if hourly.empty or "time" not in hourly:
        return pd.DataFrame(columns=COLUMNS, index=pd.Index([], name="date"))
    df = hourly.copy()
    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = df["time"].astype(str).str.slice(0, 10)

    # Wind direction is where the wind comes FROM; average it as a vector so that
    # 350 and 10 degrees average to north, not south.
    speed = df["wind_speed_10m"]
    rad = np.deg2rad(df["wind_direction_10m"])
    df["u"] = -speed * np.sin(rad)
    df["v"] = -speed * np.cos(rad)

    g = df.groupby("date")
    daily = pd.DataFrame({
        "temp": g["temperature_2m"].mean(),
        "rh": g["relative_humidity_2m"].mean(),
        "precip": g["precipitation"].sum(min_count=1),
        "wind": g["wind_speed_10m"].mean(),
        "wind_u": g["u"].mean(),
        "wind_v": g["v"].mean(),
        "pressure": g["surface_pressure"].mean(),
    })
    if "boundary_layer_height" in df:
        daily["blh"] = g["boundary_layer_height"].mean()
        daily["blh_min"] = g["boundary_layer_height"].min()
    else:
        daily["blh"] = np.nan
        daily["blh_min"] = np.nan

    # Keep only days with (nearly) all 24 hours, so partial edge days don't skew means.
    daily = daily[g["temperature_2m"].count() >= 20]
    daily.index.name = "date"
    return daily[COLUMNS]


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=COLUMNS, index=pd.Index([], name="date"))
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()
