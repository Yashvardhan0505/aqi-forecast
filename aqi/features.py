"""Turn the daily PM2.5 series and weather into a supervised learning table.

One row per base date t (the latest complete day when the forecast is made) for a
given horizon h. The target is the change in log PM2.5 from t to t+h, so the model
learns how much tomorrow differs from today instead of re-learning the level. That
makes "no change" (persistence) the natural starting point it has to improve on.

Everything in a row is known on the morning after t: PM2.5 up to t, the weather
forecast for t+h, and the calendar. tests/test_features.py checks that nothing
from after t leaks in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = 7
WEATHER_FEATURES = ["temp", "rh", "precip", "wind", "wind_u", "wind_v", "pressure", "blh", "blh_min"]
CHANGE_FEATURES = ["temp", "rh", "wind", "blh"]

META_COLUMNS = ["target", "anchor", "actual", "last_week_value", "target_date"]


def daily_series(city: pd.DataFrame) -> pd.Series:
    """City PM2.5 as a float Series on a gap-free daily DatetimeIndex (NaN = missing)."""
    if city.empty:
        return pd.Series(dtype=float)
    s = pd.Series(pd.to_numeric(city["pm25"], errors="coerce").to_numpy(),
                  index=pd.to_datetime(city["date"]), name="pm25").sort_index()
    return s.asfreq("D")


def weather_frame(weather: pd.DataFrame) -> pd.DataFrame:
    if weather.empty:
        return pd.DataFrame(columns=WEATHER_FEATURES)
    w = weather.copy()
    w.index = pd.to_datetime(w.pop("date"))
    return w.sort_index().apply(pd.to_numeric, errors="coerce")


def build_frame(city: pd.DataFrame, weather: pd.DataFrame, horizon: int) -> pd.DataFrame:
    y = daily_series(city)
    if y.empty:
        return pd.DataFrame()
    wx = weather_frame(weather)
    idx = pd.date_range(y.index.min(), y.index.max() + pd.Timedelta(days=horizon), freq="D")
    y = y.reindex(idx)
    ly = np.log1p(y)
    wx = wx.reindex(idx)

    f = pd.DataFrame(index=idx)
    for k in range(LAGS):
        f[f"lag{k}"] = ly.shift(k)
    f["mean7"] = ly.rolling(7, min_periods=4).mean()
    f["mean14"] = ly.rolling(14, min_periods=7).mean()
    f["std7"] = ly.rolling(7, min_periods=4).std()
    f["trend3"] = ly - ly.shift(3)
    f["last_week"] = ly.shift(7 - horizon)  # same weekday as the target, one week earlier

    for col in WEATHER_FEATURES:
        f[f"wx_{col}"] = wx[col].shift(-horizon) if col in wx else np.nan
    for col in CHANGE_FEATURES:
        if col in wx:
            f[f"dwx_{col}"] = wx[col].shift(-horizon) - wx[col]

    target_day = idx + pd.Timedelta(days=horizon)
    doy = target_day.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["dow"] = target_day.dayofweek.to_numpy()

    f["target"] = ly.shift(-horizon) - ly
    f["anchor"] = y
    f["actual"] = y.shift(-horizon)
    f["last_week_value"] = y.shift(7 - horizon)
    f["target_date"] = target_day
    return f[f["lag0"].notna()]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in META_COLUMNS]
