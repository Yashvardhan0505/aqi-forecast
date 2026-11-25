"""The learning table must only use information available when the forecast is made."""

import numpy as np
import pandas as pd

from aqi import features


def make_inputs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    city = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "pm25": rng.uniform(20, 200, n), "n_stations": 4})
    wdates = pd.date_range("2026-01-01", periods=n + 5, freq="D")
    weather = pd.DataFrame({"date": wdates.strftime("%Y-%m-%d"),
                            **{c: rng.normal(size=len(wdates)) for c in features.WEATHER_FEATURES}})
    return city, weather


def test_no_future_pm25_leaks_into_features():
    city, weather = make_inputs()
    h = 2
    base = pd.Timestamp("2026-03-01")
    before = features.build_frame(city, weather, h).loc[base]

    changed = city.copy()
    future = pd.to_datetime(changed["date"]) > base
    changed.loc[future, "pm25"] = 999.0  # rewrite everything after the base date
    after = features.build_frame(changed, weather, h).loc[base]

    cols = features.feature_columns(features.build_frame(city, weather, h))
    pd.testing.assert_series_equal(before[cols], after[cols], check_names=False)
    assert before["actual"] != after["actual"]  # only the target moved


def test_alignment_of_target_weather_and_last_week():
    city, weather = make_inputs()
    h = 3
    frame = features.build_frame(city, weather, h)
    base = pd.Timestamp("2026-02-10")
    row = frame.loc[base]
    y = features.daily_series(city)
    target = base + pd.Timedelta(days=h)

    assert row["target_date"] == target
    assert np.isclose(row["actual"], y[target])
    assert np.isclose(row["anchor"], y[base])
    assert np.isclose(row["last_week_value"], y[target - pd.Timedelta(days=7)])
    assert np.isclose(row["target"], np.log1p(y[target]) - np.log1p(y[base]))
    w = weather.set_index("date")
    assert np.isclose(row["wx_temp"], w.loc[target.strftime("%Y-%m-%d"), "temp"])


def test_rows_need_todays_value():
    city, weather = make_inputs()
    city.loc[city["date"] == "2026-02-10", "pm25"] = np.nan
    frame = features.build_frame(city, weather, 1)
    assert pd.Timestamp("2026-02-10") not in frame.index
