"""City averaging and the append-only forecast log."""

import numpy as np
import pandas as pd

from aqi import data, tracking


def station_days(values_by_sensor, start="2026-01-01"):
    rows = []
    for sid, values in values_by_sensor.items():
        for i, v in enumerate(values):
            if v is not None:
                d = (pd.Timestamp(start) + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
                rows.append({"date": d, "sensor_id": sid, "value": v, "coverage": 0.95})
    return pd.DataFrame(rows, columns=data.STATION_DAILY_COLUMNS)


def test_station_dropping_out_does_not_shift_the_city_level(small_cfg):
    base = np.linspace(40, 80, 60)
    roadside = list(2 * base[:50]) + [None] * 10  # reads 2x higher, then goes offline
    park = list(base)
    city = data.build_city_series(small_cfg, station_days({1: roadside, 2: park}))
    both = city["pm25"].iloc[:50].to_numpy()
    alone = city["pm25"].iloc[50:].to_numpy()
    assert np.allclose(both, 1.5 * base[:50], rtol=0.01)
    assert np.allclose(alone, 1.5 * base[50:], rtol=0.01)  # a plain mean would drop to 1.0x
    assert list(city["n_stations"].iloc[[0, -1]]) == [2, 1]


def test_invalid_readings_are_dropped(small_cfg):
    df = station_days({1: [50, 999.9, 0, 60]})
    df.loc[3, "coverage"] = 0.3
    city = data.build_city_series(small_cfg, df)
    assert city["pm25"].tolist() == [50.0]


def log_rows(base, values, issued="2026-11-26T03:40:00Z"):
    rows = []
    for h, v in enumerate(values, start=1):
        target = (pd.Timestamp(base) + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        for method in ("model", "persistence", "last_week"):
            rows.append({"base_date": base, "target_date": target, "horizon": h, "method": method,
                         "pred": v, "lo": v - 5 if method == "model" else None,
                         "hi": v + 5 if method == "model" else None, "issued_at": issued})
    return pd.DataFrame(rows, columns=tracking.LOG_COLUMNS)


def test_log_is_append_only():
    log_df, added = tracking.append(pd.DataFrame(columns=tracking.LOG_COLUMNS), log_rows("2026-11-25", [60, 70]))
    assert added == 6
    # A rerun on the same morning with different numbers must not rewrite history.
    log_df, added = tracking.append(log_df, log_rows("2026-11-25", [99, 99]))
    assert added == 0 and set(log_df["pred"]) == {60, 70}
    log_df, added = tracking.append(log_df, log_rows("2026-11-26", [80, 90]))
    assert added == 6 and len(log_df) == 12


def test_log_round_trips_through_csv(small_cfg):
    log_df, _ = tracking.append(pd.DataFrame(columns=tracking.LOG_COLUMNS), log_rows("2026-11-25", [60]))
    tracking.save_log(small_cfg, log_df)
    again = tracking.load_log(small_cfg)
    _, added = tracking.append(again, log_rows("2026-11-25", [61]))
    assert added == 0


def test_pairs_join_forecasts_with_measurements():
    log_df, _ = tracking.append(pd.DataFrame(columns=tracking.LOG_COLUMNS), log_rows("2026-11-25", [60, 70]))
    city = pd.DataFrame({"date": ["2026-11-26"], "pm25": [65.0], "n_stations": [4]})
    pairs = tracking.pairs(log_df, city)
    first = pairs[pairs["horizon"] == 1].iloc[0]
    assert first["actual"] == 65.0 and first["model"] == 60 and first["lo"] == 55
    assert pd.isna(pairs[pairs["horizon"] == 2].iloc[0]["actual"])  # not measured yet
