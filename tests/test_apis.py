"""OpenAQ and Open-Meteo parsing, paging, retries and error handling."""

import datetime as dt

import pandas as pd
import pytest

from aqi import openaq, weather
from aqi.demo import API_KEY, FakeAPIs, FakeResponse, SyntheticWorld, no_sleep
from aqi.http import Http, HttpError


def day_item(local_from, local_to, value=50.0, coverage=None, utc_from=None, utc_to=None):
    return {
        "value": value,
        "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"},
        "period": {"label": "1 day", "interval": "24:00:00",
                   "datetimeFrom": {"utc": utc_from, "local": local_from},
                   "datetimeTo": {"utc": utc_to, "local": local_to}},
        "coverage": coverage,
    }


@pytest.mark.parametrize("local_from, local_to", [
    ("2026-11-25T00:00:00+05:30", "2026-11-26T00:00:00+05:30"),  # labelled by the day's start
    ("2026-11-25T01:00:00+05:30", "2026-11-26T00:00:00+05:30"),  # labelled by the first hour
])
def test_day_is_the_local_calendar_day(local_from, local_to):
    assert openaq.parse_day(day_item(local_from, local_to))["date"] == "2026-11-25"


def test_day_falls_back_to_utc_shifted_to_india():
    item = day_item(None, None, utc_from="2026-11-24T18:30:00Z", utc_to="2026-11-25T18:30:00Z")
    assert openaq.parse_day(item)["date"] == "2026-11-25"


def test_coverage_and_missing_values():
    cov = {"expectedCount": 24, "observedCount": 18, "percentComplete": 75.0}
    assert openaq.parse_day(day_item("2026-11-25T00:00:00+05:30", "2026-11-26T00:00:00+05:30",
                                     coverage=cov))["coverage"] == 0.75
    assert openaq._coverage({"percentComplete": 50.0}) == 0.5
    assert openaq._coverage(None) is None
    assert openaq.parse_day(day_item("2026-11-25T00:00:00+05:30", "2026-11-26T00:00:00+05:30",
                                     value=None)) is None


@pytest.fixture
def fake():
    world = SyntheticWorld(dt.date(2025, 6, 1), dt.date(2026, 11, 26))
    api = FakeAPIs(world)
    api.today = dt.date(2026, 11, 26)
    return api


def test_finds_only_pm25_sensors(fake):
    http = openaq.make_http(API_KEY, session=fake, sleep=no_sleep)
    sensors = openaq.find_pm25_sensors(http, (75.6, 26.7, 76.0, 27.1))
    ids = {s["sensor_id"] for s in sensors}
    assert ids == {9101, 9102, 9103, 9104, 9201}  # PM10 sensors skipped
    assert {s["sensor_id"] for s in sensors if not s["is_monitor"]} == {9201}


def test_daily_means_pages_and_filters(fake, monkeypatch):
    monkeypatch.setattr(openaq, "PAGE_LIMIT", 50)  # force several pages
    http = openaq.make_http(API_KEY, session=fake, sleep=no_sleep)
    rows = openaq.daily_means(http, 9102, dt.date(2026, 6, 1), dt.date(2026, 11, 25))
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates)
    assert dates[0] >= "2026-06-01" and dates[-1] <= "2026-11-25"
    assert len(dates) > 150  # ~4% of days are outages in the synthetic data
    assert len(fake.calls) > 3


def test_bad_key_is_a_clear_error(fake):
    http = openaq.make_http("wrong-key", session=fake, sleep=no_sleep)
    with pytest.raises(HttpError) as info:
        openaq.find_pm25_sensors(http, (75.6, 26.7, 76.0, 27.1))
    assert info.value.status == 401


def test_missing_key_message():
    with pytest.raises(RuntimeError, match="OPENAQ_API_KEY"):
        openaq.make_http("")


class FlakySession:
    """429 with a reset header, then success."""

    def __init__(self):
        self.n = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.n += 1
        if self.n == 1:
            r = FakeResponse(429, {"detail": "Too many requests"})
            r.headers = {"x-ratelimit-reset": "7"}
            return r
        return FakeResponse(200, {"ok": True})


def test_retries_after_rate_limit():
    slept = []
    http = Http(session=FlakySession(), sleep=slept.append)
    assert http.get_json("https://api.openaq.org/v3/locations") == {"ok": True}
    assert 8 in slept  # waited for the reset (+1 s)


def test_client_errors_are_not_retried():
    class BadRequest:
        calls = 0

        def get(self, *a, **k):
            BadRequest.calls += 1
            return FakeResponse(422, {"detail": "bad"})

    with pytest.raises(HttpError):
        Http(session=BadRequest(), sleep=no_sleep).get_json("https://x.test")
    assert BadRequest.calls == 1


def test_weather_daily_rollup_and_optional_variable(fake):
    city = type("C", (), {"latitude": 26.9, "longitude": 75.8, "timezone": "Asia/Kolkata"})
    http = weather.make_http(session=fake, sleep=no_sleep)
    daily = weather.recent_and_forecast(http, city, past_days=3, forecast_days=2)
    assert list(daily.columns) == weather.COLUMNS
    assert len(daily) == 5  # 3 past days + today + tomorrow
    assert daily["blh"].notna().all()
    assert (daily["blh_min"] <= daily["blh"]).all()

    fake.reject_optional_weather = True
    daily = weather.recent_and_forecast(http, city, past_days=3, forecast_days=2)
    assert daily["blh"].isna().all() and daily["temp"].notna().all()


def test_wind_direction_is_averaged_as_a_vector():
    hours = pd.DataFrame({
        "time": [f"2026-11-25T{h:02d}:00" for h in range(24)],
        "temperature_2m": 20.0, "relative_humidity_2m": 50.0, "precipitation": 0.0,
        "wind_speed_10m": 10.0, "wind_direction_10m": [350.0, 10.0] * 12, "surface_pressure": 960.0,
    })
    day = weather.to_daily(hours).iloc[0]
    # Wind from the north blows towards the south: v is negative, u is ~0.
    assert day["wind_v"] < -9 and abs(day["wind_u"]) < 0.5
