"""Offline demo: fake OpenAQ and Open-Meteo servers backed by synthetic data.

`python -m aqi demo` replays the real pipeline day by day against these fakes,
so you can see the page and the live tracking without an API key or network.
The fakes return the same JSON shapes as the real APIs (OpenAQ v3 /locations and
/sensors/{id}/days, Open-Meteo hourly), so the parsing code is exercised too.
All numbers are made up and the demo page says so.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import openaq, pipeline, weather
from .config import Config, load_configs

log = logging.getLogger(__name__)

API_KEY = "demo-key"
IST = "+05:30"


class FakeResponse:
    def __init__(self, status: int, payload: object):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)
        self.headers: dict[str, str] = {}

    def json(self) -> object:
        return self._payload


class SyntheticWorld:
    """Made-up weather and PM2.5 for one city, where pollution responds to the weather."""

    # Monitors around the city centre: lat offset, lon offset, typical ratio to the
    # city level, and how many days after the start the monitor came online.
    LAYOUT = [(-0.010, 0.039, 1.15, 0), (0.036, 0.005, 1.00, 0), (-0.060, -0.027, 0.82, 0),
              (-0.131, 0.062, 1.32, 420)]

    def __init__(self, start: dt.date, end: dt.date, seed: int = 7, name: str = "Jaipur",
                 center: tuple[float, float] = (26.9124, 75.7873), index: int = 0,
                 level: float = 0.0, winter_amp: float = 0.85):
        self.name = name
        self.center = center
        base = 9100 + 200 * index  # Jaipur (index 0): monitors 9101-9104, low-cost sensor 9201
        self.station_meta = [
            (base + k + 1, 800 + 10 * index + k + 1, f"{name} monitor {k + 1} (synthetic)",
             round(center[0] + dlat, 4), round(center[1] + dlon, 4), ratio, offset)
            for k, (dlat, dlon, ratio, offset) in enumerate(self.LAYOUT)
        ]
        self.low_cost = (base + 101, 900 + index, f"{name} rooftop sensor (synthetic, low-cost)",
                         round(center[0] + 0.003, 4), round(center[1] + 0.013, 4))
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.start = start - dt.timedelta(days=45)
        self.days = pd.date_range(self.start, end + dt.timedelta(days=20), freq="D")
        n = len(self.days)
        doy = self.days.dayofyear.to_numpy()
        winter = np.cos(2 * np.pi * (doy - 10) / 365.25)
        monsoon = np.exp(-(((doy - 215) / 32.0) ** 2))

        def ar1(phi: float, sd: float) -> np.ndarray:
            shocks = rng.normal(0, sd, n)
            out = np.zeros(n)
            for i in range(1, n):
                out[i] = phi * out[i - 1] + shocks[i]
            return out

        temp = 26 - 9 * winter + ar1(0.8, 1.2)
        rh = np.clip(36 + 40 * monsoon + 8 * winter + ar1(0.7, 6), 8, 98)
        wind = np.clip(8.5 + 2.5 * (1 - winter) / 2 + 4 * monsoon + ar1(0.55, 2.4), 1, 35)
        rain = np.where(rng.random(n) < 0.03 + 0.45 * monsoon, rng.gamma(1.2, 7, n), 0.0)
        blh = np.clip(950 - 380 * winter + 28 * (wind - 9) - 4 * (rh - 40) + ar1(0.5, 110), 150, 2500)
        wdir = (290 - 70 * monsoon + rng.normal(0, 35, n)) % 360
        pressure = 965 + 7 * winter + ar1(0.8, 1.5)
        # Weather explains part of the day-to-day swings; the rest is noise nobody can forecast.
        log_pm = (3.75 + level + winter_amp * winter - 0.45 * monsoon
                  - 0.03 * (wind - 9) - 0.0004 * (blh - 900) - 0.03 * np.minimum(rain, 20)
                  + ar1(0.75, 0.2) + rng.normal(0, 0.09, n))
        self.pm = pd.Series(np.exp(log_pm), index=self.days)
        self.weather = pd.DataFrame(
            {"temp": temp, "rh": rh, "wind": wind, "rain": rain, "blh": blh, "wdir": wdir, "pressure": pressure},
            index=self.days,
        )

        self.stations: dict[int, pd.DataFrame] = {}
        for sid, _, _, _, _, ratio, offset in self.station_meta:
            values = self.pm.to_numpy() * ratio * np.exp(rng.normal(0, 0.12, n))
            values[rng.random(n) < 0.04] = np.nan          # outages
            values[rng.random(n) < 0.003] = 999.9          # stuck sensor readings
            if offset:
                values[: 45 + offset] = np.nan
            coverage = np.clip(rng.normal(0.93, 0.08, n), 0.4, 1.0)
            self.stations[sid] = pd.DataFrame({"value": values, "coverage": coverage}, index=self.days)

    def first_day(self, sid: int) -> dt.date:
        s = self.stations[sid]["value"].dropna()
        return s.index.min().date()

    def hourly(self, start: dt.date, end: dt.date, issued: dt.date | None) -> pd.DataFrame:
        """Hourly weather; days on/after `issued` get forecast errors growing with lead time."""
        hours = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
        daily = self.weather.reindex(hours.normalize())
        hr = hours.hour.to_numpy()
        wave = np.sin(2 * np.pi * (hr - 9) / 24)
        shape = np.where((hr >= 7) & (hr <= 19), 0.3 + 1.6 * np.sin(np.pi * (hr - 7) / 12), 0.3)

        temp = daily["temp"].to_numpy() + 6 * wave
        rh = daily["rh"].to_numpy() - 12 * wave
        wind = daily["wind"].to_numpy() * (1 + 0.45 * np.sin(2 * np.pi * (hr - 8) / 24))
        blh = daily["blh"].to_numpy() * shape
        precip = daily["rain"].to_numpy() / 24
        wdir = daily["wdir"].to_numpy()
        pressure = daily["pressure"].to_numpy()

        if issued is not None:
            lead = (hours.normalize() - pd.Timestamp(issued)).days.to_numpy()
            future = lead >= 0
            rng = np.random.default_rng([self.seed, issued.toordinal()])
            day_noise = {d: rng.normal(0, 1, 4) for d in sorted(set(lead[future]))}
            for i in np.flatnonzero(future):
                z = day_noise[lead[i]] * (lead[i] + 1)
                temp[i] += 0.6 * z[0]
                wind[i] *= np.exp(0.1 * z[1])
                blh[i] *= np.exp(0.12 * z[2])
                rh[i] += 2.5 * z[3]

        return pd.DataFrame({
            "time": hours.strftime("%Y-%m-%dT%H:%M"),
            "temperature_2m": np.round(temp, 1),
            "relative_humidity_2m": np.round(np.clip(rh, 3, 100)),
            "precipitation": np.round(precip, 2),
            "wind_speed_10m": np.round(np.clip(wind, 0.2, None), 1),
            "wind_direction_10m": np.round(wdir),
            "surface_pressure": np.round(pressure, 1),
            "boundary_layer_height": np.round(blh),
        })


class FakeAPIs:
    """A requests-like session that answers like OpenAQ v3 and Open-Meteo."""

    def __init__(self, worlds: SyntheticWorld | list[SyntheticWorld], reject_optional_weather: bool = False):
        self.worlds = list(worlds) if isinstance(worlds, (list, tuple)) else [worlds]
        self.world = self.worlds[0]
        self.today = dt.date.today()
        self.reject_optional_weather = reject_optional_weather
        self.calls: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):
        params = params or {}
        parsed = urlparse(url)
        self.calls.append(f"{parsed.netloc}{parsed.path}")
        if parsed.netloc == "api.openaq.org":
            if (headers or {}).get("X-API-Key") != API_KEY:
                return FakeResponse(401, {"detail": "Unauthorized. A valid API key must be provided."})
            if parsed.path == "/v3/locations":
                return self._locations(params)
            match = re.fullmatch(r"/v3/sensors/(\d+)/days", parsed.path)
            if match:
                return self._days(int(match.group(1)), params)
        if parsed.netloc in ("api.open-meteo.com", "historical-forecast-api.open-meteo.com"):
            return self._weather(parsed.netloc, params)
        return FakeResponse(404, {"detail": "Not Found"})

    # ---------------------------------------------------------------- OpenAQ
    def _locations(self, params):
        west, south, east, north = (float(v) for v in params["bbox"].split(","))
        inside = lambda lat, lon: west <= lon <= east and south <= lat <= north  # noqa: E731
        results = []
        for w in self.worlds:
            for sid, loc_id, name, lat, lon, _, _ in w.station_meta:
                if inside(lat, lon):
                    results.append(self._location(w, loc_id, name, lat, lon, sid, monitor=True))
            sid, loc_id, name, lat, lon = w.low_cost
            if inside(lat, lon):
                results.append(self._location(w, loc_id, name, lat, lon, sid, monitor=False))
        page, limit = int(params.get("page", 1)), int(params.get("limit", 100))
        chunk = results[(page - 1) * limit: page * limit]
        return FakeResponse(200, {"meta": {"name": "openaq-api", "website": "/", "page": page,
                                           "limit": limit, "found": len(results)}, "results": chunk})

    def _location(self, world, loc_id, name, lat, lon, pm25_sensor, monitor):
        last = self.today - dt.timedelta(days=1)
        return {
            "id": loc_id, "name": name, "locality": world.name, "timezone": "Asia/Kolkata",
            "country": {"id": 9, "code": "IN", "name": "India"},
            "owner": {"id": 4, "name": "Unknown Governmental Organization"},
            "provider": {"id": 168, "name": "CPCB" if monitor else "Low-cost network"},
            "isMobile": False, "isMonitor": monitor,
            "instruments": [{"id": 2, "name": "Government Monitor"}],
            "sensors": [
                {"id": pm25_sensor + 50000, "name": "pm10 µg/m³",
                 "parameter": {"id": 1, "name": "pm10", "units": "µg/m³", "displayName": "PM10"}},
                {"id": pm25_sensor, "name": "pm25 µg/m³",
                 "parameter": {"id": 2, "name": "pm25", "units": "µg/m³", "displayName": "PM2.5"}},
            ],
            "coordinates": {"latitude": lat, "longitude": lon},
            "licenses": None, "bounds": [lon, lat, lon, lat], "distance": None,
            "datetimeFirst": {"utc": "2019-01-01T00:00:00Z", "local": "2019-01-01T05:30:00+05:30"},
            "datetimeLast": {"utc": f"{last}T17:30:00Z", "local": f"{last}T23:00:00+05:30"},
        }

    def _days(self, sid, params):
        world = next((w for w in self.worlds if sid in w.stations), None)
        if world is None:
            return FakeResponse(200, {"meta": {"page": 1, "limit": 100, "found": 0}, "results": []})
        start = dt.date.fromisoformat(params["date_from"][:10])
        end = min(dt.date.fromisoformat(params["date_to"][:10]), self.today)  # today is partial
        data = world.stations[sid]
        items = []
        for ts in pd.date_range(start, end, freq="D"):
            if ts not in data.index:
                continue
            value, coverage = data.loc[ts, "value"], data.loc[ts, "coverage"]
            if np.isnan(value):
                continue
            if ts.date() == self.today:
                coverage = 0.38
            observed = int(round(coverage * 24))
            day, nxt = ts.date(), ts.date() + dt.timedelta(days=1)
            prev = day - dt.timedelta(days=1)
            items.append({
                "value": round(float(value), 2),
                "flagInfo": {"hasFlags": False},
                "parameter": {"id": 2, "name": "pm25", "units": "µg/m³", "displayName": "PM2.5"},
                "period": {
                    "label": "1 day", "interval": "24:00:00",
                    "datetimeFrom": {"utc": f"{prev}T18:30:00Z", "local": f"{day}T00:00:00{IST}"},
                    "datetimeTo": {"utc": f"{day}T18:30:00Z", "local": f"{nxt}T00:00:00{IST}"},
                },
                "coordinates": None,
                "summary": None,
                "coverage": {"expectedCount": 24, "expectedInterval": "24:00:00",
                             "observedCount": observed, "observedInterval": f"{observed}:00:00",
                             "percentComplete": round(100 * observed / 24, 1),
                             "percentCoverage": round(100 * observed / 24, 1)},
            })
        page, limit = int(params.get("page", 1)), int(params.get("limit", 100))
        chunk = items[(page - 1) * limit: page * limit]
        return FakeResponse(200, {"meta": {"name": "openaq-api", "website": "/", "page": page,
                                           "limit": limit, "found": len(items)}, "results": chunk})

    # ---------------------------------------------------------------- Open-Meteo
    def _weather(self, host, params):
        variables = params["hourly"].split(",")
        if self.reject_optional_weather and "boundary_layer_height" in variables:
            return FakeResponse(400, {"error": True, "reason": "Cannot initialize ForecastVariable "
                                                               "from invalid String value boundary_layer_height"})
        lat, lon = float(params["latitude"]), float(params["longitude"])
        world = min(self.worlds, key=lambda w: (w.center[0] - lat) ** 2 + (w.center[1] - lon) ** 2)
        if host.startswith("historical-forecast-api"):
            start = dt.date.fromisoformat(params["start_date"])
            end = dt.date.fromisoformat(params["end_date"])
            frame = world.hourly(start, end, issued=None)
            # The archive lags by a couple of days: newer hours come back empty.
            cutoff = (self.today - dt.timedelta(days=2)).isoformat()
            late = frame["time"].str.slice(0, 10) > cutoff
            frame.loc[late, frame.columns != "time"] = np.nan
        else:
            past, ahead = int(params.get("past_days", 0)), int(params.get("forecast_days", 7))
            start = self.today - dt.timedelta(days=past)
            end = self.today + dt.timedelta(days=ahead - 1)
            frame = world.hourly(start, end, issued=self.today)
        hourly = {"time": frame["time"].tolist()}
        for var in variables:
            hourly[var] = [None if pd.isna(v) else float(v) for v in frame[var]]
        return FakeResponse(200, {
            "latitude": params["latitude"], "longitude": params["longitude"],
            "timezone": params["timezone"], "utc_offset_seconds": 19800,
            "hourly_units": {v: "" for v in variables}, "hourly": hourly,
        })


def no_sleep(_: float) -> None:
    pass


# Level and winter swing per city in the demo, so the cities look different.
PROFILES = [(0.0, 0.85), (0.45, 0.95), (-0.35, 0.5)]


def worlds_for(cfgs: list[Config], start: dt.date, end: dt.date, seed: int = 7) -> list[SyntheticWorld]:
    worlds = []
    for i, cfg in enumerate(cfgs):
        level, winter_amp = PROFILES[i % len(PROFILES)]
        worlds.append(SyntheticWorld(start, end, seed + i, name=cfg.city.name,
                                     center=(cfg.city.latitude, cfg.city.longitude), index=i,
                                     level=level, winter_amp=winter_amp))
    return worlds


def run_demo(out_dir: str | Path = "demo", days: int = 16, seed: int = 7,
             cfgs: list[Config] | None = None, end: dt.date | None = None) -> Path:
    """Simulate `days` daily runs ending today for every city; returns the main page's path."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfgs = [c.with_root(out_dir) for c in (cfgs or load_configs())]
    tz = ZoneInfo(cfgs[0].city.timezone)
    end = end or dt.datetime.now(tz).date()
    fake = FakeAPIs(worlds_for(cfgs, cfgs[0].history_start, end + dt.timedelta(days=10), seed))
    oa_http = openaq.make_http(API_KEY, session=fake, sleep=no_sleep)
    wx_http = weather.make_http(session=fake, sleep=no_sleep)
    first = end - dt.timedelta(days=days - 1)
    for i in range(days):
        day = first + dt.timedelta(days=i)
        fake.today = day
        now = dt.datetime.combine(day, dt.time(9, 10), tz)
        pipeline.run(cfgs, oa_http, wx_http, now=now,
                     backtest="force" if i == days - 1 else "skip", demo=True)
    return cfgs[0].site_root / "index.html"
