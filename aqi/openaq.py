"""OpenAQ v3 client: find PM2.5 monitors in an area and download their daily means.

API docs: https://docs.openaq.org (needs a free API key sent as X-API-Key).
OpenAQ's daily values are means of hourly averages from 01:00 to 00:00 local time,
so a "day" here is an IST calendar day, the same convention CPCB uses.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from .http import Http

log = logging.getLogger(__name__)

BASE_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2
PAGE_LIMIT = 1000
MAX_PAGES = 50
# Free keys allow 60 requests/minute; 1.1 s between calls stays under that.
MIN_INTERVAL_S = 1.1


def make_http(api_key: str, session: Any = None, sleep: Any = None) -> Http:
    if not api_key:
        raise RuntimeError(
            "OPENAQ_API_KEY is not set. Get a free key at https://explore.openaq.org/register "
            "and add it as a repository secret (see README)."
        )
    kwargs = {"sleep": sleep} if sleep else {}
    return Http(headers={"X-API-Key": api_key, "Accept": "application/json"},
                min_interval=MIN_INTERVAL_S, session=session, **kwargs)


def find_pm25_sensors(http: Http, bbox: tuple[float, float, float, float]) -> list[dict]:
    """Every PM2.5 sensor at a location inside bbox = (west, south, east, north)."""
    west, south, east, north = bbox
    params = {
        "bbox": f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}",
        "parameters_id": PM25_PARAMETER_ID,
        "limit": PAGE_LIMIT,
    }
    sensors: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        data = http.get_json(f"{BASE_URL}/locations", {**params, "page": page})
        results = data.get("results") or []
        for loc in results:
            sensors.extend(_pm25_sensors_at(loc))
        if len(results) < PAGE_LIMIT:
            break
    log.info("OpenAQ: %d PM2.5 sensors in the search area", len(sensors))
    return sensors


def _pm25_sensors_at(loc: dict) -> list[dict]:
    coords = loc.get("coordinates") or {}
    last = (loc.get("datetimeLast") or {}).get("utc") or ""
    out = []
    for sensor in loc.get("sensors") or []:
        param = sensor.get("parameter") or {}
        if param.get("id") != PM25_PARAMETER_ID and param.get("name") != "pm25":
            continue
        out.append({
            "sensor_id": int(sensor["id"]),
            "location_id": int(loc["id"]),
            "name": loc.get("name") or f"Location {loc['id']}",
            "provider": (loc.get("provider") or {}).get("name") or "",
            "latitude": coords.get("latitude"),
            "longitude": coords.get("longitude"),
            "is_monitor": bool(loc.get("isMonitor", False)),
            "last_seen_utc": last,
        })
    return out


def daily_means(http: Http, sensor_id: int, date_from: dt.date, date_to: dt.date) -> list[dict]:
    """Daily mean PM2.5 for one sensor, inclusive of both dates (local calendar days).

    Long ranges are requested a year at a time to keep each response small.
    """
    rows: dict[str, dict] = {}
    chunk_start = date_from
    while chunk_start <= date_to:
        chunk_end = min(date_to, chunk_start + dt.timedelta(days=365))
        params = {
            # Pad by a day on each side; rows are filtered by local date below.
            "date_from": (chunk_start - dt.timedelta(days=1)).isoformat(),
            "date_to": (chunk_end + dt.timedelta(days=1)).isoformat(),
            "limit": PAGE_LIMIT,
        }
        for page in range(1, MAX_PAGES + 1):
            data = http.get_json(f"{BASE_URL}/sensors/{sensor_id}/days", {**params, "page": page})
            results = data.get("results") or []
            for item in results:
                row = parse_day(item)
                if row and date_from.isoformat() <= row["date"] <= date_to.isoformat():
                    rows[row["date"]] = row
            if len(results) < PAGE_LIMIT:
                break
        chunk_start = chunk_end + dt.timedelta(days=1)
    return [rows[d] for d in sorted(rows)]


def parse_day(item: dict) -> dict | None:
    value = item.get("value")
    day = _local_day(item.get("period") or {})
    if value is None or day is None:
        return None
    return {"date": day.isoformat(), "value": float(value), "coverage": _coverage(item.get("coverage"))}


def _local_day(period: dict) -> dt.date | None:
    """The calendar day a daily period covers, in the station's local time.

    Uses the midpoint of datetimeFrom/datetimeTo, so it is right whether the
    API labels the day by its start (00:00) or by its first hour (01:00).
    """
    frm, to = period.get("datetimeFrom") or {}, period.get("datetimeTo") or {}
    start, end = _parse_dt(frm.get("local")), _parse_dt(to.get("local"))
    if not (start and end):
        # No local times: fall back to UTC shifted to India (+05:30).
        ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
        start = start or (_parse_dt(frm.get("utc")) or None)
        end = end or (_parse_dt(to.get("utc")) or None)
        start = start.astimezone(ist) if start and start.tzinfo else start
        end = end.astimezone(ist) if end and end.tzinfo else end
    if start and end and end > start:
        return (start + (end - start) / 2).date()
    if start:
        return start.date()
    return None


def _parse_dt(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coverage(cov: dict | None) -> float | None:
    """Fraction of the day's hours that had data (0-1), or None if unknown."""
    if not cov:
        return None
    expected, observed = cov.get("expectedCount"), cov.get("observedCount")
    if expected and observed is not None:
        return round(min(observed / expected, 1.0), 3)
    pct = cov.get("percentComplete")
    if pct is None:
        return None
    return round(min(pct / 100 if pct > 1 else pct, 1.0), 3)
