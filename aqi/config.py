"""Load config.toml into one typed Config per city."""

from __future__ import annotations

import datetime as dt
import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class City:
    name: str
    slug: str  # used in folder names and the page URL, e.g. "delhi"
    latitude: float
    longitude: float
    timezone: str
    bbox: tuple[float, float, float, float]  # west, south, east, north


@dataclass(frozen=True)
class Config:
    """Settings for one city. `cities` lists every configured city (for the page's switcher)."""

    city: City
    cities: tuple[City, ...]
    history_start: dt.date
    min_daily_coverage: float
    max_valid_pm25: float
    refetch_days: int
    station_refresh_days: int
    reference_monitors_only: bool
    horizons: tuple[int, ...]
    interval: tuple[float, float]
    min_train_days: int
    max_data_age_days: int
    backtest_days: int
    backtest_retrain_every: int
    backtest_refresh_days: int
    repo_url: str
    data_root: Path  # data/
    site_root: Path  # docs/
    data_dir: Path   # data/<slug>/
    site_dir: Path   # docs/<slug>/

    def with_root(self, root: str | Path) -> "Config":
        """Same settings, but read and write data/ and docs/ under another folder."""
        root = Path(root)
        return replace(self, data_root=root / "data", site_root=root / "docs",
                       data_dir=root / "data" / self.city.slug, site_dir=root / "docs" / self.city.slug)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "city"


def _city(table: dict) -> City:
    bbox = tuple(float(v) for v in table["bbox"])
    name = str(table["name"])
    if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError(f"{name}: bbox must be [west, south, east, north] with west < east and south < north")
    return City(
        name=name,
        slug=slugify(str(table.get("slug", name))),
        latitude=float(table["latitude"]),
        longitude=float(table["longitude"]),
        timezone=str(table.get("timezone", "Asia/Kolkata")),
        bbox=bbox,  # type: ignore[arg-type]
    )


def load_configs(path: str | Path | None = None) -> list[Config]:
    """One Config per [[cities]] entry, in file order (the first is the default page)."""
    path = Path(path) if path else ROOT / "config.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    tables = raw.get("cities") or ([raw["city"]] if "city" in raw else [])
    if not tables:
        raise ValueError("config.toml needs at least one [[cities]] entry")
    cities = tuple(_city(t) for t in tables)
    slugs = [c.slug for c in cities]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"City names must be different: {slugs}")

    data = raw.get("data", {})
    fc = raw.get("forecast", {})
    bt = raw.get("backtest", {})
    site = raw.get("site", {})

    horizons = tuple(sorted({int(h) for h in fc.get("horizons", [1, 2, 3])}))
    if not horizons or horizons[0] < 1 or horizons[-1] > 6:
        raise ValueError("forecast.horizons must be between 1 and 6 days")
    lo, hi = (float(q) for q in fc.get("interval", [0.1, 0.9]))
    if not 0 < lo < 0.5 < hi < 1:
        raise ValueError("forecast.interval must look like [0.1, 0.9]")

    root = path.parent
    common = dict(
        cities=cities,
        history_start=dt.date.fromisoformat(str(data.get("history_start", "2023-01-01"))),
        min_daily_coverage=float(data.get("min_daily_coverage", 0.66)),
        max_valid_pm25=float(data.get("max_valid_pm25", 999)),
        refetch_days=int(data.get("refetch_days", 10)),
        station_refresh_days=int(data.get("station_refresh_days", 7)),
        reference_monitors_only=bool(data.get("reference_monitors_only", True)),
        horizons=horizons,
        interval=(lo, hi),
        min_train_days=int(fc.get("min_train_days", 180)),
        max_data_age_days=int(fc.get("max_data_age_days", 3)),
        backtest_days=int(bt.get("days", 180)),
        backtest_retrain_every=int(bt.get("retrain_every", 14)),
        backtest_refresh_days=int(bt.get("refresh_days", 7)),
        repo_url=str(site.get("repo_url", "")),
        data_root=root / "data",
        site_root=root / "docs",
    )
    return [Config(city=c, data_dir=root / "data" / c.slug, site_dir=root / "docs" / c.slug, **common)
            for c in cities]


def load_config(path: str | Path | None = None) -> Config:
    """The first (default) city's config."""
    return load_configs(path)[0]
