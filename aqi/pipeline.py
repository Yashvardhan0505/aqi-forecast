"""The daily job for every configured city: update data -> forecast -> log -> score -> publish."""

from __future__ import annotations

import datetime as dt
import logging
import shutil
from zoneinfo import ZoneInfo

import pandas as pd

from . import backtest as backtest_mod
from . import data, features, model, site, tracking
from .config import Config
from .http import Http

log = logging.getLogger(__name__)

BACKTEST_FILE = "backtest.csv"
# Files the one-city version kept directly in data/; they now live in data/<city>/.
DATA_FILES = (data.STATIONS_FILE, data.STATION_DAILY_FILE, data.CITY_DAILY_FILE, data.WEATHER_FILE,
              tracking.LOG_FILE, BACKTEST_FILE, data.STATE_FILE)


def local_now(cfg: Config) -> dt.datetime:
    return dt.datetime.now(ZoneInfo(cfg.city.timezone))


def _as_list(cfgs: Config | list[Config]) -> list[Config]:
    return [cfgs] if isinstance(cfgs, Config) else list(cfgs)


def run(cfgs: Config | list[Config], openaq_http: Http, weather_http: Http, now: dt.datetime | None = None,
        backtest: str = "auto", demo: bool = False) -> dict[str, dict]:
    """Run every city, then write the site. `backtest` is "auto" (rerun when stale), "force" or "skip".

    A city that fails (an API outage, no monitors in its box) is logged and keeps its
    last page; the other cities still update. Only if every city fails does the run fail.
    """
    cfgs = _as_list(cfgs)
    now = now or local_now(cfgs[0])
    migrate_single_city_layout(cfgs[0])

    contexts: dict[str, dict] = {}
    failures: dict[str, str] = {}
    for cfg in cfgs:
        try:
            contexts[cfg.city.slug] = run_city(cfg, openaq_http, weather_http, now, backtest)
        except Exception as exc:  # noqa: BLE001 - one city's problem must not stop the others
            failures[cfg.city.slug] = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s", cfg.city.name, exc)
            fallback = stored_context(cfg, now)
            if fallback:
                contexts[cfg.city.slug] = fallback

    if contexts:
        site.write(cfgs, contexts, now, demo)
    for slug, message in failures.items():
        print(f"::warning title={slug} failed::{message}")  # shows up on the GitHub Actions run page
    if len(failures) == len(cfgs):
        raise RuntimeError("Every city failed. First error: " + next(iter(failures.values())))
    return contexts


def run_city(cfg: Config, openaq_http: Http, weather_http: Http, now: dt.datetime, backtest: str) -> dict:
    today = now.astimezone(ZoneInfo(cfg.city.timezone)).date()
    log.info("== %s, %s", cfg.city.name, today)
    state = data.load_state(cfg)
    stations = data.update_stations(cfg, openaq_http, today, state)
    station_daily = data.update_pm25(cfg, openaq_http, stations, today)
    city = data.build_city_series(cfg, station_daily)
    weather_df = data.update_weather(cfg, weather_http, today)
    if city.empty:
        raise RuntimeError(f"No usable PM2.5 data for {cfg.city.name} yet. "
                           f"Check data/{cfg.city.slug}/stations.csv.")
    return score(cfg, stations, city, weather_df, now, state, backtest=backtest, new_forecasts=True)


def rebuild(cfgs: Config | list[Config], backtest: str = "skip") -> dict[str, dict]:
    """Rebuild the pages (and optionally the backtests) from stored data, without downloading."""
    cfgs = _as_list(cfgs)
    now = local_now(cfgs[0])
    contexts = {}
    for cfg in cfgs:
        ctx = stored_context(cfg, now, backtest)
        if ctx:
            contexts[cfg.city.slug] = ctx
    if not contexts:
        raise RuntimeError("No stored data yet: run `python -m aqi run` first.")
    site.write(cfgs, contexts, now, demo=False)
    return contexts


def stored_context(cfg: Config, now: dt.datetime, backtest: str = "skip") -> dict | None:
    """Page data from what's already on disk (used for rebuilds and when a city's update fails)."""
    city = data.read_csv(cfg.data_dir / data.CITY_DAILY_FILE, data.CITY_COLUMNS)
    if city.empty:
        return None
    stations = data.read_csv(cfg.data_dir / data.STATIONS_FILE, data.STATION_COLUMNS)
    weather_df = data.read_csv(cfg.data_dir / data.WEATHER_FILE, ["date"])
    return score(cfg, stations, city, weather_df, now, data.load_state(cfg), backtest=backtest,
                 new_forecasts=False)


def score(cfg: Config, stations: pd.DataFrame, city: pd.DataFrame, weather_df: pd.DataFrame,
          now: dt.datetime, state: dict, backtest: str, new_forecasts: bool) -> dict:
    """Make and log today's forecasts, refresh the backtest when due, and return the page data."""
    today = now.astimezone(ZoneInfo(cfg.city.timezone)).date()
    base_date = dt.date.fromisoformat(str(city["date"].iloc[-1]))
    data_age = (today - base_date).days - 1  # 0 when yesterday is the newest complete day
    log_df = tracking.load_log(cfg)
    changed = False

    if new_forecasts:
        if data_age <= cfg.max_data_age_days:
            issued_at = now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows = make_forecasts(cfg, city, weather_df, base_date, issued_at)
            log_df, added = tracking.append(log_df, rows)
            tracking.save_log(cfg, log_df)
            log.info("Logged %d new forecast rows from base date %s", added, base_date)
        else:
            log.warning("Newest PM2.5 data for %s is from %s, %d days ago: not forecasting today",
                        cfg.city.name, base_date, data_age + 1)
        state["last_run"] = now.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        changed = True

    bt_path = cfg.data_dir / BACKTEST_FILE
    bt_age = data._days_since(state.get("backtest_run"), today)
    due = backtest == "force" or (
        backtest == "auto" and (not bt_path.exists() or bt_age is None or bt_age >= cfg.backtest_refresh_days)
    )
    if due:
        log.info("Running the %s backtest over the last %d days", cfg.city.name, cfg.backtest_days)
        bt_pairs, _ = backtest_mod.run(cfg, city, weather_df)
        data.write_csv(bt_pairs, bt_path)
        state["backtest_run"] = today.isoformat()
        changed = True
    bt_pairs = data.read_csv(bt_path, backtest_mod.PAIR_COLUMNS)

    if changed:
        data.save_state(cfg, state)
    live = tracking.pairs(log_df, city)
    return site.context(cfg, city=city, stations=stations, log_df=log_df, live=live,
                        backtest_pairs=bt_pairs, now=now)


def make_forecasts(cfg: Config, city: pd.DataFrame, weather_df: pd.DataFrame,
                   base_date: dt.date, issued_at: str) -> pd.DataFrame:
    """Train one model per horizon on everything measured so far; forecast from base_date."""
    rows = []
    base = pd.Timestamp(base_date)
    for h in cfg.horizons:
        frame = features.build_frame(city, weather_df, h)
        train = frame[frame["target"].notna()]
        if len(train) < cfg.min_train_days or base not in frame.index:
            log.warning("h=%d: %d training rows (need %d); no forecast yet", h, len(train), cfg.min_train_days)
            continue
        x = frame.loc[[base]]
        pred = model.fit(train, h, cfg.interval).predict(x).iloc[0]
        common = {"base_date": base_date.isoformat(),
                  "target_date": (base_date + dt.timedelta(days=h)).isoformat(),
                  "horizon": h, "issued_at": issued_at}
        last_week = x["last_week_value"].iloc[0]
        rows += [
            {**common, "method": "model", "pred": round(pred["pred"], 1),
             "lo": round(pred["lo"], 1), "hi": round(pred["hi"], 1)},
            {**common, "method": "persistence", "pred": round(float(x["anchor"].iloc[0]), 1),
             "lo": None, "hi": None},
            {**common, "method": "last_week", "pred": round(float(last_week), 1) if pd.notna(last_week) else None,
             "lo": None, "hi": None},
        ]
        log.info("  %s (h=%d): %.0f ug/m3 (80%% range %.0f-%.0f)", common["target_date"], h,
                 pred["pred"], pred["lo"], pred["hi"])
    return pd.DataFrame(rows, columns=tracking.LOG_COLUMNS)


def migrate_single_city_layout(cfg: Config) -> None:
    """Move data written by the one-city version (data/*.csv) into data/<first city>/.

    Keeps the forecast log, so live accuracy carries on. If the old stations clearly
    belong to a different city, the files go to data/_previous_city/ instead.
    """
    root = cfg.data_root
    old = [root / name for name in DATA_FILES if (root / name).exists()]
    if not old:
        return
    target = cfg.data_dir
    if target.exists() and any(target.iterdir()):
        log.warning("Old files in %s left alone: %s already has data", root, target)
        return
    stations = data.read_csv(root / data.STATIONS_FILE, data.STATION_COLUMNS)
    if not stations.empty:
        lat = pd.to_numeric(stations["latitude"], errors="coerce").median()
        lon = pd.to_numeric(stations["longitude"], errors="coerce").median()
        west, south, east, north = cfg.city.bbox
        if not (west <= lon <= east and south <= lat <= north):
            target = root / "_previous_city"
            log.warning("The old data isn't from %s; moving it to %s", cfg.city.name, target)
    target.mkdir(parents=True, exist_ok=True)
    for path in old:
        shutil.move(str(path), target / path.name)
    log.info("Moved %d files from the single-city layout into %s", len(old), target)
