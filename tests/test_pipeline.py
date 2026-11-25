"""End to end: the real pipeline against fake OpenAQ / Open-Meteo servers."""

import dataclasses
import datetime as dt
import json
import re
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from aqi import data, openaq, pipeline, tracking, weather
from aqi.demo import API_KEY, FakeAPIs, SyntheticWorld, no_sleep, worlds_for

from conftest import END

TZ = ZoneInfo("Asia/Kolkata")


def setup(cfgs, key=API_KEY, **fake_kwargs):
    cfgs = cfgs if isinstance(cfgs, list) else [cfgs]
    worlds = worlds_for(cfgs, cfgs[0].history_start, END + dt.timedelta(days=5))
    fake = FakeAPIs(worlds, **fake_kwargs)
    oa = openaq.make_http(key, session=fake, sleep=no_sleep)
    wx = weather.make_http(session=fake, sleep=no_sleep)
    return fake, oa, wx


def run_day(cfgs, fake, oa, wx, day, backtest="skip"):
    fake.today = day
    now = dt.datetime.combine(day, dt.time(9, 10), TZ)
    return pipeline.run(cfgs, oa, wx, now=now, backtest=backtest, demo=True)


def page_data(html):
    blob = re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1)
    return json.loads(blob)


def test_three_daily_runs(small_cfg):
    cfg = small_cfg
    fake, oa, wx = setup(cfg)
    days = [END - dt.timedelta(days=2), END - dt.timedelta(days=1), END]

    run_day(cfg, fake, oa, wx, days[0])
    first_log = tracking.load_log(cfg)
    assert len(first_log) == 9  # 3 horizons x (model + 2 baselines)

    run_day(cfg, fake, oa, wx, days[1])
    run_day(cfg, fake, oa, wx, days[2], backtest="force")
    log_df = tracking.load_log(cfg)
    assert len(log_df) == 27
    # Day one's forecasts were not touched by later runs.
    again = log_df[log_df["base_date"] == first_log["base_date"].iloc[0]].reset_index(drop=True)
    pd.testing.assert_frame_equal(first_log.reset_index(drop=True), again, check_dtype=False)
    assert (log_df["target_date"] > log_df["base_date"]).all()

    # Stored data: no partial "today", no low-cost sensor, weather reaches a few days ahead.
    station_daily = pd.read_csv(cfg.data_dir / data.STATION_DAILY_FILE, dtype={"date": str})
    assert station_daily["date"].max() < END.isoformat()
    stations = pd.read_csv(cfg.data_dir / data.STATIONS_FILE)
    assert 9201 not in set(stations["sensor_id"])
    wx_df = pd.read_csv(cfg.data_dir / data.WEATHER_FILE, dtype={"date": str})
    assert wx_df["date"].max() >= (END + dt.timedelta(days=2)).isoformat()

    # Backtest ran and the page is complete, at the city's URL and at the main one.
    bt = pd.read_csv(cfg.data_dir / pipeline.BACKTEST_FILE)
    assert set(bt["horizon"]) == {1, 2, 3} and bt["model"].notna().all()
    page = (cfg.site_dir / "index.html").read_text()
    for text in ("Jaipur air quality forecast", "How accurate is it?", "Demo page with synthetic data",
                 "Today", "Tomorrow", "Show all scores"):
        assert text in page
    assert "{{" not in page  # every template slot was filled
    assert 'class="cities"' not in page  # one city: no switcher
    assert (cfg.site_root / "index.html").read_text().count("Jaipur air quality forecast") >= 1
    chart = page_data(page)["chart"]
    assert len(chart["future"]) == 3 and chart["measured"]
    latest = json.loads((cfg.site_dir / "data" / "latest.json").read_text())
    assert [f["date"] for f in latest["forecast"]] == [d.isoformat() for d in
                                                       (END, END + dt.timedelta(days=1), END + dt.timedelta(days=2))]


def test_three_cities_with_a_switcher(small_cfgs):
    fake, oa, wx = setup(small_cfgs)
    contexts = run_day(small_cfgs, fake, oa, wx, END)
    assert set(contexts) == {"jaipur", "delhi", "mumbai"}

    root = small_cfgs[0].site_root
    main = (root / "index.html").read_text()
    assert "Jaipur air quality forecast" in main
    assert re.findall(r'<nav class="cities".*?</nav>', main, re.S)
    for slug in ("jaipur", "delhi", "mumbai"):
        assert f'href="{slug}/"' in main  # links from the main page
        city_page = (root / slug / "index.html").read_text()
        assert f'href="../{slug}/" aria-current="page"' in city_page  # its own pill is marked
    delhi = (root / "delhi" / "index.html").read_text()
    assert "Delhi air quality forecast" in delhi and 'href="../mumbai/"' in delhi

    # Each city keeps its own data, from its own monitors.
    delhi_ids = set(pd.read_csv(small_cfgs[1].data_dir / data.STATIONS_FILE)["sensor_id"])
    assert delhi_ids == {9301, 9302, 9303, 9304}
    summary = json.loads((root / "data" / "latest.json").read_text())
    assert [c["city"] for c in summary["cities"]] == ["Jaipur", "Delhi", "Mumbai"]


def test_one_city_failing_does_not_stop_the_others(small_cfgs, capsys):
    cfgs = list(small_cfgs)
    cfgs[2] = dataclasses.replace(cfgs[2], city=dataclasses.replace(cfgs[2].city, bbox=(60.0, 10.0, 60.5, 10.5)))
    fake, oa, wx = setup(cfgs)
    contexts = run_day(cfgs, fake, oa, wx, END)
    assert set(contexts) == {"jaipur", "delhi"}
    main = (cfgs[0].site_root / "index.html").read_text()
    assert 'href="delhi/"' in main and 'href="mumbai/"' not in main
    assert "::warning title=mumbai failed::" in capsys.readouterr().out


def test_every_city_failing_fails_the_run(small_cfgs):
    fake, oa, wx = setup(small_cfgs, key="wrong-key")
    with pytest.raises(RuntimeError, match="Every city failed"):
        run_day(small_cfgs, fake, oa, wx, END)


def test_moves_single_city_data_into_its_folder(small_cfgs):
    cfg = small_cfgs[0]
    fake, oa, wx = setup(cfg)
    run_day(cfg, fake, oa, wx, END - dt.timedelta(days=1))
    # Recreate the old layout: everything directly in data/.
    for path in cfg.data_dir.iterdir():
        path.rename(cfg.data_root / path.name)
    cfg.data_dir.rmdir()
    old_log = tracking.load_log(dataclasses.replace(cfg, data_dir=cfg.data_root))

    run_day(small_cfgs, fake, oa, wx, END)
    assert not (cfg.data_root / tracking.LOG_FILE).exists()
    log_df = tracking.load_log(cfg)
    assert set(old_log["base_date"]) < set(log_df["base_date"])  # old forecasts kept, new ones added


def test_works_without_boundary_layer_height(small_cfg):
    fake, oa, wx = setup(small_cfg, reject_optional_weather=True)
    run_day(small_cfg, fake, oa, wx, END)
    wx_df = pd.read_csv(small_cfg.data_dir / data.WEATHER_FILE)
    assert wx_df["blh"].isna().all()
    assert len(tracking.load_log(small_cfg)) == 9


def test_stale_data_means_no_forecast(small_cfg):
    cfg = small_cfg
    fake, oa, wx = setup(cfg)
    for station in fake.world.stations.values():  # monitors stop reporting a week ago
        station.loc[pd.Timestamp(END - dt.timedelta(days=7)):, "value"] = float("nan")
    run_day(cfg, fake, oa, wx, END)
    assert tracking.load_log(cfg).empty
    assert "No forecast today" in (cfg.site_dir / "index.html").read_text()


def test_rebuild_without_network(small_cfg):
    fake, oa, wx = setup(small_cfg)
    run_day(small_cfg, fake, oa, wx, END)
    ctx = pipeline.rebuild(small_cfg)["jaipur"]
    assert ctx["upcoming"] or ctx["stale"]


def test_one_broken_sensor_does_not_stop_the_run(small_cfg):
    fake, oa, wx = setup(small_cfg)
    original = fake._days

    def broken(sid, params):
        if sid == 9103:
            from aqi.demo import FakeResponse
            return FakeResponse(404, {"detail": "Not Found"})
        return original(sid, params)

    fake._days = broken
    run_day(small_cfg, fake, oa, wx, END)
    station_daily = pd.read_csv(small_cfg.data_dir / data.STATION_DAILY_FILE)
    assert 9103 not in set(station_daily["sensor_id"])
    assert len(tracking.load_log(small_cfg)) == 9


def test_forecasts_that_only_cover_past_days_are_not_shown(small_cfg):
    cfg = small_cfg
    fake, oa, wx = setup(cfg)
    # Newest data is 4 days old: still fresh enough to forecast (limit is 3 days of lag),
    # but every target (base + 1..3 days) is already in the past.
    for station in fake.world.stations.values():
        station.loc[pd.Timestamp(END - dt.timedelta(days=3)):, "value"] = float("nan")
    ctx = run_day(cfg, fake, oa, wx, END)["jaipur"]
    log_df = tracking.load_log(cfg)
    assert len(log_df) == 9 and (log_df["target_date"] < END.isoformat()).all()
    assert ctx["stale"] and not ctx["upcoming"]
    assert "No forecast today" in (cfg.site_dir / "index.html").read_text()


def test_synthetic_world_defaults_to_jaipur():
    world = SyntheticWorld(dt.date(2026, 1, 1), dt.date(2026, 2, 1))
    assert world.station_meta[0][0] == 9101 and world.low_cost[0] == 9201
