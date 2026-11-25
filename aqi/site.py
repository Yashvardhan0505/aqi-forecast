"""Build the public page that GitHub Pages serves from docs/.

Static HTML is rendered here (it works without JavaScript); a small script in
page.html draws the trend chart and its hover readout from the embedded JSON.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd

from .aqi_india import PM25_BANDS, category, pm25_to_aqi
from .config import Config
from .evaluation import METHOD_LABELS, METHODS, summarize

TEMPLATE = Path(__file__).with_name("page.html")
FONT_URL = "https://fonts.googleapis.com/css2?family=Anek+Latin:wdth,wght@75..125,100..800&display=swap"

# The forecast strip is tinted like the sky: clear blue when the air is clean,
# milky, then dusty brown as PM2.5 rises. The category name is always printed too.
HAZE = {
    "Good": "#8cc4ec",
    "Satisfactory": "#b4d3ea",
    "Moderately polluted": "#d0d8dc",
    "Poor": "#d8cdbb",
    "Very poor": "#bba88f",
    "Severe": "#76665a",
}
# Small swatches after CPCB's green-to-maroon colour code, always next to the name.
CATEGORY_DOT = {
    "Good": "#1b9e4b",
    "Satisfactory": "#8bc34a",
    "Moderately polluted": "#f2c500",
    "Poor": "#f28c00",
    "Very poor": "#e53935",
    "Severe": "#8e1b1b",
}
LIVE_MIN_DAYS = 14   # show live accuracy once this many forecasts per lead time are scored
LIVE_WINDOW = 30
CHART_DAYS = 90


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- write

def write(cfgs: list[Config], contexts: dict[str, dict], now: dt.datetime, demo: bool = False) -> None:
    """One page per city at docs/<slug>/, plus the first city's page at docs/ itself.

    The main link keeps showing the first city; the switcher at the top links to the rest.
    """
    available = [c for c in cfgs if c.city.slug in contexts]
    if not available:
        return
    root = cfgs[0].site_root
    for cfg in available:
        ctx = contexts[cfg.city.slug]
        out = root / cfg.city.slug
        (out / "data").mkdir(parents=True, exist_ok=True)
        page = render(cfg, ctx, nav=city_nav(available, contexts, cfg.city.slug, prefix="../"), demo=demo)
        (out / "index.html").write_text(page, encoding="utf-8")
        (out / "data" / "latest.json").write_text(
            json.dumps({**ctx["public"], "demo": demo}, indent=2) + "\n", encoding="utf-8")

    first = available[0]
    page = render(first, contexts[first.city.slug],
                  nav=city_nav(available, contexts, first.city.slug, prefix=""), demo=demo)
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(page, encoding="utf-8")
    (root / ".nojekyll").write_text("")
    summary = {"demo": demo, "cities": [{**contexts[c.city.slug]["public"], "page": f"{c.city.slug}/"}
                                        for c in available]}
    (root / "data" / "latest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def city_nav(cfgs: list[Config], contexts: dict[str, dict], current: str, prefix: str) -> str:
    """Links to every city with today's forecast AQI; empty when there's only one city."""
    if len(cfgs) < 2:
        return ""
    items = []
    for cfg in cfgs:
        ctx = contexts[cfg.city.slug]
        today = next((u for u in ctx["upcoming"] if u["label"] == "Today"), None)
        if today:
            dot = f'<span class="dot" style="background:{CATEGORY_DOT[today["category"]]}"></span>'
            aqi = (f'<span class="nav-aqi"><span class="sr">AQI today: </span>{today["aqi"]}'
                   f'<span class="sr">, {esc(today["category"].lower())}</span></span>')
        else:
            dot, aqi = '<span class="dot dot-none"></span>', ""
        here = ' aria-current="page"' if cfg.city.slug == current else ""
        items.append(f'<li><a href="{esc(prefix + cfg.city.slug + "/")}"{here}>{dot}'
                     f'<span class="nav-name">{esc(cfg.city.name)}</span>{aqi}</a></li>')
    return f'<nav class="cities" aria-label="Choose a city"><ul>{"".join(items)}</ul></nav>'


def context(cfg: Config, *, city, stations, log_df, live, backtest_pairs, now) -> dict:
    tz = ZoneInfo(cfg.city.timezone)
    now = now.astimezone(tz)
    today = now.date()
    city = city.copy()
    city["pm25"] = pd.to_numeric(city["pm25"], errors="coerce")
    last = city.dropna(subset=["pm25"]).iloc[-1]
    base_date = dt.date.fromisoformat(str(last["date"]))
    stale = (today - base_date).days - 1 > cfg.max_data_age_days

    # Upcoming forecasts: the newest base date's model rows for today onwards.
    upcoming = []
    if not log_df.empty and not stale:
        model_rows = log_df[log_df["method"] == "model"]
        newest = model_rows["base_date"].max()
        rows = model_rows[(model_rows["base_date"] == newest) & (model_rows["target_date"] >= today.isoformat())]
        for r in rows.sort_values("target_date").itertuples(index=False):
            target = dt.date.fromisoformat(r.target_date)
            aqi = pm25_to_aqi(float(r.pred))
            upcoming.append({
                "date": r.target_date, "label": day_label(target, today), "horizon": int(r.horizon),
                "pm25": float(r.pred), "lo": float(r.lo), "hi": float(r.hi),
                "aqi": aqi, "category": category(aqi),
            })

    metrics = accuracy(cfg, live, backtest_pairs, today)
    chart = chart_data(cfg, city, live, backtest_pairs, upcoming, today)
    n_stations = int(last["n_stations"]) if pd.notna(last.get("n_stations")) else len(stations)
    latest_aqi = pm25_to_aqi(float(last["pm25"]))

    public = {
        "city": cfg.city.name,
        "generated_at": now.isoformat(timespec="minutes"),
        "latest_measurement": {"date": base_date.isoformat(), "pm25": float(last["pm25"]),
                               "aqi": latest_aqi, "category": category(latest_aqi),
                               "stations": n_stations},
        "forecast": upcoming,
        "accuracy": metrics["public"],
    }
    # Forecasts exist, but none reach today (data a few days late): same message as stale.
    stale = stale or (not upcoming and not log_df.empty)
    return {
        "today": today, "now": now, "base_date": base_date, "stale": stale,
        "latest": {"pm25": float(last["pm25"]), "aqi": latest_aqi, "category": category(latest_aqi)},
        "n_stations": n_stations, "stations": stations, "upcoming": upcoming,
        "metrics": metrics, "chart": chart, "public": public,
    }


def day_label(target: dt.date, today: dt.date) -> str:
    if target == today:
        return "Today"
    if target == today + dt.timedelta(days=1):
        return "Tomorrow"
    return f"{target:%a}, {target.day} {target:%b}"


def short_date(d: dt.date) -> str:
    return f"{d.day} {d:%b}"


# --------------------------------------------------------------------------- data for charts and tables

def accuracy(cfg: Config, live: pd.DataFrame, backtest_pairs: pd.DataFrame, today: dt.date) -> dict:
    scored = live.dropna(subset=["actual"]) if not live.empty else live
    window_start = (today - dt.timedelta(days=LIVE_WINDOW)).isoformat()
    live30 = {m["horizon"]: m for m in summarize(scored[scored["target_date"] >= window_start])} if not scored.empty else {}
    live_all = {m["horizon"]: m for m in summarize(scored)} if not scored.empty else {}
    bt = {m["horizon"]: m for m in summarize(backtest_pairs)} if not backtest_pairs.empty else {}

    shown = {}
    for h in cfg.horizons:
        if h in live30 and live30[h]["n"] >= LIVE_MIN_DAYS:
            shown[h] = ("live", live30[h])
        elif h in bt:
            shown[h] = ("backtest", bt[h])
    public = {
        "live_last_30_days": list(live30.values()),
        "live_all": list(live_all.values()),
        "backtest": list(bt.values()),
    }
    return {"live30": live30, "live_all": live_all, "backtest": bt, "shown": shown, "public": public}


def chart_data(cfg: Config, city: pd.DataFrame, live: pd.DataFrame, backtest_pairs: pd.DataFrame,
               upcoming: list[dict], today: dt.date) -> dict:
    start = today - dt.timedelta(days=CHART_DAYS)
    days = pd.date_range(start, today - dt.timedelta(days=1), freq="D").strftime("%Y-%m-%d")
    values = dict(zip(city["date"].astype(str), city["pm25"]))
    measured = [[d, _r(values.get(d))] for d in days]

    h0 = min(cfg.horizons)
    live1 = live[(live["horizon"] == h0) & live["model"].notna()] if not live.empty else live
    live_by_day = {r.target_date: r for r in live1.itertuples(index=False)} if not live1.empty else {}
    live_start = min(live_by_day) if live_by_day else None
    bt1 = backtest_pairs[backtest_pairs["horizon"] == h0] if not backtest_pairs.empty else backtest_pairs
    bt_by_day = {r.target_date: r for r in bt1.itertuples(index=False)} if not bt1.empty else {}

    forecast = []
    for d in days:
        if d in live_by_day:
            r, src = live_by_day[d], "live"
        elif d in bt_by_day and (live_start is None or d < live_start):
            r, src = bt_by_day[d], "backtest"
        else:
            continue
        forecast.append([d, _r(r.model), _r(r.lo), _r(r.hi), src])
    future = [[u["date"], _r(u["pm25"]), _r(u["lo"]), _r(u["hi"]), u["horizon"]] for u in upcoming]
    has_backtest = any(f[4] == "backtest" for f in forecast)
    return {
        "measured": measured, "forecast": forecast, "future": future,
        "liveStart": live_start if has_backtest else None, "horizon": h0,
    }


def _r(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if v != v else round(v, 1)


# --------------------------------------------------------------------------- render

def render(cfg: Config, ctx: dict, nav: str = "", demo: bool = False) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    city_name = cfg.city.name
    first = ctx["upcoming"][0] if ctx["upcoming"] else None
    if first:
        description = (f"{first['label']}: AQI {first['aqi']} ({first['category'].lower()}), "
                       f"PM2.5 about {first['pm25']:.0f} µg/m³. Forecast updated every morning.")
    else:
        description = f"Daily PM2.5 and AQI forecast for {city_name}, updated every morning."
    favicon_color = HAZE.get(first["category"] if first else ctx["latest"]["category"], "#d0d8dc")
    favicon = "data:image/svg+xml," + quote(
        f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
        f"<circle cx='8' cy='8' r='7' fill='{favicon_color}' stroke='#555' stroke-width='1'/></svg>"
    )
    page_data = {
        "today": ctx["today"].isoformat(),
        "bands": [[b[1], b[4]] for b in PM25_BANDS],
        "chart": ctx["chart"],
        "city": city_name,
    }
    slots = {
        "TITLE": esc(f"{city_name} air quality forecast"),
        "DESCRIPTION": esc(description),
        "FAVICON": esc(favicon),
        "FONT_URL": esc(FONT_URL),
        "BANNER": render_banner(demo),
        "NAV": nav,
        "HEADER": render_header(cfg, ctx),
        "HERO": render_hero(cfg, ctx),
        "TREND": render_trend(cfg, ctx),
        "ACCURACY": render_accuracy(cfg, ctx),
        "METHOD": render_method(cfg, ctx),
        "FOOTER": render_footer(cfg, ctx),
        "DATA_JSON": json.dumps(page_data, separators=(",", ":")).replace("</", "<\\/"),
    }
    for key, value in slots.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def render_banner(demo: bool) -> str:
    if not demo:
        return ""
    return ('<p class="banner" role="note">Demo page with synthetic data, built offline. '
            'The real page replaces this after the GitHub Action’s first run.</p>')


def render_header(cfg: Config, ctx: dict) -> str:
    now = ctx["now"]
    hour = now.strftime("%I").lstrip("0") or "12"
    updated = f"{now.day} {now:%b %Y}, {hour}:{now:%M} {now:%p}".replace("AM", "am").replace("PM", "pm")
    tzname = now.tzname() or cfg.city.timezone
    latest = ctx["latest"]
    stations = f"{ctx['n_stations']} monitor{'s' if ctx['n_stations'] != 1 else ''}"
    return (
        f'<h1>{esc(cfg.city.name)} air quality forecast</h1>\n'
        f'<p class="lede">PM2.5 for the next three days, forecast every morning from government '
        f'monitor readings and the weather forecast.</p>\n'
        f'<p class="meta">Updated {esc(updated)} {esc(tzname)}. Latest measurement: '
        f'{esc(short_date(ctx["base_date"]))}, PM2.5 {latest["pm25"]:.0f} µg/m³ '
        f'(AQI {latest["aqi"]}, {esc(latest["category"].lower())}) averaged over {esc(stations)}.</p>'
    )


def _ink(bg_hex: str) -> tuple[str, str]:
    """Primary and secondary text colour with enough contrast on a strip panel."""
    r, g, b = (int(bg_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return ("#ffffff", "#eee8e2") if luminance < 0.25 else ("#191919", "#3a3a3a")


def render_hero(cfg: Config, ctx: dict) -> str:
    if ctx["stale"]:
        return (f'<div class="notice" role="status"><p><strong>No forecast today.</strong> The newest '
                f'measurements are from {esc(short_date(ctx["base_date"]))}, so the monitors or OpenAQ are '
                f'running late. The next run will try again.</p></div>')
    if not ctx["upcoming"]:
        return ('<div class="notice" role="status"><p><strong>No forecast yet.</strong> The model needs '
                f'{cfg.min_train_days} days of history before it makes its first forecast.</p></div>')
    panels = []
    for u in ctx["upcoming"]:
        bg = HAZE[u["category"]]
        ink, ink2 = _ink(bg)
        panels.append(
            f'<li class="day" style="--bg:{bg};--ink-on:{ink};--ink-on-2:{ink2}">'
            f'<h3 class="when">{esc(u["label"])}</h3>'
            f'<p class="aqi"><span class="num">{u["aqi"]}</span><span class="unit">AQI</span></p>'
            f'<p class="cat"><span class="dot" style="background:{CATEGORY_DOT[u["category"]]}"></span>'
            f'{esc(u["category"])}</p>'
            f'<p class="pm">PM2.5 about {u["pm25"]:.0f} µg/m³<br>'
            f'likely {u["lo"]:.0f}–{u["hi"]:.0f}</p>'
            f'</li>'
        )
    return (
        '<section aria-label="Forecast">\n'
        f'<ol class="sky">{"".join(panels)}</ol>\n'
        '<p class="note">AQI here is India’s PM2.5 sub-index on the CPCB scale. The official AQI '
        'takes the worst of up to eight pollutants, and on dusty days PM10 can set it instead. '
        '“Likely” is the 80% range.</p>\n'
        '</section>'
    )


def render_trend(cfg: Config, ctx: dict) -> str:
    chart = ctx["chart"]
    live_start = chart["liveStart"]
    h0 = chart["horizon"]
    ahead = "the day before" if h0 == 1 else f"{h0} days before"
    caption = (f"The blue line is the forecast made {ahead} each day; the shaded band is its 80% range. "
               "Grid lines mark the CPCB category boundaries.")
    if live_start:
        caption += (f" Before {short_date(dt.date.fromisoformat(live_start))}, the blue line comes from the "
                    "backtest: the model replayed with only the data it would have had at the time.")
    rows = []
    forecast = {f[0]: f for f in chart["forecast"]}
    for d, v in reversed(chart["measured"]):
        f = forecast.get(d)
        rows.append(
            f"<tr><td>{esc(short_date(dt.date.fromisoformat(d)))}</td>"
            f"<td>{'' if v is None else f'{v:.0f}'}</td>"
            f"<td>{'' if not f or f[1] is None else f'{f[1]:.0f}'}</td>"
            f"<td>{'' if not f or f[2] is None else f'{f[2]:.0f}–{f[3]:.0f}'}</td>"
            f"<td>{'' if not f else ('Live' if f[4] == 'live' else 'Backtest')}</td></tr>"
        )
    future_rows = "".join(
        f"<tr><td>{esc(short_date(dt.date.fromisoformat(d)))}</td><td></td><td>{p:.0f}</td>"
        f"<td>{lo:.0f}–{hi:.0f}</td><td>Forecast</td></tr>"
        for d, p, lo, hi, _ in reversed(chart["future"])
    )
    return (
        '<h2 id="trend-h">The last 90 days</h2>\n'
        f'<p class="sub">Daily mean PM2.5 in µg/m³, measured and forecast {esc(ahead)}.</p>\n'
        '<ul class="legend" aria-hidden="true">'
        '<li><span class="key key-measured"></span>Measured</li>'
        f'<li><span class="key key-forecast"></span>Forecast made {esc(ahead)}</li>'
        '<li><span class="key key-band"></span>80% range</li></ul>\n'
        '<div id="trend-chart" class="chart" tabindex="0" role="group" '
        'aria-label="Chart of measured and forecast PM2.5. Use the left and right arrow keys to read values.">'
        '<p class="chart-fallback">Turn on JavaScript to see the chart. The numbers are in the table below.</p>'
        '</div>\n'
        f'<p class="caption">{esc(caption)}</p>\n'
        '<details><summary>Show the numbers</summary>\n'
        '<div class="table-wrap"><table><thead><tr><th scope="col">Date</th><th scope="col">Measured</th>'
        '<th scope="col">Forecast</th><th scope="col">80% range</th><th scope="col">Type</th></tr></thead>'
        f'<tbody>{future_rows}{"".join(rows)}</tbody></table></div>\n'
        '</details>'
    )


def _lead_label(h: int) -> str:
    return "1 day ahead" if h == 1 else f"{h} days ahead"


def _lead_hint(h: int) -> str:
    return {1: "Usually today", 2: "Usually tomorrow", 3: "Usually the day after tomorrow"}.get(
        h, f"{h} days after the latest measurement")


def _forecast_name(h: int) -> str:
    return "next-day forecast" if h == 1 else f"{h}-day-ahead forecast"


def render_accuracy(cfg: Config, ctx: dict) -> str:
    metrics = ctx["metrics"]
    shown = metrics["shown"]
    if not shown:
        return ('<p class="sub">Not enough history to score the model yet. The backtest runs once '
                f'{cfg.min_train_days} days of data are stored.</p>')

    h0 = min(shown)
    src, m0 = shown[h0]
    where = (f"Over the last {LIVE_WINDOW} days of live forecasts" if src == "live"
             else f"In a backtest over the last {cfg.backtest_days} days")
    skill = m0.get("skill_vs_persistence")
    if skill is not None and skill > 0.005:
        verdict = (f"the {esc(_forecast_name(h0))} missed by {m0['mae_model']:.1f} µg/m³ on average, "
                   f"{skill * 100:.0f}% less than assuming the air stays the same as the last measured day.")
    else:
        verdict = (f"the {esc(_forecast_name(h0))} missed by {m0['mae_model']:.1f} µg/m³ on average, "
                   f"no better than assuming the air stays the same as the last measured day "
                   f"({m0['mae_persistence']:.1f}).")
    headline = f'<p class="headline">{esc(where)}, {verdict}</p>'

    scale = max(max(m[f"mae_{k}"] for k in METHODS) for _, m in shown.values()) or 1.0
    blocks = []
    for h, (source, m) in sorted(shown.items()):
        bars = []
        for k in METHODS:
            value = m[f"mae_{k}"]
            bars.append(
                f'<li class="bar-row{" is-model" if k == "model" else ""}">'
                f'<span class="bar-label">{esc(METHOD_LABELS[k])}</span>'
                f'<span class="bar-track"><span class="bar" style="--w:{value / scale:.3f}"></span>'
                f'<span class="bar-value">{value:.1f}</span></span></li>'
            )
        src_text = (f"Live, last {LIVE_WINDOW} days ({m['n']} forecasts)" if source == "live"
                    else f"Backtest ({m['n']} days)")
        blocks.append(
            f'<section class="acc"><h3>{esc(_lead_label(h))}</h3>'
            f'<p class="acc-sub">{esc(_lead_hint(h))}. {esc(src_text)}.</p>'
            f'<ul class="bars">{"".join(bars)}</ul></section>'
        )

    table_rows = []
    for h in cfg.horizons:
        for label, source in (("Live, last 30 days", metrics["live30"]), ("Live, all", metrics["live_all"]),
                              ("Backtest", metrics["backtest"])):
            m = source.get(h)
            if not m:
                continue
            skill_p = m.get("skill_vs_persistence")
            cov = m.get("interval_coverage")
            table_rows.append(
                f"<tr><th scope='row'>{esc(_lead_label(h))}</th><td>{esc(label)}</td><td>{m['n']}</td>"
                f"<td>{m['mae_model']:.1f}</td><td>{m['mae_persistence']:.1f}</td><td>{m['mae_last_week']:.1f}</td>"
                f"<td>{'' if skill_p is None else f'{skill_p * 100:+.0f}%'}</td>"
                f"<td>{m['category_hit_model'] * 100:.0f}%</td>"
                f"<td>{'' if cov is None else f'{cov * 100:.0f}%'}</td></tr>"
            )
    return (
        f'{headline}\n'
        '<p class="sub">Average miss (mean absolute error) in µg/m³, lower is better, against two simple '
        'rules the model has to beat.</p>\n'
        f'<div class="acc-grid">{"".join(blocks)}</div>\n'
        '<details><summary>Show all scores</summary>\n'
        '<div class="table-wrap"><table><thead><tr><th scope="col">Lead time</th><th scope="col">Scored on</th>'
        '<th scope="col">Days</th><th scope="col">Model</th><th scope="col">Last day</th>'
        '<th scope="col">Last week</th><th scope="col">Gain</th>'
        '<th scope="col">Category right</th><th scope="col">In 80% range</th></tr></thead>'
        f'<tbody>{"".join(table_rows)}</tbody></table></div>\n'
        '<p class="note">Model, Last day and Last week are average misses in µg/m³. Gain is how much '
        'smaller the model’s miss is than “same as last day” (negative means the simple rule did better). '
        'Category right is how often the forecast landed in the measured AQI category. A well calibrated '
        '80% range contains the measured value about 80% of the time.</p>\n'
        '</details>'
    )


def render_method(cfg: Config, ctx: dict) -> str:
    n = ctx["n_stations"]
    names = ", ".join(sorted(str(s) for s in ctx["stations"]["name"].tolist()))[:600] if not ctx["stations"].empty else ""
    return (
        '<div class="method">'
        f'<p>Every morning a GitHub Action downloads the previous day’s PM2.5 from {n} government '
        f'monitor{"s" if n != 1 else ""} in {esc(cfg.city.name)} (CPCB data, via OpenAQ) and combines them '
        'into one value for the city. Monitors that read consistently high or low are scaled to the '
        'city average first, so a station going offline doesn’t make the city look cleaner.</p>'
        '<p>A gradient-boosted tree model (scikit-learn) predicts how much PM2.5 will change from the '
        'latest day. It uses the last week of readings, the weather forecast from Open-Meteo (wind, '
        'boundary layer height, humidity, rain, temperature and pressure) and the time of year. Two more '
        'models give the 80% range. All of them are retrained on every run.</p>'
        '<p>The model is scored against two simple rules: the air stays the same as the last measured '
        'day, and it is the same as the same day last week. A forecast is only useful if it beats them.</p>'
        '<p>Each forecast is written to <code>data/forecast_log.csv</code> before its day is measured and '
        'is never edited, so the live scores are what the model actually said at the time.</p>'
        + (f'<p class="note">Monitors used: {esc(names)}.</p>' if names else "")
        + '</div>'
    )


def render_footer(cfg: Config, ctx: dict) -> str:
    repo = cfg.repo_url.strip()
    repo_html = (f'<a href="{esc(repo)}">Source code and data on GitHub</a>. ' if repo else "")
    return (
        f'<p>{repo_html}Air quality data from CPCB monitors via '
        '<a href="https://openaq.org">OpenAQ</a>. '
        '<a href="https://open-meteo.com/">Weather data by Open-Meteo.com</a> (CC BY 4.0). '
        'This is a student project, not an official forecast.</p>'
    )
