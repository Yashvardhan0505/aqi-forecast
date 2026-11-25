"""Rolling-origin backtest: replay the last N days as if forecasting live.

The model is refit every `retrain_every` days using only rows whose target was
already measured on the refit day, then used for the following block of days.

One honest caveat: for past days the stored weather is the short-range archive,
not the 2-3 day-ahead forecast the live model gets, so the backtest is a little
optimistic at longer horizons. The live log (tracking.py) has no such shortcut.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import features, model
from .config import Config
from .evaluation import summarize

log = logging.getLogger(__name__)

PAIR_COLUMNS = ["base_date", "target_date", "horizon", "actual", "model", "lo", "hi",
                "persistence", "last_week"]


def run(cfg: Config, city: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    parts = []
    step = pd.Timedelta(days=cfg.backtest_retrain_every)
    for h in cfg.horizons:
        frame = features.build_frame(city, weather, h)
        known = frame[frame["actual"].notna()]
        if known.empty:
            continue
        first = known.index.max() - pd.Timedelta(days=cfg.backtest_days - 1)
        test = known[known.index >= first]

        for start in pd.date_range(first, known.index.max(), freq=step):
            block = test[(test.index >= start) & (test.index < start + step)]
            if block.empty:
                continue
            train = frame[(frame["target_date"] <= start) & frame["target"].notna()]
            if len(train) < cfg.min_train_days:
                continue
            pred = model.fit(train, h, cfg.interval).predict(block)
            parts.append(pd.DataFrame({
                "base_date": block.index.strftime("%Y-%m-%d"),
                "target_date": block["target_date"].dt.strftime("%Y-%m-%d"),
                "horizon": h,
                "actual": block["actual"].to_numpy(),
                "model": pred["pred"].to_numpy(),
                "lo": pred["lo"].to_numpy(),
                "hi": pred["hi"].to_numpy(),
                "persistence": block["anchor"].to_numpy(),
                "last_week": block["last_week_value"].to_numpy(),
            }))

    if not parts:
        log.warning("Backtest skipped: not enough history yet (need %d days)", cfg.min_train_days)
        return pd.DataFrame(columns=PAIR_COLUMNS), []
    pairs = pd.concat(parts, ignore_index=True).round({"model": 1, "lo": 1, "hi": 1})
    metrics = summarize(pairs)
    for m in metrics:
        log.info("Backtest h=%d: MAE model %.1f | same as last day %.1f | same as last week %.1f (n=%d)",
                 m["horizon"], m["mae_model"], m["mae_persistence"], m["mae_last_week"], m["n"])
    return pairs, metrics
