import dataclasses
import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aqi.config import load_configs  # noqa: E402

END = dt.date(2026, 11, 26)


def _shorten(cfg):
    """Short history so pipeline tests run quickly."""
    return dataclasses.replace(cfg, history_start=END - dt.timedelta(days=420),
                               backtest_days=60, backtest_retrain_every=30)


@pytest.fixture
def small_cfgs(tmp_path):
    """Every city in the real config, writing under a temporary folder."""
    return [_shorten(c.with_root(tmp_path)) for c in load_configs()]


@pytest.fixture
def small_cfg(small_cfgs):
    """Just the first city (Jaipur)."""
    return small_cfgs[0]
