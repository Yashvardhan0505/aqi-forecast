"""Gradient-boosted trees, one set per horizon: a point forecast and an 80% range.

scikit-learn's HistGradientBoostingRegressor handles missing values natively, which
matters here: stations go offline and some weather variables aren't always served.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import feature_columns

# Early stopping holds out 15% of the training rows and stops adding trees once
# they stop helping: about 3x faster than a fixed 250 trees, with the same accuracy
# and better-calibrated ranges in the backtest.
PARAMS = dict(
    learning_rate=0.08,
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=15,
    max_leaf_nodes=15,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=0,
)


@dataclass
class HorizonModel:
    horizon: int
    features: list[str]
    point: HistGradientBoostingRegressor
    lower: HistGradientBoostingRegressor
    upper: HistGradientBoostingRegressor

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """PM2.5 forecasts (ug/m3) with a lower/upper bound for each row of `frame`."""
        X = frame.reindex(columns=self.features).to_numpy(dtype=float)
        anchor = frame["lag0"].to_numpy(dtype=float)  # log1p(PM2.5) on the base day
        point = np.expm1(anchor + self.point.predict(X))
        lo = np.expm1(anchor + self.lower.predict(X))
        hi = np.expm1(anchor + self.upper.predict(X))
        # Quantile models are fit separately and can cross; keep lo <= point <= hi.
        lo, hi = np.minimum(lo, point), np.maximum(hi, point)
        return pd.DataFrame(
            {"pred": np.clip(point, 0, None), "lo": np.clip(lo, 0, None), "hi": np.clip(hi, 0, None)},
            index=frame.index,
        )


def fit(train: pd.DataFrame, horizon: int, interval: tuple[float, float]) -> HorizonModel:
    train = train[train["target"].notna()]
    # Skip features missing on most training days (e.g. a weather variable that only
    # the recent forecasts include): the model would barely learn them anyway.
    features = [c for c in feature_columns(train) if train[c].notna().mean() >= 0.5]
    X = train[features].to_numpy(dtype=float)
    z = train["target"].to_numpy(dtype=float)
    point = HistGradientBoostingRegressor(loss="squared_error", **PARAMS).fit(X, z)
    lower = HistGradientBoostingRegressor(loss="quantile", quantile=interval[0], **PARAMS).fit(X, z)
    upper = HistGradientBoostingRegressor(loss="quantile", quantile=interval[1], **PARAMS).fit(X, z)
    return HorizonModel(horizon, features, point, lower, upper)
