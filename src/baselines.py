"""Baseline models (PRD FR-3): seasonal naive and gradient boosting.

All model functions in this project have the same interface:
    model(wide, cal, end, horizon) -> dict with "point" (horizon x n_skus array)
    and optionally "p10" / "p90".
`wide` = full sales table (dates x SKUs), `cal` = calendar features,
`end`  = forecast origin: the model may only use rows wide.iloc[:end].
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, make_feature_table

# Gradient boosting settings: common defaults, not tuned (PRD 8.2 says avoid tuning).
GBM_MAX_ITER = 200        # number of trees added one after another
GBM_LEARNING_RATE = 0.1   # each new tree corrects only 10% of the remaining error
GBM_MAX_DEPTH = 6         # limits how complex one tree can be
GBM_RANDOM_STATE = 0
# Rows of history needed to rebuild the features of one future day:
# the longest look-back is lag_28 / roll_mean_28 (28 days), plus a margin.
HISTORY_TAIL = 40


def seasonal_naive(wide: pd.DataFrame, cal: pd.DataFrame, end: int, horizon: int = 28,
                   season: int = 7) -> dict:
    """Repeat the last `season` days again and again.
    season=7  -> "same weekday last week"
    season=28 -> "same weekday 4 weeks ago" """
    last_season = wide.iloc[end - season:end].values          # (season, n_skus)
    point = np.array([last_season[h % season] for h in range(horizon)])
    return {"point": point}


def seasonal_naive_28(wide: pd.DataFrame, cal: pd.DataFrame, end: int, horizon: int = 28) -> dict:
    """Seasonal naive with a 28-day season (repeats the last 4 weeks)."""
    return seasonal_naive(wide, cal, end, horizon, season=28)


def gradient_boosting(wide: pd.DataFrame, cal: pd.DataFrame, end: int, horizon: int = 28) -> dict:
    """One pooled HistGradientBoostingRegressor for all SKUs.

    Training uses only days before `end`. Forecasting is recursive: we predict
    day 1, write the prediction into the history, then build the lag features
    for day 2 from it, and so on. So the 28 forecasts never see real future sales.
    """
    train_wide = wide.iloc[:end]

    # 1. training table (lags are built with shift, so no leakage)
    table = make_feature_table(train_wide, cal).dropna(subset=FEATURES)
    # Default loss is squared error, so the model predicts a MEAN (not a median
    # like the LSTM's P50). item_code is treated as a category, not a number.
    model = HistGradientBoostingRegressor(
        max_iter=GBM_MAX_ITER, learning_rate=GBM_LEARNING_RATE, max_depth=GBM_MAX_DEPTH,
        categorical_features=[FEATURES.index("item_code")], random_state=GBM_RANDOM_STATE)
    model.fit(table[FEATURES], table["sales"])

    # 2. recursive forecast. lag_1 of day 2 is the sales of day 1, which is
    # unknown at the origin, so our own prediction for day 1 is used instead.
    # The cost: errors can build up over the 28 days.
    history = train_wide.copy()
    point = []
    for h in range(horizon):
        day = wide.index[end + h]
        history.loc[day] = np.nan                       # tomorrow is unknown
        rows = make_feature_table(history.tail(HISTORY_TAIL), cal)
        rows = rows[rows["date"] == day]                # only the day we predict
        pred = np.clip(model.predict(rows[FEATURES]), 0, None)
        history.loc[day] = pred                         # feed the prediction back
        point.append(pred)
    return {"point": np.array(point)}
