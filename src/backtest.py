"""Rolling-origin backtest (PRD FR-6, FR-8) and metric tables.

Every model is run on exactly the same SKUs, the same forecast origins and the
same 28-day horizon. All forecasts are stored in one tidy table.
"""
import numpy as np
import pandas as pd

import metrics

HORIZON = 28
N_ORIGINS = 4
MIN_HISTORY = 56   # SKUs with fewer days of selling history use the cold-start fallback


def get_origins(n_days: int, horizon: int = HORIZON, n_origins: int = N_ORIGINS) -> list:
    """Position of each forecast origin. The model may use wide.iloc[:origin].
    The last origin leaves exactly `horizon` days to forecast; earlier origins
    are spaced `horizon` days apart, so the 4 test windows never overlap."""
    return [n_days - horizon * (n_origins - i) for i in range(n_origins)]


def history_length(wide: pd.DataFrame, end: int) -> np.ndarray:
    """Days since the first non-zero sale of every SKU (0 if it never sold)."""
    values = wide.iloc[:end].values
    first_sale = (values > 0).argmax(axis=0)
    never_sold = (values > 0).sum(axis=0) == 0
    length = end - first_sale
    length[never_sold] = 0
    return length


def cold_start_forecast(wide: pd.DataFrame, end: int, horizon: int = HORIZON,
                        min_history: int = MIN_HISTORY) -> np.ndarray:
    """Fallback for SKUs with too little history: for each weekday take the
    MEDIAN daily demand of the other (established) SKUs of the same store/category
    over the last 28 days."""
    established = history_length(wide, end) >= min_history
    last_28 = wide.iloc[end - 28:end].loc[:, established]
    medians = {dow: np.median(last_28[last_28.index.dayofweek == dow].values)
               for dow in range(7)}
    future_days = wide.index[end:end + horizon]
    return np.array([medians[d.dayofweek] for d in future_days])


def run_backtest(wide: pd.DataFrame, cal: pd.DataFrame, models: dict, origins: list,
                 horizon: int = HORIZON, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """models: dict name -> forecast function. Returns the tidy predictions table.

    Every model gets the same `wide`, `cal`, origins and horizon (PRD fairness
    rule). The full table is passed, but a model may only read wide.iloc[:end];
    the leakage test in notebook 02 (future sales set to 999) checks this.
    """
    tables = []
    for end in origins:
        dates = wide.index[end:end + horizon]
        actual = wide.iloc[end:end + horizon].values
        cold = history_length(wide, end) < min_history
        fallback = cold_start_forecast(wide, end, horizon, min_history) if cold.any() else None

        for name, model in models.items():
            out = model(wide, cal, end, horizon)
            point = out["point"].copy()
            p10 = out.get("p10", np.full_like(point, np.nan)).copy()
            p90 = out.get("p90", np.full_like(point, np.nan)).copy()
            for j in np.where(cold)[0]:        # cold-start rule
                point[:, j] = fallback
                p10[:, j] = p90[:, j] = np.nan

            n_items = len(wide.columns)
            tables.append(pd.DataFrame({
                "origin": wide.index[end - 1],
                "date": np.repeat(dates, n_items),
                "step": np.repeat(np.arange(1, horizon + 1), n_items),
                "item_id": np.tile(wide.columns, horizon),
                "model": name,
                "actual": actual.ravel(),
                "y_pred": point.ravel(),
                "p10": p10.ravel(),
                "p90": p90.ravel(),
                "cold_start": np.tile(cold, horizon),
            }))
    return pd.concat(tables, ignore_index=True)


def summarize(preds: pd.DataFrame, by: tuple = ("model",)) -> pd.DataFrame:
    """Metric table. Point metrics use y_pred (= P50 for probabilistic models).
    Pinball / coverage are only computed where P10 and P90 exist."""
    rows = []
    for key, g in preds.groupby(list(by)):
        y, yhat = g["actual"].values, g["y_pred"].values
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        row.update(MAE=metrics.mae(y, yhat), RMSE=metrics.rmse(y, yhat), WAPE=metrics.wape(y, yhat))
        # Interval metrics use only the rows that have an interval. Point-only
        # models (naive, gradient boosting) have none; cold-start SKUs get the
        # fallback forecast, which has no interval either.
        has_interval = (g["p10"].notna() & g["p90"].notna()).values
        if has_interval.any():
            yi, mid = y[has_interval], yhat[has_interval]
            low, high = g["p10"].values[has_interval], g["p90"].values[has_interval]
            row["pinball_0.1"] = metrics.pinball(yi, low, 0.1)
            row["pinball_0.5"] = metrics.pinball(yi, mid, 0.5)
            row["pinball_0.9"] = metrics.pinball(yi, high, 0.9)
            row["pinball_avg"] = np.mean([row["pinball_0.1"], row["pinball_0.5"], row["pinball_0.9"]])
            row["coverage_P10_P90"] = metrics.coverage(yi, low, high)
        rows.append(row)
    return pd.DataFrame(rows)
