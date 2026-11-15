"""Leakage-safe features (PRD FR-2).

Every lag / rolling feature for day t only looks at days BEFORE t.
We get this by shifting the table first (shift(k) moves values k days down)
and only then taking rolling means.
"""
import numpy as np
import pandas as pd

LAGS = [1, 7, 14, 28]   # yesterday, and the same weekday 1, 2 and 4 weeks ago (PRD FR-2)
WINDOWS = [7, 28]       # rolling means over the last week and the last 4 weeks

CALENDAR_FEATURES = ["dayofweek", "month", "is_weekend", "is_event", "snap"]
LAG_FEATURES = [f"lag_{k}" for k in LAGS] + [f"roll_mean_{w}" for w in WINDOWS]
FEATURES = LAG_FEATURES + CALENDAR_FEATURES + ["item_code"]


def make_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per date with calendar features (these are known in advance)."""
    cal = panel.drop_duplicates("date").set_index("date").sort_index()
    out = pd.DataFrame(index=cal.index)
    out["dayofweek"] = out.index.dayofweek          # 0 = Monday
    out["month"] = out.index.month
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["is_event"] = cal["is_event"]
    out["snap"] = cal["snap"]
    return out


def make_lag_features(wide: pd.DataFrame) -> pd.DataFrame:
    """wide: dates x SKUs sales table. Returns a long table with lag/rolling
    features and the target `sales`, one row per (date, SKU)."""
    parts = {}
    for k in LAGS:
        parts[f"lag_{k}"] = wide.shift(k)
    for w in WINDOWS:
        # shift(1) first -> the window for day t covers days t-w ... t-1
        parts[f"roll_mean_{w}"] = wide.shift(1).rolling(w).mean()
    parts["sales"] = wide

    index = pd.MultiIndex.from_product([wide.index, wide.columns], names=["date", "item_id"])
    table = pd.DataFrame({name: df.values.ravel() for name, df in parts.items()}, index=index)
    return table.reset_index()


def make_feature_table(wide: pd.DataFrame, cal_feats: pd.DataFrame) -> pd.DataFrame:
    """Lag features + calendar features + item code = full model input table.
    item_code is a whole number per SKU, which gradient boosting treats as a
    category so one pooled model can still learn SKU-specific levels."""
    table = make_lag_features(wide)
    table = table.merge(cal_feats.reset_index(), on="date", how="left")
    item_codes = {item: i for i, item in enumerate(wide.columns)}
    table["item_code"] = table["item_id"].map(item_codes)
    return table
