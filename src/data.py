"""Loading, cleaning and filtering the M5 data (PRD stage 1: Prepare).

The raw M5 files are huge (30,490 series). We only keep a small subset:
one store + one category, about 40 SKUs, and turn it into a "long" table
with one row per (SKU, day).
"""
import numpy as np
import pandas as pd

STORE = "CA_1"
CATEGORY = "FOODS"
N_SKUS = 40
MIN_MEAN_SALES = 0.3     # drop SKUs that almost never sell
MAX_FIRST_SALE_DAY = 200  # SKU must start selling early -> long history


def load_raw(raw_dir="../data/raw"):
    """Read the three M5 csv files."""
    calendar = pd.read_csv(f"{raw_dir}/calendar.csv")
    sales = pd.read_csv(f"{raw_dir}/sales_train_evaluation.csv")
    prices = pd.read_csv(f"{raw_dir}/sell_prices.csv")
    return calendar, sales, prices


def select_skus(sales, store=STORE, category=CATEGORY, n_skus=N_SKUS):
    """Pick n_skus SKUs from one store/category.

    Rules (all explained in the notebook):
    1. same store and same category
    2. first sale happens in the first 200 days (so history is long)
    3. average daily sales >= 0.3 (otherwise the series is almost all zeros)
    4. from the remaining SKUs take evenly spaced ones after sorting by
       average sales -> we get a mix of fast sellers and slow/intermittent ones
    """
    sub = sales[(sales["store_id"] == store) & (sales["cat_id"] == category)].copy()
    day_cols = [c for c in sub.columns if c.startswith("d_")]
    values = sub[day_cols].values

    sub["first_sale_day"] = (values > 0).argmax(axis=1) + 1
    sub["mean_sales"] = values.mean(axis=1)

    ok = sub[(sub["first_sale_day"] <= MAX_FIRST_SALE_DAY) & (sub["mean_sales"] >= MIN_MEAN_SALES)]
    ok = ok.sort_values("mean_sales").reset_index(drop=True)

    # evenly spaced positions in the sorted list; np.unique drops repeats that
    # would appear if fewer than n_skus SKUs passed the filters
    positions = np.unique(np.linspace(0, len(ok) - 1, n_skus).round().astype(int))
    chosen = ok.iloc[positions]
    return chosen.drop(columns=["first_sale_day", "mean_sales"])


def build_panel(chosen_sales, calendar, prices):
    """Wide sales table -> clean long panel with calendar + price + flags."""
    day_cols = [c for c in chosen_sales.columns if c.startswith("d_")]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # 1. wide -> long (one row per SKU per day)
    panel = chosen_sales.melt(id_vars=id_cols, value_vars=day_cols,
                              var_name="d", value_name="sales")

    # 2. attach calendar (d_1 -> real date, weekday, events, SNAP)
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])
    panel = panel.merge(calendar, on="d", how="left")

    # event columns: NaN just means "no event that day"
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        panel[col] = panel[col].fillna("None")
    panel["is_event"] = (panel["event_name_1"] != "None").astype(int)

    # SNAP (food stamp) day for the state of the store
    state = panel["state_id"].iloc[0]
    panel["snap"] = panel[f"snap_{state}"]
    panel = panel.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    # 3. attach weekly selling price (missing price = item not on sale that week)
    panel = panel.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    panel = panel.sort_values(["item_id", "date"]).reset_index(drop=True)

    # 4. availability PROXY (not a real stockout label!)
    panel["price_missing"] = panel["sell_price"].isna().astype(int)
    panel["potential_unavailable"] = ((panel["sales"] == 0) & (panel["price_missing"] == 1)).astype(int)

    # price used as a feature: fill gaps with the last known price of that SKU
    panel["sell_price_filled"] = panel.groupby("item_id")["sell_price"].ffill()

    panel["sales"] = panel["sales"].astype("int16")
    return panel


def check_panel(panel):
    """Simple data-quality checks. Raises AssertionError if something is wrong."""
    assert panel.duplicated(["item_id", "date"]).sum() == 0, "duplicate rows"
    assert (panel["sales"] >= 0).all(), "negative sales"
    counts = panel.groupby("item_id")["date"].agg(["count", "min", "max", "nunique"])
    n_days = panel["date"].nunique()
    assert (counts["count"] == n_days).all(), "some SKU has missing days"
    full_index = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    assert len(full_index) == n_days, "date index has gaps"
    return counts


def to_wide(panel):
    """Panel -> table with dates as rows and SKUs as columns (sales only)."""
    return panel.pivot(index="date", columns="item_id", values="sales").astype(float)


def load_clean_panel(path="../data/processed/m5_clean_panel.csv"):
    """Read the cleaned panel saved by notebook 01."""
    return pd.read_csv(path, parse_dates=["date"])


def demand_class(series, adi_cut=1.32, cv2_cut=0.49):
    """Syntetos-Boylan demand classification.
    ADI = average gap (in days) between sales, CV2 = squared coefficient of
    variation of the non-zero sales sizes.
      smooth       ADI < 1.32 and CV2 < 0.49
      erratic      ADI < 1.32 and CV2 >= 0.49
      intermittent ADI >= 1.32 and CV2 < 0.49
      lumpy        ADI >= 1.32 and CV2 >= 0.49
    Zeros before the first sale (item not launched yet) are ignored.
    Returns (class_name, adi, cv2)."""
    series = np.asarray(series, dtype=float)
    if not (series > 0).any():
        return "no sales", np.inf, np.nan     # never sold: ADI and CV2 are undefined
    series = series[np.argmax(series > 0):]
    nonzero = series[series > 0]
    adi = len(series) / len(nonzero)
    cv2 = (nonzero.std() / nonzero.mean()) ** 2
    if adi < adi_cut:
        name = "smooth" if cv2 < cv2_cut else "erratic"
    else:
        name = "intermittent" if cv2 < cv2_cut else "lumpy"
    return name, adi, cv2


def weekly_strength(series):
    """Share of the variance explained by the day-of-week (needs a Series with a
    DatetimeIndex). Close to 0 = no weekly pattern, close to 1 = very strong.
    Pre-launch zeros are ignored."""
    series = series.iloc[int(np.argmax(series.values > 0)):]
    weekday_mean = series.groupby(series.index.dayofweek).transform("mean")
    return float(weekday_mean.var() / series.var())
