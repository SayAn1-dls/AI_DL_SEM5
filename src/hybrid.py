"""Phase 2 hybrid: Kalman filter states as extra inputs to the global LSTM.

Why combine them? Each model is weak exactly where the other is strong:
  * The per-SKU Kalman filter tracks a SKU's current level within a day or two,
    but it knows nothing about the calendar and learns nothing from other SKUs.
    In Phase 1 it was competitive only in the first forecast week.
  * The global LSTM knows the calendar and shares patterns across all 40 SKUs,
    but it only sees a raw, noisy 56-day window of sales.
So the filter acts as a feature engineer for the LSTM: for every past day the
LSTM now also reads the filter's level, trend and uncertainty for that SKU.

Nothing else changes: same network, same pinball loss, same backtest. The
function `kalman_lstm` has the standard model signature
`model(wide, cal, end, horizon) -> {"point", "p10", "p90"}`.

Leakage: global_lstm() calls kalman_inputs() with the training rows only
(wide.iloc[:end]), and filtered_states() runs the filter forward in time, so
the value for day t uses sales up to day t and never anything after the origin.
One honest detail: build_model() sets the measurement noise R from the last
365 TRAINING days, so the noise level of early days is set with some later
training data. That is ordinary in-sample parameter fitting, not test leakage.
"""
from functools import partial

import numpy as np
import pandas as pd

from global_model import global_lstm
from kalman import filtered_states

KALMAN_FEATURES = ["kalman_level", "kalman_trend", "kalman_sd"]


def kalman_inputs(train_wide: pd.DataFrame) -> np.ndarray:
    """Run a level+trend Kalman filter over every SKU's training history.

    train_wide: dates x SKUs, training rows only.
    Returns (n_days, n_skus, 3): level, trend and one-step standard deviation
    for every day and SKU, all in units sold (global_lstm scales them like sales).
    """
    per_sku = [filtered_states(train_wide[sku].values) for sku in train_wide.columns]
    return np.stack(per_sku, axis=1)


# The hybrid model: the Phase 1 LSTM plus three Kalman input channels.
kalman_lstm = partial(global_lstm, extra_inputs=kalman_inputs)
