"""Forecast accuracy metrics: MAE, RMSE, WAPE, pinball loss, coverage."""
import numpy as np


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error, in units sold. Minimised by the median forecast."""
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean squared error, in units sold. Squaring makes big misses count
    much more. Minimised by the mean forecast."""
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Sum of absolute errors / sum of actuals. Unlike MAPE it does not divide
    by each day's actual, so zero-sales days do not break it.
    If the actuals add up to 0 the ratio is undefined, so we return NaN
    (these windows are flagged instead of giving a misleading number)."""
    total = np.sum(np.abs(y))
    if total == 0:
        return np.nan
    return float(np.sum(np.abs(y - yhat)) / total)


def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for quantile level q.
    Under-forecasting is punished with weight q, over-forecasting with 1-q."""
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of actual values that fall inside [low, high].
    For a P10-P90 interval the target is 0.80."""
    return float(np.mean((y >= low) & (y <= high)))
