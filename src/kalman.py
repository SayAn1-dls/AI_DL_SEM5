"""Local state model: one Kalman filter per SKU (PRD FR-4, FR-7).

State-space model
    hidden state  x_t = [level, trend]
    x_t = F x_{t-1} + process noise  (covariance Q)
    y_t = H x_t     + measurement noise (variance R)

For the "local linear trend" model:
    level_t = level_{t-1} + trend_{t-1} + noise
    trend_t = trend_{t-1} + noise
The weekly version adds 6 seasonal states so the filter can learn day-of-week effects.
"""
import numpy as np

Z_P10_P90 = 1.2816   # normal distribution: 10% / 90% points are mean -/+ 1.2816 sd

NOISE_WINDOW_DAYS = 365        # R is measured on the last year of training data
MIN_MEASUREMENT_VAR = 0.1      # floor for R, so a near-constant series cannot give R = 0
INITIAL_STATE_VARIANCE = 100.0 # P0: "we know almost nothing at the start" (a diffuse prior)


class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x, self.P = x0.copy(), P0.copy()

    def update(self, y):
        """One step: predict the new state, then correct it with observation y."""
        # predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # correct
        S = self.H @ P_pred @ self.H.T + self.R            # innovation variance
        K = P_pred @ self.H.T / S                          # Kalman gain
        self.x = x_pred + (K * (y - self.H @ x_pred)).ravel()
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred
        # How uncertain today's prediction was BEFORE seeing y (in units sold^2).
        # Stored for the hybrid, which feeds it to the LSTM.
        self.last_S = float(S.item())

    def filter(self, series):
        """Run update() over a whole series, oldest day first."""
        for y in series:
            self.update(y)

    def forecast(self, horizon):
        """Mean and variance of y for the next `horizon` days (no new data)."""
        means, variances = [], []
        x, P = self.x.copy(), self.P.copy()
        for _ in range(horizon):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            means.append((self.H @ x).item())
            variances.append((self.H @ P @ self.H.T + self.R).item())
        return np.array(means), np.array(variances)


def build_model(train, weekly=False, q_level=0.05, q_trend=0.0005, q_season=0.001):
    """Create the filter for one SKU. Noise levels come from a simple heuristic
    (no optimiser): R = variance of sales around the weekday average over the
    last year; Q = fixed small fractions of R."""
    # R (measurement noise) = how much daily sales scatter around the usual
    # level for that weekday. Removing the weekday average first means the
    # weekly pattern is not mistaken for noise.
    recent = train[-NOISE_WINDOW_DAYS:]
    weekday_mean = np.array([recent[i::7].mean() for i in range(7)])
    residual = recent - np.tile(weekday_mean, len(recent) // 7 + 1)[:len(recent)]
    R = max(residual.var(), MIN_MEASUREMENT_VAR)

    # Q (process noise) is a small fraction of R: the level may drift a little
    # each day, the trend much less. Small Q/R = smooth, slow filter;
    # large Q/R = jumpy filter that follows every sale.
    if not weekly:
        F = np.array([[1.0, 1.0],     # level_t = level_{t-1} + trend_{t-1}
                      [0.0, 1.0]])    # trend_t = trend_{t-1}
        H = np.array([[1.0, 0.0]])    # we observe the level (plus noise)
        Q = np.diag([q_level * R, q_trend * R])
        x0 = np.array([train[:7].mean(), 0.0])     # start at the first week's average, flat
        P0 = np.eye(2) * INITIAL_STATE_VARIANCE
    else:
        n = 8                                   # level, trend, 6 seasonal states
        F = np.zeros((n, n))
        F[0, 0] = F[0, 1] = F[1, 1] = 1.0       # same level/trend rows as above
        # The 7 weekday effects must add up to 0, so today's effect is minus the
        # sum of the previous 6. Storing 6 states is enough to know all 7.
        F[2, 2:] = -1.0
        for i in range(3, n):                   # shift yesterday's effects down one slot
            F[i, i - 1] = 1.0
        H = np.zeros((1, n))
        H[0, 0] = H[0, 2] = 1.0                 # observation = level + today's weekday effect
        Q = np.diag([q_level * R, q_trend * R, q_season * R] + [0.0] * 5)
        x0 = np.zeros(n)
        x0[0] = train[:7].mean()
        P0 = np.eye(n) * INITIAL_STATE_VARIANCE
    return KalmanFilter(F, H, Q, np.array([[R]]), x0, P0)


def filtered_states(series: np.ndarray, weekly: bool = False) -> np.ndarray:
    """Day-by-day output of the filter for one SKU, used by the hybrid.

    Returns a (n_days, 3) array. Row t holds what the filter believes after
    seeing day t: the level, the trend, and the standard deviation of its
    one-step-ahead prediction for day t (sqrt of the innovation variance S).

    This is the FILTER, not the smoother: row t uses days 0..t only. A smoother
    would revise day t using later days, which would leak the future.
    """
    kf = build_model(series, weekly=weekly)
    states = np.empty((len(series), 3))
    for t, y in enumerate(series):
        kf.update(y)
        states[t] = kf.x[0], kf.x[1], np.sqrt(kf.last_S)
    return states


def kalman_forecast(wide, cal, end, horizon=28, weekly=False):
    """Run an independent filter for every SKU, then forecast `horizon` days."""
    train = wide.iloc[:end]
    point, p10, p90 = [], [], []
    for item in wide.columns:
        y = train[item].values
        kf = build_model(y, weekly=weekly)
        kf.filter(y)
        mean, var = kf.forecast(horizon)
        # Gaussian interval from the forecast variance. Sales cannot be
        # negative, so everything is clipped at 0 afterwards.
        sd = np.sqrt(var)
        point.append(np.clip(mean, 0, None))
        p10.append(np.clip(mean - Z_P10_P90 * sd, 0, None))
        p90.append(np.clip(mean + Z_P10_P90 * sd, 0, None))
    return {"point": np.array(point).T, "p10": np.array(p10).T, "p90": np.array(p90).T}


def kalman_llt(wide, cal, end, horizon=28):
    """PRD model: local linear trend (level + trend)."""
    return kalman_forecast(wide, cal, end, horizon, weekly=False)


def kalman_llt_weekly(wide, cal, end, horizon=28):
    """Extension: local linear trend + weekly seasonality."""
    return kalman_forecast(wide, cal, end, horizon, weekly=True)
