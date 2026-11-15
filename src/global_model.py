"""Global probabilistic model (PRD FR-5, FR-7): ONE LSTM for all SKUs, trained
with quantile (pinball) loss to output P10 / P50 / P90 for the next 28 days.

This is the PRD's accepted fallback to DeepAR. Like DeepAR it
  * is ONE model trained on all SKUs together (shared patterns),
  * uses the item id as a static categorical feature (embedding),
  * uses calendar fields known in advance as dynamic features.
Unlike DeepAR it predicts the three quantiles directly, for all 28 days in one
pass (the MQ-RNN idea), instead of sampling the future one day at a time.

Phase 2: `global_lstm` has switches used by the ablation study
(src/ablation.py) and an `extra_inputs` hook used by the Kalman-LSTM hybrid
(src/hybrid.py). With default arguments it is exactly the Phase 1 model.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

QUANTILES = [0.1, 0.5, 0.9]   # P10, P50 (used as the point forecast), P90
LOOKBACK = 56                 # default days of history the LSTM reads (8 full weeks)
N_COV = 5                     # calendar features per day, see make_covariates()

# Network size and training settings. None of these were tuned: they are small,
# common defaults so that one fit takes about a minute on a laptop CPU.
HIDDEN_SIZE = 32       # size of the LSTM memory (hidden state h and cell state c)
EMBEDDING_DIM = 4      # each SKU is described by 4 learned numbers
HEAD_SIZE = 64         # width of the hidden layer in the output network
BATCH_SIZE = 256
LEARNING_RATE = 1e-3   # Adam's usual default
MAX_EPOCHS = 20
PATIENCE = 3           # early stopping: stop after 3 epochs without improvement

LOSSES = ("pinball", "mse")

# A function that receives the TRAINING sales only (dates x SKUs, rows before the
# origin) and returns extra per-day inputs with shape (n_train_days, n_skus, k).
ExtraInputs = Callable[[pd.DataFrame], np.ndarray]


def make_covariates(cal: pd.DataFrame) -> np.ndarray:
    """Calendar features that are known in advance, as a (n_days, 5) array.

    The weekday is encoded as a point on a circle (sin, cos) instead of the
    numbers 0..6, so Sunday (6) and Monday (0) end up next to each other.
    """
    dow = cal["dayofweek"].values
    cov = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    cal["is_weekend"].values, cal["is_event"].values, cal["snap"].values], axis=1)
    return cov.astype(np.float32)


class GlobalLSTM(nn.Module):
    """past window --LSTM--> summary h
    [h, SKU embedding, future calendar] --small MLP--> (horizon x n_outputs)

    n_outputs is 3 (P10, P50, P90) for pinball loss and 1 for MSE loss.
    """

    def __init__(self, n_items: int, horizon: int = 28, n_inputs: int = 1 + N_COV,
                 n_outputs: int = len(QUANTILES), use_embedding: bool = True,
                 use_future_cov: bool = True, hidden: int = HIDDEN_SIZE,
                 emb_dim: int = EMBEDDING_DIM):
        super().__init__()
        self.horizon = horizon
        self.n_outputs = n_outputs
        self.use_future_cov = use_future_cov

        # The creation order (LSTM, embedding, head) decides which random numbers
        # each layer's starting weights get. Keep it, so that the default model
        # reproduces the Phase 1 results exactly.
        self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
        self.embedding = nn.Embedding(n_items, emb_dim) if use_embedding else None

        head_inputs = hidden
        if use_embedding:
            head_inputs += emb_dim
        if use_future_cov:
            head_inputs += horizon * N_COV
        self.head = nn.Sequential(
            nn.Linear(head_inputs, HEAD_SIZE), nn.ReLU(),
            nn.Linear(HEAD_SIZE, horizon * n_outputs))

    def forward(self, past: torch.Tensor, future_cov: torch.Tensor,
                item: torch.Tensor) -> torch.Tensor:
        # past: (batch, lookback, n_inputs), future_cov: (batch, horizon, N_COV), item: (batch,)
        # h is the hidden state after reading the LAST past day: (1 layer, batch, hidden).
        # h[-1] takes that one layer, giving a (batch, hidden) summary of the window.
        _, (h, _) = self.lstm(past)
        parts = [h[-1]]
        if self.embedding is not None:
            parts.append(self.embedding(item))            # which SKU is this?
        if self.use_future_cov:
            parts.append(future_cov.flatten(1))           # the 28 future calendar days, 28*5 numbers
        out = self.head(torch.cat(parts, dim=1))
        return out.view(-1, self.horizon, self.n_outputs)


def make_windows(y: np.ndarray, cov: np.ndarray, starts, horizon: int,
                 lookback: int = LOOKBACK, extra: Optional[np.ndarray] = None):
    """Cut training/prediction windows. For each forecast start t and each SKU:
        past   = days t-lookback ... t-1   (sales, extra inputs, calendar)
        target = days t ... t+horizon-1    (sales only)

    y: (n_days, n_items) sales, cov: (n_days, N_COV) calendar,
    extra: optional (n_days, n_items, k) extra inputs (the hybrid's Kalman states).

    Sales are divided by (1 + mean of the past window), so big and small SKUs
    look alike to the network (DeepAR scales the same way). The "+1" stops a
    division by zero when a window has no sales at all. Extra inputs measured
    in units sold (Kalman level, trend, uncertainty) get the same scale.

    Returns (tensors, scale): tensors = [past, future_cov, item] plus the target
    when the future is known; scale is needed to turn predictions back into units.
    """
    n_items = y.shape[1]
    y_win = sliding_window_view(y, lookback, axis=0)            # (n_days-L+1, n_items, L)
    cov_win = sliding_window_view(cov, lookback, axis=0)        # (n_days-L+1, N_COV, L)
    starts = np.asarray(starts)

    past_y = y_win[starts - lookback]                           # (n_t, n_items, L)
    scale = 1.0 + past_y.mean(axis=2)                           # (n_t, n_items)
    past_y = past_y / scale[:, :, None]
    channels = [past_y[..., None]]                              # sales: 1 channel

    if extra is not None:
        extra_win = sliding_window_view(extra, lookback, axis=0)  # (n_days-L+1, n_items, k, L)
        past_extra = extra_win[starts - lookback].transpose(0, 1, 3, 2)   # (n_t, n_items, L, k)
        channels.append((past_extra / scale[:, :, None, None]).astype(np.float32))

    # the calendar is the same for every SKU, so it is copied once per SKU
    past_cov = np.repeat(cov_win[starts - lookback].transpose(0, 2, 1)[:, None], n_items, axis=1)
    channels.append(past_cov)
    past = np.concatenate(channels, axis=3)                     # (n_t, n_items, L, n_inputs)

    fut_idx = starts[:, None] + np.arange(horizon)[None, :]
    future_cov = np.repeat(cov[fut_idx][:, None], n_items, axis=1)  # (n_t, n_items, H, N_COV)
    item = np.tile(np.arange(n_items), (len(starts), 1))

    if starts.max() + horizon <= len(y):
        target = np.stack([y[fut_idx[i]] for i in range(len(starts))])   # (n_t, H, n_items)
        target = target.transpose(0, 2, 1) / scale[:, :, None]
    else:
        target = None   # prediction time: the future is unknown

    # merge the (window start, SKU) axes into one batch axis
    flat = lambda a: a.reshape(-1, *a.shape[2:])
    tensors = [torch.tensor(flat(past), dtype=torch.float32),
               torch.tensor(flat(future_cov), dtype=torch.float32),
               torch.tensor(item.ravel(), dtype=torch.long)]
    if target is not None:
        tensors.append(torch.tensor(target.reshape(-1, horizon), dtype=torch.float32))
    return tensors, scale.ravel()


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Average pinball loss over the 3 quantiles. pred: (B, H, 3), target: (B, H).

    For quantile q an under-forecast costs q per unit and an over-forecast costs
    (1 - q). For q = 0.9 under-forecasting is 9 times more expensive, so the
    best P90 sits high: about 90% of actuals fall below it.
    """
    q = torch.tensor(QUANTILES).view(1, 1, -1)
    diff = target.unsqueeze(-1) - pred
    return torch.max(q * diff, (q - 1) * diff).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on the single output. pred: (B, H, 1), target: (B, H).
    Used only by the ablation: MSE is minimised by the MEAN, gives no intervals."""
    return ((pred[..., 0] - target) ** 2).mean()


def global_lstm(wide: pd.DataFrame, cal: pd.DataFrame, end: int, horizon: int = 28,
                max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, seed: int = 0, *,
                lookback: int = LOOKBACK, use_embedding: bool = True,
                use_future_cov: bool = True, loss: str = "pinball",
                extra_inputs: Optional[ExtraInputs] = None) -> dict:
    """Train on data before `end`, forecast the next `horizon` days.

    Ablation switches (defaults = the Phase 1 model):
      lookback        days of history per window
      use_embedding   False removes the SKU embedding
      use_future_cov  False removes the future calendar from the output network
      loss            "pinball" (P10/P50/P90) or "mse" (point forecast only)
      extra_inputs    function adding per-day input channels (the hybrid)
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {LOSSES}, got {loss!r}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    y = wide.values.astype(np.float32)
    cov = make_covariates(cal.loc[wide.index])
    # Extra inputs are computed from the training rows only, so by construction
    # they cannot contain anything from on or after the forecast origin.
    extra = None
    if extra_inputs is not None:
        extra = np.asarray(extra_inputs(wide.iloc[:end]), dtype=np.float32)
    n_extra = 0 if extra is None else extra.shape[2]

    # train windows: targets end at least 28 days before the origin
    # validation windows: the 28 most recent starts, whose targets end inside the
    # last 28 days before the origin (the most "test-like" data we may use)
    train_starts = np.arange(lookback, end - 2 * horizon + 1)
    val_starts = np.arange(end - 2 * horizon + 1, end - horizon + 1)
    train_t, _ = make_windows(y[:end], cov[:end], train_starts, horizon, lookback, extra)
    val_t, _ = make_windows(y[:end], cov[:end], val_starts, horizon, lookback, extra)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*train_t), batch_size=BATCH_SIZE, shuffle=True)

    n_outputs = len(QUANTILES) if loss == "pinball" else 1
    model = GlobalLSTM(n_items=y.shape[1], horizon=horizon, n_inputs=1 + n_extra + N_COV,
                       n_outputs=n_outputs, use_embedding=use_embedding,
                       use_future_cov=use_future_cov)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = pinball_loss if loss == "pinball" else mse_loss

    best_loss, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        for past, fut, item, target in train_loader:
            optimizer.zero_grad()                      # forget last batch's gradients
            batch_loss = loss_fn(model(past, fut, item), target)
            batch_loss.backward()                      # backpropagation
            optimizer.step()                           # Adam weight update
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(*val_t[:3]), val_t[3]).item()
        if val_loss < best_loss:                       # early stopping: remember the best epoch
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)                  # go back to the best epoch's weights

    # forecast: one window per SKU starting at the origin
    model.eval()
    # (sales are cut at `end`; only the calendar is allowed to look ahead)
    pred_t, scale = make_windows(y[:end], cov[:end + horizon], [end], horizon, lookback, extra)
    with torch.no_grad():
        out = model(*pred_t[:3]).numpy()               # (n_items, H, n_outputs)
    # Back to units sold. Sorting the 3 outputs fixes "quantile crossing" (the
    # network is not forced to keep P10 <= P50 <= P90). Sales cannot be negative.
    out = np.sort(out * scale[:, None, None], axis=2)
    out = np.clip(out, 0, None)
    if loss == "mse":
        return {"point": out[:, :, 0].T}
    return {"p10": out[:, :, 0].T, "point": out[:, :, 1].T, "p90": out[:, :, 2].T}
