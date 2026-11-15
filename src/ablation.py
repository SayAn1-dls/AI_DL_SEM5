"""Phase 2 ablation study: what does each part of the global LSTM contribute?

Every variant is the Phase 1 LSTM with exactly ONE thing changed, run through
the same rolling backtest (4 origins x 28 days x 40 SKUs) with several seeds.
A change "matters" only if its WAPE moves by more than the seed-to-seed spread,
because a smaller difference could be produced by the random seed alone.

Run from the repository root (takes about 1 hour for 6 variants x 3 seeds):
    .venv/bin/python src/ablation.py --seeds 0 1 2
Finished (variant, seed) runs are saved immediately and skipped on a restart,
so the script can be stopped and resumed. Outputs:
    outputs/ablation/predictions.csv   every forecast (backtest format + seed)
    outputs/ablation/runs.csv          metrics of every (variant, seed) run
    outputs/ablation/summary.csv       mean and SD over seeds, change vs reference
    outputs/figures/19_ablation.png    WAPE per variant and per forecast week
"""
import argparse
import logging
import time
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # write the figure to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest
import data
import features
from global_model import global_lstm
from hybrid import kalman_lstm

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "data" / "processed" / "m5_clean_panel.csv"
OUT_DIR = ROOT / "outputs" / "ablation"
FIGURE_PATH = ROOT / "outputs" / "figures" / "19_ablation.png"

REFERENCE = "lstm"
METRICS = ["MAE", "RMSE", "WAPE", "pinball_avg", "coverage_P10_P90"]

# name -> model function. Each entry changes ONE thing compared with "lstm".
VARIANTS = {
    "lstm": global_lstm,                                          # Phase 1 model, the reference
    "no_embedding": partial(global_lstm, use_embedding=False),    # model no longer knows which SKU
    "no_future_calendar": partial(global_lstm, use_future_cov=False),  # no calendar for the 28 days ahead
    "lookback_28": partial(global_lstm, lookback=28),             # 4 weeks of history instead of 8
    "mse_loss": partial(global_lstm, loss="mse"),                 # mean forecast, no intervals
    "hybrid": kalman_lstm,                                        # + Kalman level/trend/uncertainty
}

log = logging.getLogger("ablation")


def load_data():
    """The same cleaned panel, calendar and origins as notebook 02."""
    panel = data.load_clean_panel(PANEL_PATH)
    wide = data.to_wide(panel)
    cal = features.make_calendar_features(panel)
    return wide, cal, backtest.get_origins(len(wide))


def finished_runs(runs_path: Path) -> set:
    """(variant, seed) pairs already saved, so a restart can skip them."""
    if not runs_path.exists():
        return set()
    runs = pd.read_csv(runs_path)
    return set(zip(runs["model"], runs["seed"]))


def append_csv(df: pd.DataFrame, path: Path) -> None:
    """Append rows to a CSV, writing the header only when the file is new."""
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_one(name: str, seed: int, wide, cal, origins) -> tuple:
    """Backtest one variant with one seed. Returns (predictions, metrics row)."""
    start = time.time()
    model = partial(VARIANTS[name], seed=seed)
    preds = backtest.run_backtest(wide, cal, {name: model}, origins)
    preds["seed"] = seed
    metrics = backtest.summarize(preds)
    metrics["seed"] = seed
    metrics["seconds"] = round(time.time() - start, 1)
    return preds, metrics


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over seeds per variant, plus the WAPE change vs the reference.

    beyond_seed_spread is True when |change| is larger than the bigger of the
    two seed SDs (reference and variant): only then is the difference real.
    """
    order = [v for v in VARIANTS if v in set(runs["model"])]
    cols = [m for m in METRICS if m in runs.columns]
    stats = runs.groupby("model")[cols].agg(["mean", "std"]).reindex(order)
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    stats["n_seeds"] = runs.groupby("model")["seed"].nunique().reindex(order)

    if REFERENCE in stats.index:
        ref = stats.loc[REFERENCE]
        ref_sd = ref["WAPE_std"] if pd.notna(ref["WAPE_std"]) else 0.0
        stats["WAPE_change_vs_lstm"] = stats["WAPE_mean"] - ref["WAPE_mean"]
        spread = np.maximum(stats["WAPE_std"].fillna(0.0), ref_sd)
        stats["beyond_seed_spread"] = stats["WAPE_change_vs_lstm"].abs() > spread
    return stats.reset_index().rename(columns={"model": "variant"})


def wape_by_week(preds: pd.DataFrame) -> pd.DataFrame:
    """WAPE per variant and forecast week (steps 1-7 = week 1, ..., 22-28 = week 4)."""
    preds = preds.assign(week=(preds["step"] - 1) // 7 + 1,
                         abs_err=(preds["actual"] - preds["y_pred"]).abs())
    sums = preds.groupby(["model", "week"])[["abs_err", "actual"]].sum()
    return (sums["abs_err"] / sums["actual"]).unstack("week")


def plot_summary(summary: pd.DataFrame, by_week: pd.DataFrame, path: Path) -> None:
    """Left: change in pooled WAPE vs the reference, with the seed SD as error bar
    (bars right of 0 = worse than the reference). Right: WAPE by forecast week."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))
    change = summary.get("WAPE_change_vs_lstm", summary["WAPE_mean"] * 0)
    left.barh(summary["variant"], change, xerr=summary["WAPE_std"].fillna(0),
              color=["tab:red" if c > 0 else "tab:green" for c in change])
    left.axvline(0, color="black", lw=1)
    left.invert_yaxis()
    left.set_xlabel("change in WAPE vs lstm (right = worse), error bar = seed SD")
    left.set_title("Ablation: what each change does to WAPE")

    for variant in summary["variant"]:
        if variant in by_week.index:
            right.plot(by_week.columns, by_week.loc[variant], marker="o", label=variant,
                       lw=2.5 if variant == REFERENCE else 1.2)
    right.set_xticks(by_week.columns)
    right.set_xlabel("forecast week")
    right.set_ylabel("WAPE")
    right.set_title("WAPE by forecast week (all seeds pooled)")
    right.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs_path, preds_path = OUT_DIR / "runs.csv", OUT_DIR / "predictions.csv"
    wide, cal, origins = load_data()
    done = finished_runs(runs_path)

    for name in args.variants:
        for seed in args.seeds:
            if (name, seed) in done:
                log.info("skip %s seed %d (already saved)", name, seed)
                continue
            log.info("run  %s seed %d ...", name, seed)
            preds, metrics = run_one(name, seed, wide, cal, origins)
            append_csv(preds, preds_path)          # save at once: a crash loses one run at most
            append_csv(metrics, runs_path)
            log.info("done %s seed %d: WAPE %.4f in %.0f s", name, seed,
                     metrics["WAPE"].iloc[0], metrics["seconds"].iloc[0])

    summary = summarize_runs(pd.read_csv(runs_path))
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    plot_summary(summary, wape_by_week(pd.read_csv(preds_path)), FIGURE_PATH)
    log.info("summary:\n%s", summary[["variant", "WAPE_mean", "WAPE_std", "WAPE_change_vs_lstm",
                                      "beyond_seed_spread"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
