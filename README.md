# Local Kalman filters vs a global quantile LSTM for SKU-level demand forecasting

Applied Machine Learning project (CSAI), Newton School of Technology, Rishihood University. Compiled October 25, 2026. Numbers are cited by the repository file they come from; the full write-up is the IEEE-format report, `AML-DL-Report.pdf`. Diagrams use Mermaid, which renders on GitHub.

----

## 1. The short answer

We forecast daily unit sales 28 days ahead for 40 Walmart SKUs from the M5 dataset (store CA_1, category FOODS) and compare six forecasters under one identical, leakage-tested rolling backtest. A single global LSTM trained on all SKUs with quantile loss is the most accurate model and the only one with well-calibrated uncertainty. A per-SKU Kalman filter never wins.

| Model | MAE | RMSE | WAPE | P10-P90 coverage (target 0.80) |
|---|---|---|---|---|
| Global LSTM (quantile) | **1.555** | **3.533** | **0.499** | **0.812** |
| Gradient boosting | 1.763 | 3.672 | 0.566 | n/a |
| Seasonal naive (lag-7) | 1.940 | 4.310 | 0.623 | n/a |
| Seasonal naive (lag-28) | 1.992 | 4.511 | 0.639 | n/a |
| Kalman (+weekly) | 2.108 | 5.025 | 0.676 | 0.950 |
| Kalman (level+trend) | 2.445 | 6.475 | 0.785 | 0.945 |

Source: `outputs/metrics.csv`, pooled over 40 SKUs, 4 forecast origins and 28 days (4,480 SKU-days per model).

The LSTM's lead holds at all four origins, in all four demand classes, in every weekly horizon block, on 31 of 40 SKUs, across three training seeds (WAPE 0.501 ± 0.003), and after removing the one SKU that supplies 42.9% of test volume. The Kalman intervals hit their coverage only by being almost twice as wide (10.6 units against 6.0). Gradient boosting is the strongest classical model and the practical choice when deep learning is not an option. Sources: recomputed from `outputs/predictions.csv`, seeds from notebook 02 section 7.5; details in [docs/RESULTS.md](docs/RESULTS.md).

![Pooled WAPE by model](outputs/figures/11_model_comparison.png)

Source: `outputs/figures/11_model_comparison.png`.

---

## 2. The question

Retail replenishment needs a forecast per item per day, plus a credible range to set safety stock. Two kinds of model answer that differently. Local models such as the Kalman filter fit each series on its own and adapt fast to its level. Global models fit one model across all series and share patterns between them. The report asks three questions:

1. Does a global model forecast daily SKU demand more accurately than per-SKU state-space models and simple baselines?
2. Under which conditions (demand class, horizon, origin) does the local Kalman filter stay competitive?
3. How well calibrated are each family's P10, P50 and P90 forecasts?

The answers, in short: yes; only in the first forecast week, and only against naive baselines; and the LSTM's intervals are close to nominal while the Kalman intervals are far too wide.

---

## 3. How it works

```mermaid
flowchart LR
    A["Raw M5 csv files<br/>(not in git)"] --> B["Notebook 01<br/>audit, clean, EDA"]
    B --> C["data/processed/<br/>m5_clean_panel.csv"]
    C --> D["Notebook 02<br/>features + leakage tests"]
    D --> E["6 forecasters<br/>same interface"]
    E --> F["Rolling backtest<br/>4 origins x 28 days"]
    F --> G["outputs/<br/>predictions, metrics, figures"]
```

Notebook 01 turns the wide raw files into a long panel of 40 SKUs by 1,941 days, flags missing prices and spikes without deleting anything, and runs the EDA that drives the design. Notebook 02 builds lag, rolling and calendar features, proves they cannot see the future, runs every model through the same backtest, and analyses the errors by origin, horizon, demand class and SKU. It ends with an automated acceptance check of 11 criteria, all of which pass in the committed run.

Every model implements one function, `model(wide, cal, end, horizon)`, and may read only data before `end`. That shared contract is what lets one backtest loop and one leakage test cover all six models. See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for the equations.

| Model | Type | How it forecasts | Intervals |
|---|---|---|---|
| Seasonal naive, lag 7 and lag 28 | Baseline | Repeats the last week or last four weeks | none |
| Gradient boosting | Global, classical ML | One `HistGradientBoostingRegressor` on lag, rolling and calendar features, forecasts recursively | none |
| Kalman, level + trend | Local, state-space | One filter per SKU tracking level and slope | Gaussian, from state variance |
| Kalman, + weekly | Local, state-space | Adds six seasonal states (our extension) | Gaussian, from state variance |
| Global LSTM | Global, deep learning | One network for all SKUs, item embedding, outputs all 28 days and 3 quantiles at once, pinball loss | P10 and P90 learned directly |

Source: `src/baselines.py`, `src/kalman.py`, `src/global_model.py`.

---

## 4. Quickstart

The cleaned dataset is committed, so you can reproduce every model and result without downloading M5.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 02_model_comparison.ipynb
```

Notebook 02 takes about 15 minutes on a laptop CPU, most of it training the LSTM at four origins plus two extra seeds for the stability check. Run the notebooks from inside `notebooks/`, because the code uses paths relative to that folder.

To rebuild the clean panel from raw data, download `calendar.csv`, `sales_train_evaluation.csv` and `sell_prices.csv` from the [M5 competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) into `data/raw/` and run `01_data_cleaning_eda.ipynb` first (about one minute). On Google Colab, upload the repository, run `!pip install -r requirements.txt` and open the notebooks from `notebooks/`.

---

## 5. Repository map

```
notebooks/
  01_data_cleaning_eda.ipynb   raw-data audit, cleaning, availability proxy, EDA
  02_model_comparison.ipynb    literature, theory, features, leakage tests, backtest, results
src/
  data.py          SKU selection, cleaning, panel checks, demand classes
  features.py      leakage-safe lag, rolling and calendar features
  baselines.py     seasonal naive and gradient boosting
  kalman.py        Kalman filter class and per-SKU forecasts
  global_model.py  global quantile LSTM (PyTorch)
  backtest.py      rolling origins, cold-start fallback, metric tables
  metrics.py       MAE, RMSE, WAPE, pinball loss, coverage
  plots.py         forecast and interval charts
data/
  raw/             M5 csv files (gitignored)
  processed/       m5_clean_panel.csv, the cleaned 40-SKU panel
outputs/           predictions.csv, metrics.csv, metrics_by_sku.csv, sku_classes.csv, figures/
docs/              data card, methodology, results, viva preparation
```

---

## 6. Decisions worth knowing

**Zeros are kept, never imputed.** 42.9% of SKU-days in the panel are zero. M5 has no stockout label, so a zero cannot be told apart from "no demand". Our availability proxy, `potential_unavailable` (zero sales and no weekly price), marks 679 days, and every one of them falls before the SKU's first sale, so in this subset it flags launches, not stockouts. Source: `data/processed/m5_clean_panel.csv`; see [docs/DATA.md](docs/DATA.md).

**Leakage is tested, not assumed.** Lags use `shift(k)` and rolling means call `shift(1)` before the window. Notebook 02 runs two tests: changing future sales must not change past features, and replacing every post-origin sale with 999 must not change any model's forecast. Both pass.

**The deep model is a quantile LSTM, not DeepAR.** The project specification allowed it as a fallback. It keeps DeepAR's main ideas, a global network, an item embedding, known future calendar inputs and scale normalisation, but predicts P10, P50 and P90 directly with pinball loss, for all 28 days at once (the MQ-RNN design).

**Results are reported by origin, horizon, demand class and SKU.** One SKU, FOODS_3_090, supplies 42.9% of test units, so a pooled score alone could be misleading. Every claim in section 1 was checked at all these levels.

**WAPE replaces MAPE.** MAPE divides by each actual and breaks on zeros. WAPE divides total absolute error by total sales; a forecast of zero everywhere scores exactly 1.0, which makes it easy to read.

---

## 7. Limitations

The study covers one store, one category, 40 SKUs and four consecutive 28-day windows from February to May 2016. No model was tuned, and the Kalman noise comes from a fixed rule rather than maximum likelihood. The LSTM forecasts a median while the other models forecast means, which favours it on MAE and WAPE; it also wins RMSE, but by a smaller margin. Croston-type intermittent-demand baselines were not included, even though 22 of the 40 SKUs are intermittent. The raw data came from a public Hugging Face mirror of the Kaggle files whose shapes match the originals; checksums were not compared. The report's section IX-C and [docs/RESULTS.md](docs/RESULTS.md) section 9 discuss these threats in full.

---

## 8. Team

| Member | Primary responsibility |
|---|---|
| Sayan Bhattacharya | Cleaning pipeline |
| Aryu Rao | Demand classification, data audit |
| Debasish Karan | Feature engineering |
| Saswataduity Bhuin | EDA |
| Aryan Vibhuti | Model training |

Source: `AML-DL-Report.pdf`, Appendix B.

---

## 9. Documentation and further reading

| Document | What it covers |
|---|---|
| [docs/DATA.md](docs/DATA.md) | Data source, SKU selection, cleaning, flags, column schemas of every CSV |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Problem formulation, features, every model's equations, backtest, metrics, alternatives considered |
| [docs/RESULTS.md](docs/RESULTS.md) | All result tables and figures, case study, seed stability, threats to validity |
| [docs/VIVA_QA.md](docs/VIVA_QA.md) | Viva preparation for Phase 1 (ML) and Phase 2 (DL) |
| `AML-DL-Report.pdf` | The IEEE-format project report with the full literature review and 29 references |

The two papers that frame the project are the M5 organisers' [results paper (Makridakis et al., 2022)](https://doi.org/10.1016/j.ijforecast.2021.11.013) and [Montero-Manso and Hyndman (2021)](https://doi.org/10.1016/j.ijforecast.2021.03.004) on local versus global forecasting.

---

## 10. Caveats

The numbers in this README come from a single committed run (LSTM seed 0) and were recomputed from `outputs/predictions.csv` on October 25, 2026. Seed-stability figures come from notebook 02's output. The M5 data is Walmart's, released for the 2020 competition; check the Kaggle competition rules before reusing the derived panel outside coursework.
