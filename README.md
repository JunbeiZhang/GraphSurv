# GraphSurv

GraphSurv is a graph-based discrete-time survival model for irregular multivariate longitudinal observations.

## Installation

The validated environment is Python 3.11.16 with PyTorch 2.12.0+cu126 (CUDA 12.6). GPU acceleration is used when available; the same workflow also runs on CPU.

```bash
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements.txt
```

The pinned runtime dependencies are PyTorch 2.12.0+cu126, NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, Optuna 4.6.0, and SciPy 1.17.1.

## Input format

GraphSurv reads a CSV with one row per subject observation. Required columns are:

| Column | Meaning |
| --- | --- |
| `id` | Subject identifier; all rows for a subject share this value. |
| `time` | Observation time. Visits may be irregular and are sorted within subject by the loader. |
| `tte` | Subject-level event or censoring time, repeated on each row for that subject. |
| `event` | Subject-level event indicator (`1` = event; `0` = censored), repeated on each row for that subject. |

All remaining columns are longitudinal feature channels. Numeric missing values are supported and non-numeric channels are encoded by the loader. Static covariates, when used, should be included as feature columns and repeated for each observation of the subject. The loader also accepts `times` in place of `time` and `label` in place of `event`.

## Minimal reproducible example

From the repository root, run:

```bash
python examples/minimal_reproducible_example.py --output-dir examples/minimal_reproducible_output
```

This single example generates a small synthetic longitudinal survival dataset and then calls the formal GraphSurv workflow in `graphsurv/workflow.py`. It performs hyperparameter tuning, model fitting, held-out prediction, and IPCW evaluation (C-index, integrated Brier score, and iAUC). It does not read or write clinical data.

The demonstration uses three outer folds, two Optuna trials, two tuning epochs, and four final-fit epochs solely to make the workflow quick to reproduce; it is not the full paper tuning budget.

The output directory contains the synthetic CSV, `cv_folds_metrics.csv`, `cv_metrics.txt`, per-fold `tuning/best_params.json`, fitted `best_model.pt` files, subject-level `oof_predictions.csv` files, and per-fold IPCW time-dependent AUC files.

## Cite This Paper
```Latex
@ARTICLE{zhang2026graphsurv,
  author={Zhang, Junbei and Zhao, Xuejing},
  journal={IEEE Transactions on Computational Biology and Bioinformatics}, 
  title={GraphSurv: A Graph-Based Framework for Dynamic Survival Analysis With Longitudinal Data}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TCBBIO.2026.3741569}}
```
