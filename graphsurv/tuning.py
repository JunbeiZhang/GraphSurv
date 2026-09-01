import copy
import json
import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import KFold

from .data import GraphSurvDataset, collate_graphsurv_batch
from .model import GraphSurv
from .training import EarlyStopping, train_one_epoch, set_seed
from .training import fit_ipcw_from_loader, evaluate_cindex_ibs_ipcw_with_G

warnings.filterwarnings("ignore")


# =========================================================
# 通用辅助函数
# =========================================================
def ensure_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "times" in df.columns and "time" not in df.columns:
        rename_map["times"] = "time"
    if "label" in df.columns and "event" not in df.columns:
        rename_map["label"] = "event"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["id", "time", "tte", "event"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    # Match the project-wide one-vs-rest definition on PBC2: only label 1 is
    # the event of interest; the competing-risk label 2 is treated as censored.
    raw_event = df["event"].astype(int)
    df["event"] = (raw_event == 1).astype(int) if not set(raw_event.unique()).issubset({0, 1}) else raw_event
    df["time"] = df["time"].astype(float)
    df["tte"] = df["tte"].astype(float)
    return df


def make_graphsurv_loader(
    df_part: pd.DataFrame,
    feature_cols: List[str],
    grid: np.ndarray,
    grid_t: torch.Tensor,
    batch_size: int,
    shuffle: bool,
):
    ds = GraphSurvDataset(
        df_part,
        id_col="id",
        time_col="time",
        tte_col="tte",
        event_col="event",
        feature_cols=feature_cols,
        grid=grid,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda b: collate_graphsurv_batch(b, grid_t),
    )
    return ds, loader


def build_graphsurv_model(feature_dim: int, params: Dict) -> torch.nn.Module:
    use_event_time = os.environ.get("REPEATED_CV_TUNING_USE_EVENT_TIME", "0") == "1"
    use_gap = os.environ.get("REPEATED_CV_TUNING_USE_GAP", "0") == "1"
    use_has_history = os.environ.get("REPEATED_CV_TUNING_USE_HISTORY", "0") == "1"
    return GraphSurv(
        D=feature_dim,
        d_model=int(params["d_model"]),
        n_layers=int(params["n_layers"]),
        nhead=int(params["nhead"]),
        use_event_time=use_event_time,
        use_event_gru=False,
        use_gap=use_gap,
        use_has_history=use_has_history,
        gap_scale=1.0,
        gap_clip=4.0,
    )


def sanitize_metric(x, default_bad_value: float) -> float:
    if x is None:
        return default_bad_value
    x = float(x)
    if np.isnan(x) or np.isinf(x):
        return default_bad_value
    return x


# =========================================================
# JSON 保存 / 读取
# =========================================================
def save_best_params_json(params: Dict, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def load_best_params_json(json_path: str):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 内层 tuning 核心
# =========================================================
def _sample_hparams(trial: optuna.trial.Trial) -> Dict:
    d_model = trial.suggest_categorical("d_model", [64, 96, 128, 160, 192])
    valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]

    params = {
        "d_model": d_model,
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "nhead": trial.suggest_categorical("nhead", valid_heads),
        "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size_train": trial.suggest_categorical("batch_size_train", [8, 16, 32]),
    }
    return params


def tune_graphsurv_hyperparameters(
    df_train_outer: pd.DataFrame,
    feature_cols: List[str],
    grid: np.ndarray,
    seed: int,
    device: str,
    output_json_path: str,
    n_trials: int = 20,
    inner_splits: int = 3,
    tuning_epochs: int = 40,
    tuning_patience: int = 8,
    batch_size_val: int = 64,
    tau_quantile_cut: float = 0.95,
    study_name: str = None,
) -> Dict:
    """
    仅使用 outer-train 做 inner-CV tuning，避免 outer validation 泄露。
    若 json 已存在则直接读取。
    """
    cached = load_best_params_json(output_json_path)
    if cached is not None:
        return cached

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    grid_t = torch.tensor(grid, dtype=torch.float32)

    df_train_outer = ensure_basic_columns(df_train_outer)
    outer_ids = np.array(sorted(df_train_outer["id"].unique()))
    if len(outer_ids) < inner_splits:
        raise ValueError(
            f"Not enough subjects for inner_splits={inner_splits}. "
            f"Current unique ids={len(outer_ids)}"
        )

    feature_dim = len(feature_cols)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        set_seed(seed + trial.number)
        params = _sample_hparams(trial)

        inner_kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        inner_scores = []
        inner_ibs_list = []

        for inner_fold, (tr_idx, va_idx) in enumerate(inner_kf.split(outer_ids), start=1):
            inner_train_ids = set(outer_ids[tr_idx])
            inner_val_ids = set(outer_ids[va_idx])

            df_inner_train = df_train_outer[df_train_outer["id"].isin(inner_train_ids)].copy()
            df_inner_val = df_train_outer[df_train_outer["id"].isin(inner_val_ids)].copy()

            _, inner_train_loader = make_graphsurv_loader(
                df_part=df_inner_train,
                feature_cols=feature_cols,
                grid=grid,
                grid_t=grid_t,
                batch_size=int(params["batch_size_train"]),
                shuffle=True,
            )
            _, inner_val_loader = make_graphsurv_loader(
                df_part=df_inner_val,
                feature_cols=feature_cols,
                grid=grid,
                grid_t=grid_t,
                batch_size=batch_size_val,
                shuffle=False,
            )

            G_times, G_vals, t_cut, _, _ = fit_ipcw_from_loader(
                inner_train_loader,
                tau_quantile_cut=tau_quantile_cut,
            )

            model = build_graphsurv_model(feature_dim=feature_dim, params=params).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(params["lr"]),
                weight_decay=float(params["weight_decay"]),
            )

            tmp_ckpt = os.path.join(
                os.path.dirname(output_json_path),
                f"_tmp_trial{trial.number}_inner{inner_fold}.pt",
            )
            early_stopping = EarlyStopping(
                patience=tuning_patience,
                mode="max",
                delta=1e-4,
                save_path=tmp_ckpt,
            )

            best_inner_cidx = None
            best_inner_ibs = None

            for _ in range(1, tuning_epochs + 1):
                train_one_epoch(model, inner_train_loader, optimizer, device)
                val_cidx, val_ibs = evaluate_cindex_ibs_ipcw_with_G(
                    model=model,
                    loader_eval=inner_val_loader,
                    device=device,
                    G_times=G_times,
                    G_vals=G_vals,
                    t_cut=t_cut,
                )

                val_cidx = sanitize_metric(val_cidx, 0.0)
                val_ibs = sanitize_metric(val_ibs, 1e6)

                if best_inner_cidx is None or val_cidx > best_inner_cidx + 1e-4:
                    best_inner_cidx = val_cidx
                    best_inner_ibs = val_ibs

                early_stopping(val_cidx, model)
                if early_stopping.early_stop:
                    break

            if os.path.exists(tmp_ckpt):
                try:
                    os.remove(tmp_ckpt)
                except OSError:
                    pass

            best_inner_cidx = sanitize_metric(best_inner_cidx, 0.0)
            best_inner_ibs = sanitize_metric(best_inner_ibs, 1e6)
            inner_scores.append(best_inner_cidx)
            inner_ibs_list.append(best_inner_ibs)

            trial.report(float(np.mean(inner_scores)), step=inner_fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_cidx = float(np.mean(inner_scores)) if inner_scores else 0.0
        mean_ibs = float(np.mean(inner_ibs_list)) if inner_ibs_list else 1e6
        trial.set_user_attr("mean_inner_cindex", mean_cidx)
        trial.set_user_attr("mean_inner_ibs", mean_ibs)
        return mean_cidx

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = copy.deepcopy(best_trial.params)
    best_params["best_value"] = float(best_trial.value)
    best_params["mean_inner_cindex"] = float(best_trial.user_attrs.get("mean_inner_cindex", best_trial.value))
    best_params["mean_inner_ibs"] = float(best_trial.user_attrs.get("mean_inner_ibs", np.nan))
    best_params["n_trials"] = int(n_trials)
    best_params["inner_splits"] = int(inner_splits)
    best_params["tuning_epochs"] = int(tuning_epochs)
    best_params["tuning_patience"] = int(tuning_patience)
    best_params["selection_metric"] = "maximize_mean_inner_cindex"

    save_best_params_json(best_params, output_json_path)
    return best_params


# =========================================================
# 对外主入口
# =========================================================
def load_or_tune_graphsurv_hyperparameters(
    df_train_outer: pd.DataFrame,
    feature_cols: List[str],
    grid: np.ndarray,
    seed: int,
    device: str,
    output_json_path: str,
    n_trials: int = 20,
    inner_splits: int = 3,
    tuning_epochs: int = 40,
    tuning_patience: int = 8,
    batch_size_val: int = 64,
    tau_quantile_cut: float = 0.95,
    study_name: str = None,
) -> Dict:
    return tune_graphsurv_hyperparameters(
        df_train_outer=df_train_outer,
        feature_cols=feature_cols,
        grid=grid,
        seed=seed,
        device=device,
        output_json_path=output_json_path,
        n_trials=n_trials,
        inner_splits=inner_splits,
        tuning_epochs=tuning_epochs,
        tuning_patience=tuning_patience,
        batch_size_val=batch_size_val,
        tau_quantile_cut=tau_quantile_cut,
        study_name=study_name,
    )
