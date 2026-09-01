"""Nested cross-validation workflow for the full GraphSurv model."""

import argparse
import json
import os
import warnings
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # Optional diagnostic plot; never used for model metrics.
    plt = None
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from .data import GraphSurvDataset, make_time_grid, collate_graphsurv_batch
from .model import GraphSurv
from .tuning import ensure_basic_columns, load_or_tune_graphsurv_hyperparameters
from .training import (
    EarlyStopping,
    get_elapsed_time,
    ipcw_iauc,
    save_cv_summary,
    set_seed,
    train_one_epoch,
)
from .training import fit_ipcw_from_loader, evaluate_cindex_ibs_ipcw_with_G

warnings.filterwarnings("ignore")

SEED = 42


# =========================================================
# 1. 通用小工具
# =========================================================
def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(v, device) for v in obj)
    return obj


def forward_model(model, batch_dev):
    """
    兼容当前 GraphSurv.forward(x_time, x_vals, x_mask, tau)
    """
    if isinstance(batch_dev, dict):
        x_time = None
        x_vals = None
        x_mask = None
        tau = None

        for k in ["x_time", "time", "times", "obs_time"]:
            if k in batch_dev:
                x_time = batch_dev[k]
                break

        for k in ["x_vals", "values", "x", "X"]:
            if k in batch_dev:
                x_vals = batch_dev[k]
                break

        for k in ["x_mask", "mask", "M"]:
            if k in batch_dev:
                x_mask = batch_dev[k]
                break

        for k in ["tau", "grid", "time_grid", "t_grid"]:
            if k in batch_dev:
                tau = batch_dev[k]
                break

        if x_time is None or x_vals is None or x_mask is None or tau is None:
            raise KeyError(
                f"Cannot map batch keys to model inputs. Got keys: {list(batch_dev.keys())}"
            )

        return model(x_time=x_time, x_vals=x_vals, x_mask=x_mask, tau=tau)

    if isinstance(batch_dev, (list, tuple)):
        return model(*batch_dev)

    raise TypeError(
        f"Unsupported batch type: {type(batch_dev)}. Expected dict / tuple / list."
    )


def extract_targets_from_batch(batch):
    if isinstance(batch, dict):
        tte = None
        event = None
        ids = None

        for k in ["tte", "T", "time_to_event", "durations"]:
            if k in batch:
                tte = batch[k]
                break
        for k in ["event", "E", "status"]:
            if k in batch:
                event = batch[k]
                break
        for k in ["id", "ids", "subject_id"]:
            if k in batch:
                ids = batch[k]
                break

        if tte is None or event is None:
            raise KeyError(
                "Cannot find tte/event in batch dict. Please edit extract_targets_from_batch()."
            )
        return ids, tte, event

    raise TypeError(
        "Current helper assumes batch is a dict. "
        "If your GraphSurv collate function returns tuple/list, adapt extract_targets_from_batch()."
    )


def extract_survival_and_risk(output):
    if isinstance(output, dict):
        surv = None
        hazard = None
        risk = None
        hazard_logits = None

        for k in ["surv", "survival", "S"]:
            if k in output:
                surv = output[k]
                break

        for k in ["hazard_logits", "logits"]:
            if k in output:
                hazard_logits = output[k]
                break

        for k in ["hazard", "hazards", "h"]:
            if k in output:
                hazard = output[k]
                break

        if hazard is None and hazard_logits is not None:
            hazard = torch.sigmoid(hazard_logits)

        for k in ["risk", "risk_score", "eta", "log_risk"]:
            if k in output:
                risk = output[k]
                break

        if surv is None and hazard is not None:
            hazard = torch.clamp(hazard, min=1e-6, max=1 - 1e-6)
            surv = torch.cumprod(1.0 - hazard, dim=1)

        if risk is None and surv is not None:
            risk = 1.0 - surv[:, -1]

        if surv is None or risk is None:
            raise ValueError(
                "Cannot extract survival/risk from model output dict. "
                "Please edit extract_survival_and_risk()."
            )
        return surv, risk

    if isinstance(output, (list, tuple)):
        if len(output) >= 2:
            a, b = output[0], output[1]
            if torch.is_tensor(a) and a.ndim == 2 and torch.is_tensor(b) and b.ndim == 1:
                return a, b
            if torch.is_tensor(a) and a.ndim == 2 and torch.is_tensor(b) and b.ndim == 2:
                hazard = torch.clamp(a, min=1e-6, max=1 - 1e-6)
                surv = torch.cumprod(1.0 - hazard, dim=1)
                risk = 1.0 - surv[:, -1]
                return surv, risk
        raise ValueError("Cannot parse tuple/list model output. Please edit extract_survival_and_risk().")

    if torch.is_tensor(output):
        if output.ndim == 2:
            hazard = torch.sigmoid(output)
            hazard = torch.clamp(hazard, min=1e-6, max=1 - 1e-6)
            surv = torch.cumprod(1.0 - hazard, dim=1)
            risk = 1.0 - surv[:, -1]
            return surv, risk
        raise ValueError("Unsupported tensor output shape for model output.")

    raise TypeError("Unsupported output type in extract_survival_and_risk().")


def km_on_grid(T, E, t_grid):
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    t_grid = np.asarray(t_grid, dtype=float)

    surv_vals = np.ones_like(t_grid, dtype=float)
    unique_event_times = np.sort(np.unique(T[E == 1]))
    if len(unique_event_times) == 0:
        return surv_vals

    s = 1.0
    j = 0
    for i, t in enumerate(t_grid):
        while j < len(unique_event_times) and unique_event_times[j] <= t:
            tj = unique_event_times[j]
            d_j = np.sum((T == tj) & (E == 1))
            n_j = np.sum(T >= tj)
            if n_j > 0:
                s *= (1.0 - d_j / n_j)
            j += 1
        surv_vals[i] = s
    return surv_vals


def save_km_vs_model_curve(fold_dir, t_grid_np, T_val_np, E_val_np, surv_val_np):
    km_surv = km_on_grid(T_val_np, E_val_np, t_grid_np)
    model_mean_surv = surv_val_np.mean(axis=0)

    df_curve = pd.DataFrame({
        "time": t_grid_np,
        "km_survival": km_surv,
        "model_mean_survival": model_mean_surv,
    })
    df_curve.to_csv(os.path.join(fold_dir, "model_vs_km_survival.csv"), index=False)

    if plt is None:
        return

    plt.figure(figsize=(6.4, 4.8))
    plt.step(df_curve["time"], df_curve["km_survival"], where="post", label="KM")
    plt.plot(df_curve["time"], df_curve["model_mean_survival"], label="Model Mean Survival")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.title("Validation KM vs Model Mean Survival")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "model_vs_km_survival.png"), dpi=200)
    plt.close()


def load_or_make_time_grid(df, out_dir, K_grid):
    override = os.environ.get("REPEATED_CV_TIME_GRID_PATH")
    if override:
        grid = np.load(override)
        if grid.ndim != 1 or len(grid) < 2:
            raise ValueError(f"Invalid REPEATED_CV_TIME_GRID_PATH: {override}")
        return grid
    grid_path = os.path.join(out_dir, "time_grid.npy")
    if os.path.exists(grid_path):
        return np.load(grid_path)

    df_unique = df.drop_duplicates("id")
    grid = make_time_grid(df_unique["tte"].values, K=K_grid, mode="quantile")
    np.save(grid_path, grid)
    return grid


def load_or_make_shared_folds(df, out_dir, n_splits, seed):
    folds_dir = os.path.join(out_dir, "folds_npy")
    os.makedirs(folds_dir, exist_ok=True)
    folds_path = os.path.join(folds_dir, f"folds_{n_splits}_seed{seed}.npy")

    ids = np.array(sorted(df["id"].unique()))
    override = os.environ.get("REPEATED_CV_FOLDS_PATH")
    if override:
        obj = np.load(override, allow_pickle=True).item()
        override_ids = np.asarray(obj["ids"])
        if not np.array_equal(ids.astype(str), override_ids.astype(str)):
            raise ValueError("REPEATED_CV_FOLDS_PATH ids do not match the dataset subject ids.")
        return override_ids, obj["folds"]
    if os.path.exists(folds_path):
        obj = np.load(folds_path, allow_pickle=True).item()
        return obj["ids"], obj["folds"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in kf.split(ids):
        folds.append({"train_idx": tr_idx, "val_idx": va_idx})

    obj = {"ids": ids, "folds": folds}
    np.save(folds_path, obj, allow_pickle=True)
    return ids, folds


def split_outer_train_ids(all_ids, outer_train_idx, seed, fold, val_fraction=0.20):
    """Create an inner validation split using outer-training subjects only."""
    outer_train_ids = np.asarray(all_ids)[np.asarray(outer_train_idx, dtype=int)]
    if len(outer_train_ids) < 5:
        raise ValueError("Too few outer-training subjects for an inner validation split.")
    rng = np.random.RandomState(int(seed) + int(fold))
    perm = rng.permutation(len(outer_train_ids))
    n_val = min(max(1, int(round(len(outer_train_ids) * float(val_fraction)))), len(outer_train_ids) - 1)
    return outer_train_ids[perm[n_val:]], outer_train_ids[perm[:n_val]]


def collect_predictions_from_loader(model, loader, device):
    model.eval()
    all_ids, all_T, all_E, all_surv, all_risk = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            ids, T, E = extract_targets_from_batch(batch)
            batch_dev = move_to_device(batch, device)
            output = forward_model(model, batch_dev)
            surv, risk = extract_survival_and_risk(output)

            if ids is not None:
                if torch.is_tensor(ids):
                    all_ids.append(ids.detach().cpu().numpy())
                else:
                    all_ids.append(np.asarray(ids))

            all_T.append(T.detach().cpu())
            all_E.append(E.detach().cpu())
            all_surv.append(surv.detach().cpu())
            all_risk.append(risk.detach().cpu())

    ids_np = np.concatenate(all_ids, axis=0) if len(all_ids) > 0 else None
    T_t = torch.cat(all_T, dim=0).float()
    E_t = torch.cat(all_E, dim=0).int()
    surv_t = torch.cat(all_surv, dim=0).float()
    risk_t = torch.cat(all_risk, dim=0).float()
    return ids_np, T_t, E_t, surv_t, risk_t


def write_cv_metrics_txt(out_dir, df_folds):
    path = os.path.join(out_dir, "cv_metrics.txt")

    def fmt_mean_sd(x):
        x = pd.Series(x).dropna()
        if len(x) == 0:
            return "nan"
        if len(x) == 1:
            return f"{x.iloc[0]:.6f} ± nan"
        return f"{x.mean():.6f} ± {x.std(ddof=1):.6f}"

    best_row = df_folds.loc[df_folds["cindex"].idxmax()]
    best_fold = int(best_row["fold"])
    best_model_path = os.path.join(out_dir, f"fold{best_fold}", "best_model.pt")

    with open(path, "w", encoding="utf-8") as f:
        f.write("Cross-validation summary\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-fold metrics:\n")
        for _, row in df_folds.iterrows():
            f.write(
                f"Fold {int(row['fold'])}: best_epoch={int(row['best_epoch'])}, "
                f"cindex={row['cindex']:.6f}, ibs={row['ibs']:.6f}, "
                f"iauc={row['iauc']:.6f}, train_time_sec={row['fold_time_sec']:.6f}\n"
            )

        f.write("\n")
        f.write(f"C-index : {fmt_mean_sd(df_folds['cindex'])}\n")
        f.write(f"IBS     : {fmt_mean_sd(df_folds['ibs'])}\n")
        f.write(f"iAUC    : {fmt_mean_sd(df_folds['iauc'])}\n")
        f.write(f"Time(s) : {fmt_mean_sd(df_folds['fold_time_sec'])}\n")
        f.write("\nBest fold by validation C-index\n")
        f.write(
            f"best_fold={best_fold}, best_epoch={int(best_row['best_epoch'])}, "
            f"cindex={best_row['cindex']:.6f}, ibs={best_row['ibs']:.6f}, "
            f"iauc={best_row['iauc']:.6f}\n"
        )
        f.write(f"best_model_path={best_model_path}\n")


def build_graphsurv_from_hyperparameters(feature_dim, best_params, device):
    model = GraphSurv(
        D=feature_dim,
        d_model=int(best_params["d_model"]),
        n_layers=int(best_params["n_layers"]),
        nhead=int(best_params["nhead"]),
        dropout=float(best_params.get("dropout", 0.0)),
        use_event_time=True,
        use_event_gru=False,
        use_gap=True,
        use_has_history=True,
        gap_scale=float(best_params.get("gap_scale", 1.0)),
        gap_clip=float(best_params.get("gap_clip", 4.0)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params["lr"]),
        weight_decay=float(best_params["weight_decay"]),
    )
    return model, optimizer


# =========================================================
# 2. 主流程
# =========================================================
def run_graphsurv_cross_validation(
    data_csv: str | Path,
    output_dir: str | Path,
    dataset_name: str = "dataset",
    n_splits=5,
    seed=42,
    max_epochs=200,
    patience=20,
    K_grid=50,
    batch_size_train=16,
    batch_size_val=64,
    lr=2e-4,
    wd=1e-4,
    tau_quantile_cut=0.95,
    # tuning
    enable_tuning=True,
    tuning_trials=20,
    tuning_inner_splits=3,
    tuning_epochs=40,
    tuning_patience=8,
):
    set_seed(seed)
    cv_start_time = get_elapsed_time()

    # Tune the same full time/gap/history configuration that is used for
    # the selected model in every outer fold.
    os.environ["REPEATED_CV_TUNING_USE_EVENT_TIME"] = "1"
    os.environ["REPEATED_CV_TUNING_USE_GAP"] = "1"
    os.environ["REPEATED_CV_TUNING_USE_HISTORY"] = "1"

    df = pd.read_csv(data_csv)
    df = ensure_basic_columns(df)
    feature_cols = [c for c in df.columns if c not in ["id", "time", "tte", "event"]]

    out_dir = os.fspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid = load_or_make_time_grid(df, out_dir, K_grid)
    grid_t = torch.tensor(grid, dtype=torch.float32)

    all_ids, folds = load_or_make_shared_folds(df, out_dir, n_splits, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_records = []

    for fold, split in enumerate(folds, start=1):
        fold_start_time = get_elapsed_time()
        fold_dir = os.path.join(out_dir, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_idx = split["train_idx"]
        va_idx = split["val_idx"]
        outer_train_ids = set(all_ids[tr_idx])
        test_ids = set(all_ids[va_idx])
        inner_train_ids, inner_val_ids = split_outer_train_ids(all_ids, tr_idx, seed, fold)
        train_ids = set(inner_train_ids)
        val_ids = set(inner_val_ids)

        df_train_outer = df[df["id"].isin(outer_train_ids)].copy()
        df_train = df[df["id"].isin(train_ids)].copy()
        df_val = df[df["id"].isin(val_ids)].copy()
        df_test = df[df["id"].isin(test_ids)].copy()

        # =====================================================
        # A. tuning 只在 outer-train 内部进行，避免数据泄露
        # =====================================================
        best_params = {
            "d_model": 128,
            "n_layers": 2,
            "nhead": 4,
            "dropout": 0.0,
            "lr": lr,
            "weight_decay": wd,
            "batch_size_train": batch_size_train,
            "gap_scale": 1.0,
            "gap_clip": 4.0,
            "source": "manual_default",
            "fixed_switches": {
                "use_event_time": True,
                "use_event_gru": False,
                "use_gap": True,
                "use_has_history": True,
            },
        }

        tuning_json_path = os.path.join(fold_dir, "tuning", "best_params.json")
        if enable_tuning:
            best_params = load_or_tune_graphsurv_hyperparameters(
                df_train_outer=df_train_outer,
                feature_cols=feature_cols,
                grid=grid,
                seed=seed + fold,
                device=device,
                output_json_path=tuning_json_path,
                n_trials=tuning_trials,
                inner_splits=tuning_inner_splits,
                tuning_epochs=tuning_epochs,
                tuning_patience=tuning_patience,
                batch_size_val=batch_size_val,
                tau_quantile_cut=tau_quantile_cut,
                study_name=f"{dataset_name}_fold{fold}_optuna",
            )
            best_params["source"] = "optuna_cached_or_new"
        else:
            os.makedirs(os.path.dirname(tuning_json_path), exist_ok=True)
            with open(tuning_json_path, "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2, ensure_ascii=False)

        tuned_batch_size_train = int(best_params.get("batch_size_train", batch_size_train))

        # =====================================================
        # B. outer-train / outer-val 保持原始功能不变，只是参数改为 tuning 结果
        # =====================================================
        train_ds = GraphSurvDataset(
            df_train,
            id_col="id",
            time_col="time",
            tte_col="tte",
            event_col="event",
            feature_cols=feature_cols,
            grid=grid,
        )
        val_ds = GraphSurvDataset(
            df_val,
            id_col="id",
            time_col="time",
            tte_col="tte",
            event_col="event",
            feature_cols=feature_cols,
            grid=grid,
        )
        test_ds = GraphSurvDataset(
            df_test,
            id_col="id",
            time_col="time",
            tte_col="tte",
            event_col="event",
            feature_cols=feature_cols,
            grid=grid,
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=tuned_batch_size_train,
            shuffle=True,
            collate_fn=lambda b: collate_graphsurv_batch(b, grid_t),
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=batch_size_val,
            shuffle=False,
            collate_fn=lambda b: collate_graphsurv_batch(b, grid_t),
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size_val,
            shuffle=False,
            collate_fn=lambda b: collate_graphsurv_batch(b, grid_t),
        )

        G_times, G_vals, t_cut, _, _ = fit_ipcw_from_loader(
            train_loader,
            tau_quantile_cut=tau_quantile_cut,
        )

        model, opt = build_graphsurv_from_hyperparameters(
            feature_dim=len(feature_cols),
            best_params=best_params,
            device=device,
        )

        best_path = os.path.join(fold_dir, "best_model.pt")
        early_stopping = EarlyStopping(
            patience=patience,
            mode="max",
            delta=1e-4,
            save_path=best_path,
        )

        log_path = os.path.join(fold_dir, "fold_log.txt")
        with open(log_path, "w", encoding="utf-8") as flog:
            flog.write(f"Dataset={dataset_name}, Fold={fold}/{n_splits}\n")
            flog.write(f"Outer train/test n_ids={len(outer_train_ids)}/{len(test_ids)}\n")
            flog.write(f"Inner train/val n_ids={len(train_ids)}/{len(val_ids)}\n")
            flog.write(f"Shared grid K={len(grid)}\n")
            flog.write(f"IPCW tau_quantile_cut={tau_quantile_cut}, t_cut={t_cut:.6f}\n")
            flog.write(f"Tuning enabled={enable_tuning}\n")
            flog.write(f"Best params path={tuning_json_path}\n")
            flog.write(f"Best params={json.dumps(best_params, ensure_ascii=False)}\n\n")

        best_val_cidx = None
        best_val_ibs = None
        best_epoch = None

        for epoch in range(1, max_epochs + 1):
            epoch_start_time = get_elapsed_time()
            train_stats = train_one_epoch(model, train_loader, opt, device)
            train_loss = (
                float(train_stats["loss_total"])
                if isinstance(train_stats, dict)
                else float(train_stats)
            )

            val_cidx, val_ibs = evaluate_cindex_ibs_ipcw_with_G(
                model=model,
                loader_eval=val_loader,
                device=device,
                G_times=G_times,
                G_vals=G_vals,
                t_cut=t_cut,
            )

            epoch_time_sec = get_elapsed_time(epoch_start_time)
            msg = (
                f"[fold {fold} | epoch {epoch:03d}] loss={train_loss:.6f} | "
                f"val_cindex={val_cidx:.4f} | val_ibs={val_ibs:.4f} | "
                f"epoch_time_sec={epoch_time_sec:.3f}"
            )
            print(msg)
            with open(log_path, "a", encoding="utf-8") as flog:
                flog.write(msg + "\n")

            early_stopping(val_cidx, model)
            if best_val_cidx is None or val_cidx > best_val_cidx + 1e-4:
                best_val_cidx = float(val_cidx)
                best_val_ibs = float(val_ibs)
                best_epoch = int(epoch)

            if early_stopping.early_stop:
                stop_msg = f"Early stopping. Best val_cindex={early_stopping.best_score:.4f}"
                print(stop_msg)
                with open(log_path, "a", encoding="utf-8") as flog:
                    flog.write(stop_msg + "\n")
                break

        ckpt = torch.load(best_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        model.eval()

        ids_test_np, T_test_t, E_test_t, surv_test_t, risk_test_t = collect_predictions_from_loader(
            model, test_loader, device
        )

        test_cidx, test_ibs = evaluate_cindex_ibs_ipcw_with_G(
            model=model,
            loader_eval=test_loader,
            device=device,
            G_times=G_times,
            G_vals=G_vals,
            t_cut=t_cut,
        )

        G_times_t = torch.as_tensor(G_times, dtype=torch.float32, device=device)
        G_vals_t = torch.as_tensor(G_vals, dtype=torch.float32, device=device)
        iauc, t_eval_f, auc_t_f = ipcw_iauc(
            T=T_test_t.to(device),
            E=E_test_t.to(device),
            eta=risk_test_t.to(device),
            t_grid=torch.tensor(grid, dtype=torch.float32, device=device),
            reduction="trapz",
            gmin=1e-3,
            gmin_for_window=0.05,
            use_ge=True,
            G_times=G_times_t,
            G_vals=G_vals_t,
        )

        df_auc = pd.DataFrame({
            "time": t_eval_f.detach().cpu().numpy(),
            "td_auc": auc_t_f.detach().cpu().numpy(),
        })
        df_auc.to_csv(os.path.join(fold_dir, "ipcw_tdauc_on_time_grid.csv"), index=False)

        save_km_vs_model_curve(
            fold_dir=fold_dir,
            t_grid_np=np.asarray(grid, dtype=float),
            T_val_np=T_test_t.detach().cpu().numpy(),
            E_val_np=E_test_t.detach().cpu().numpy(),
            surv_val_np=surv_test_t.detach().cpu().numpy(),
        )

        df_pred = pd.DataFrame({
            "id": ids_test_np if ids_test_np is not None else np.arange(len(T_test_t)),
            "tte": T_test_t.detach().cpu().numpy(),
            "event": E_test_t.detach().cpu().numpy(),
            "risk": risk_test_t.detach().cpu().numpy(),
        })
        # Save complete subject-level OOF curves; they are required to compute
        # repeat-level (rather than fold-level) IBS in the revision analysis.
        for grid_index, time_value in enumerate(np.asarray(grid, dtype=float)):
            df_pred[f"S_{time_value:.6f}"] = surv_test_t[:, grid_index].detach().cpu().numpy()
        df_pred.to_csv(os.path.join(fold_dir, "val_risk_predictions.csv"), index=False)
        df_pred.to_csv(os.path.join(fold_dir, "oof_predictions.csv"), index=False)

        fold_time_sec = get_elapsed_time(fold_start_time)
        fold_time_str = get_elapsed_time(fold_start_time, return_str=True)

        fold_records.append({
            "fold": fold,
            "best_epoch": best_epoch,
            "cindex": float(test_cidx),
            "ibs": float(test_ibs),
            "iauc": float(iauc),
            "train_ids": len(outer_train_ids),
            "val_ids": len(test_ids),
            "t_cut": float(t_cut),
            "tau_quantile_cut": float(tau_quantile_cut),
            "fold_time_sec": float(fold_time_sec),
            "fold_time_str": fold_time_str,
            "tuned_d_model": int(best_params["d_model"]),
            "tuned_n_layers": int(best_params["n_layers"]),
            "tuned_nhead": int(best_params["nhead"]),
            "tuned_dropout": float(best_params.get("dropout", 0.0)),
            "tuned_lr": float(best_params["lr"]),
            "tuned_weight_decay": float(best_params["weight_decay"]),
            "tuned_batch_size_train": int(best_params["batch_size_train"]),
            "tuned_gap_scale": float(best_params.get("gap_scale", 1.0)),
            "tuned_gap_clip": float(best_params.get("gap_clip", 4.0)),
        })

        with open(os.path.join(fold_dir, "best_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_epoch: {best_epoch}\n")
            f.write(f"best_val_cindex: {best_val_cidx:.6f}\n")
            f.write(f"best_val_ibs_ipcw: {best_val_ibs:.6f}\n")
            f.write(f"best_val_iauc_ipcw: {iauc:.6f}\n")
            f.write(f"t_cut: {t_cut:.6f}\n")
            f.write(f"tau_quantile_cut: {tau_quantile_cut:.3f}\n")
            f.write(f"fold_time_sec: {fold_time_sec:.6f}\n")
            f.write(f"fold_time_str: {fold_time_str}\n")
            f.write(f"best_params_json: {tuning_json_path}\n")

        fold_msg = (
            f"[fold {fold} done] best_epoch={best_epoch} | best_cindex={best_val_cidx:.4f} | "
            f"best_ibs={best_val_ibs:.4f} | best_iauc={iauc:.4f} | fold_time={fold_time_str}"
        )
        print(fold_msg)
        with open(log_path, "a", encoding="utf-8") as flog:
            flog.write(fold_msg + "\n")

    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(os.path.join(out_dir, "cv_folds_metrics.csv"), index=False)

    save_cv_summary(out_dir, df_folds)
    write_cv_metrics_txt(out_dir, df_folds)

    cv_time_sec = get_elapsed_time(cv_start_time)
    cv_time_str = get_elapsed_time(cv_start_time, return_str=True)
    with open(os.path.join(out_dir, "cv_runtime_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"n_splits: {n_splits}\n")
        f.write(f"total_cv_time_sec: {cv_time_sec:.6f}\n")
        f.write(f"total_cv_time_str: {cv_time_str}\n")
        f.write(f"mean_fold_time_sec: {df_folds['fold_time_sec'].mean():.6f}\n")
        if len(df_folds) > 1:
            f.write(f"std_fold_time_sec: {df_folds['fold_time_sec'].std(ddof=1):.6f}\n")
        else:
            f.write("std_fold_time_sec: nan\n")

    best_row = df_folds.loc[df_folds["cindex"].idxmax()]
    best_fold = int(best_row["fold"])
    best_model_path = os.path.join(out_dir, f"fold{best_fold}", "best_model.pt")
    with open(os.path.join(out_dir, "best_fold_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"best_fold: {best_fold}\n")
        f.write(f"best_epoch: {int(best_row['best_epoch'])}\n")
        f.write(f"best_cindex: {best_row['cindex']:.6f}\n")
        f.write(f"best_ibs: {best_row['ibs']:.6f}\n")
        f.write(f"best_iauc: {best_row['iauc']:.6f}\n")
        f.write(f"best_model_path: {best_model_path}\n")

    return df_folds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dataset")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--time-grid-size", type=int, default=50)
    parser.add_argument("--tuning-trials", type=int, default=20)
    parser.add_argument("--tuning-inner-splits", type=int, default=3)
    parser.add_argument("--tuning-epochs", type=int, default=40)
    parser.add_argument("--tuning-patience", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_graphsurv_cross_validation(
        data_csv=args.data_csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        n_splits=args.n_splits,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        K_grid=args.time_grid_size,
        enable_tuning=True,
        tuning_trials=args.tuning_trials,
        tuning_inner_splits=args.tuning_inner_splits,
        tuning_epochs=args.tuning_epochs,
        tuning_patience=args.tuning_patience,
    )
