from typing import Tuple

import os
import numpy as np
import pandas as pd
import torch
import time
import torch.nn.functional as F


class EarlyStopping:
    """
    Early stop on a scalar metric.
    call: early_stopping(metric, model)
    """
    def __init__(self, patience=20, mode="max", delta=0.0, save_path="best_model.pt"):
        assert mode in ["max", "min"]
        self.patience = int(patience)
        self.mode = mode
        self.delta = float(delta)
        self.save_path = save_path

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def _is_improved(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.delta
        else:
            return score < self.best_score - self.delta

    def __call__(self, score: float, model: torch.nn.Module):
        score = float(score)
        if self._is_improved(score):
            self.best_score = score
            self.counter = 0
            torch.save({"model": model.state_dict()}, self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


@torch.no_grad()
def hazard_to_survival(hazard_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    hazard_logits: (B,K) -> discrete hazard p_k
    survival S_k = Π_{j<=k} (1 - p_j)
    """
    p = torch.sigmoid(hazard_logits).clamp(eps, 1 - eps)          # (B,K)
    one_minus = 1.0 - p
    S = torch.cumprod(one_minus, dim=1)                           # (B,K)
    return S


def cindex_from_risk(risk: np.ndarray, T: np.ndarray, E: np.ndarray) -> float:
    """
    Simple Harrell-like C-index implementation (no ties fancy handling).
    risk higher => earlier event
    """
    n = len(risk)
    num = 0.0
    den = 0.0
    for i in range(n):
        if E[i] != 1:
            continue
        for j in range(n):
            if T[j] <= T[i]:
                continue
            den += 1.0
            if risk[i] > risk[j]:
                num += 1.0
            elif risk[i] == risk[j]:
                num += 0.5
    return float(num / den) if den > 0 else float("nan")


def brier_ipcw_discrete(
    S: np.ndarray,  # (N,K)
    T: np.ndarray,  # (N,)
    E: np.ndarray,  # (N,)
    tau: np.ndarray,# (K,)
    G_times: np.ndarray,
    G_vals: np.ndarray,
    eps=1e-12,
) -> np.ndarray:
    """
    IPCW Brier score on discrete grid tau.
    Returns bs: (K,)
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    tau = np.asarray(tau, dtype=float)

    N, K = S.shape
    bs = np.zeros(K, dtype=float)

    G_T = step_lookup(G_times, G_vals, T, eps=eps)      # G(T_i)
    G_tau = step_lookup(G_times, G_vals, tau, eps=eps)  # G(tau_k)

    for k, t in enumerate(tau):
        A = (T <= t) & (E == 1)   # event observed by t
        B = (T > t)              # still at risk at t

        termA = np.mean((A.astype(float) / G_T) * (S[:, k] ** 2))
        termB = np.mean((B.astype(float) / G_tau[k]) * ((1.0 - S[:, k]) ** 2))

        bs[k] = termA + termB

    return bs


def ibs_from_brier(bs: np.ndarray, tau: np.ndarray) -> float:
    tau = np.asarray(tau, dtype=float)
    bs = np.asarray(bs, dtype=float)
    dt = np.diff(np.concatenate([[0.0], tau]))
    return float(np.sum(bs * dt) / np.sum(dt))


def km_censoring_survival(T: np.ndarray, E: np.ndarray):
    """
    Kaplan-Meier estimate for censoring survival G(t)=P(C>t).
    Treat censoring as event: delta_c = 1-E.
    Returns step function represented by (times, G_values_after_each_time).
    """
    T = np.asarray(T, dtype=float)
    E = np.asarray(E, dtype=int)
    delta_c = 1 - E

    order = np.argsort(T)
    T_sorted = T[order]
    dc_sorted = delta_c[order]

    uniq_times = np.unique(T_sorted)
    n = len(T_sorted)
    at_risk = n
    G = 1.0

    G_times = []
    G_vals = []

    for t in uniq_times:
        mask_t = (T_sorted == t)
        d_c = int(np.sum(dc_sorted[mask_t]))
        n_t = int(np.sum(mask_t))

        if at_risk > 0 and d_c > 0:
            G *= (1.0 - d_c / at_risk)

        G_times.append(float(t))
        G_vals.append(float(G))

        at_risk -= n_t

    return np.array(G_times, dtype=float), np.array(G_vals, dtype=float)


def step_lookup(times: np.ndarray, values: np.ndarray, t_query, eps=1e-12):
    """
    Right-continuous step fn: value at last time <= t.
    If t < times[0] => 1.0
    """
    t_query = np.asarray(t_query, dtype=float)
    idx = np.searchsorted(times, t_query, side="right") - 1
    out = np.ones_like(t_query, dtype=float)
    valid = idx >= 0
    out[valid] = values[idx[valid]]
    return np.clip(out, eps, 1.0)


@torch.no_grad()
def evaluate_cindex_ibs_ipcw_with_G(
    model,
    loader_eval,
    device,
    G_times: np.ndarray,
    G_vals: np.ndarray,
    t_cut: float = None,
) -> Tuple[float, float]:
    model.eval()

    all_risk, all_T, all_E, all_S = [], [], [], []
    tau_np = None

    for batch in loader_eval:
        x_time = batch["x_time"].to(device)
        x_vals = batch["x_vals"].to(device)
        x_mask = batch["x_mask"].to(device)
        tau = batch["tau"].to(device)

        T = batch["T"].cpu().numpy()
        E = batch["E"].cpu().numpy()

        logits = model(x_time, x_vals, x_mask, tau)      # (B,K)
        S = hazard_to_survival(logits).cpu().numpy()     # (B,K)

        if tau_np is None:
            tau_np = tau.cpu().numpy()

        risk = 1.0 - S[:, -1]

        all_risk.append(risk)
        all_T.append(T)
        all_E.append(E)
        all_S.append(S)

    risk = np.concatenate(all_risk, axis=0)
    T = np.concatenate(all_T, axis=0)
    E = np.concatenate(all_E, axis=0)
    S = np.concatenate(all_S, axis=0)

    cidx = cindex_from_risk(risk, T, E)

    if t_cut is not None:
        use = (tau_np <= float(t_cut))
        if np.sum(use) < 2:
            use = np.ones_like(tau_np, dtype=bool)
    else:
        use = np.ones_like(tau_np, dtype=bool)

    bs = brier_ipcw_discrete(S[:, use], T, E, tau_np[use], G_times, G_vals)
    ibs = ibs_from_brier(bs, tau_np[use])

    return float(cidx), float(ibs)


@torch.no_grad()
def fit_ipcw_from_loader(loader_train, tau_quantile_cut: float = 0.95):
    T_tr_list, E_tr_list = [], []
    for b in loader_train:
        T_tr_list.append(b["T"].cpu().numpy())
        E_tr_list.append(b["E"].cpu().numpy())
    T_tr = np.concatenate(T_tr_list, axis=0)
    E_tr = np.concatenate(E_tr_list, axis=0)

    G_times, G_vals = km_censoring_survival(T_tr, E_tr)
    t_cut = float(np.quantile(T_tr, tau_quantile_cut))

    return G_times, G_vals, t_cut, T_tr, E_tr


def discrete_hazard_nll(
    hazard_logits: torch.Tensor,
    y_event: torch.Tensor,
    at_risk: torch.Tensor,
) -> torch.Tensor:
    """
    Discrete-time hazard negative log-likelihood.

    - Uses BCE-with-logits for numerical stability
    - Applies contribution only on at-risk intervals
    - Normalizes by number of samples (subjects), not by person-time
    """
    y_event = y_event.float()
    at_risk = at_risk.float()

    # elementwise negative log-likelihood on each interval
    nll_mat = F.binary_cross_entropy_with_logits(
        hazard_logits, y_event, reduction="none"
    )

    # keep only at-risk contributions
    nll_mat = nll_mat * at_risk

    # normalize by sample count
    n_samples = hazard_logits.size(0)
    denom = torch.tensor(float(max(n_samples, 1)), device=hazard_logits.device)
    return nll_mat.sum() / denom



def risk_score_from_logits(hazard_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scalar risk score for ranking. Higher => earlier event risk.
    Here we use cumulative failure probability at the end of grid.
    """
    S = hazard_to_survival(hazard_logits, eps=eps)
    return 1.0 - S[:, -1]


def expected_time_from_logits(hazard_logits: torch.Tensor, tau: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Approximate E[T] on discrete grid via survival-area rule:
        E[T] ≈ sum_g S(tau_{g-1}) * Delta_g
    with S(tau_0)=1.

    hazard_logits: (B,K)
    tau: (K,) or (B,K)
    returns: (B,)
    """
    S = hazard_to_survival(hazard_logits, eps=eps)  # (B,K)

    if tau.ndim == 2:
        tau_vec = tau[0]
    else:
        tau_vec = tau

    dt = torch.diff(
        torch.cat([torch.zeros(1, device=tau_vec.device, dtype=tau_vec.dtype), tau_vec], dim=0)
    )  # (K,)

    S_prev = torch.cat([torch.ones_like(S[:, :1]), S[:, :-1]], dim=1)  # (B,K)
    t_exp = (S_prev * dt.unsqueeze(0)).sum(dim=1)
    return t_exp


def pairwise_ranking_loss(
    risk: torch.Tensor,
    T: torch.Tensor,
    E: torch.Tensor,
) -> torch.Tensor:
    """
    Logistic pairwise ranking loss.

    Comparable pair (i, j):
        i had an observed event earlier than j's observed/censor time.
    Desired ordering:
        risk_i > risk_j
    """
    T = T.float().view(-1)
    E = E.float().view(-1)
    risk = risk.view(-1)

    # comparable[i, j] = True if:
    #   i had an observed event
    #   and T_i < T_j
    comparable = (T.unsqueeze(1) < T.unsqueeze(0)) & (E.unsqueeze(1) > 0.5)

    if comparable.sum() == 0:
        return risk.new_zeros(())

    # diff[i, j] = risk_i - risk_j
    diff = risk.unsqueeze(1) - risk.unsqueeze(0)

    # want diff large; penalize when diff is small or negative
    loss_mat = F.softplus(-diff)

    return loss_mat[comparable].mean()


def time_regression_loss(
    hazard_logits: torch.Tensor,
    tau: torch.Tensor,
    T: torch.Tensor,
    E: torch.Tensor,
    loss_type: str = "huber",
    huber_delta: float = 1.0,
    time_scale_mode: str = "max",
    time_scale_eps: float = 1e-8,
) -> torch.Tensor:
    """
    Auxiliary time regression loss on observed-event subjects only.
    Censored subjects are excluded.

    Time is scaled before regression so that the loss magnitude is more stable.
    """
    T = T.float().view(-1)
    E = E.float().view(-1)

    observed_mask = E > 0.5
    if observed_mask.sum() == 0:
        return T.new_zeros(())

    tau = tau.float().view(-1)

    # ---- scale time ----
    if time_scale_mode == "max":
        scale = tau.max().clamp_min(time_scale_eps)
        tau_scaled = tau / scale
        T_scaled = T / scale
    elif time_scale_mode == "none":
        tau_scaled = tau
        T_scaled = T
    else:
        raise ValueError(f"Unsupported time_scale_mode: {time_scale_mode}")

    # predicted expected event time on scaled grid
    t_pred = expected_time_from_logits(hazard_logits, tau_scaled)
    t_true = T_scaled

    if loss_type == "mse":
        return F.mse_loss(t_pred[observed_mask], t_true[observed_mask])
    if loss_type == "l1":
        return F.l1_loss(t_pred[observed_mask], t_true[observed_mask])

    return F.huber_loss(
        t_pred[observed_mask],
        t_true[observed_mask],
        delta=huber_delta,
    )



def compute_total_survival_loss(
    hazard_logits: torch.Tensor,
    y_event: torch.Tensor,
    at_risk: torch.Tensor,
    T: torch.Tensor,
    E: torch.Tensor,
    tau: torch.Tensor,
    lambda_rank: float = 0.2,
    lambda_time: float = 0.1,
    time_loss_type: str = "huber",
    huber_delta: float = 1.0,
    time_scale_mode: str = "max",
):
    """
    Total loss = NLL + lambda_rank * ranking + lambda_time * time_regression
    """
    loss_nll = discrete_hazard_nll(hazard_logits, y_event, at_risk)

    risk = risk_score_from_logits(hazard_logits)
    loss_rank = pairwise_ranking_loss(risk, T, E)

    loss_time = time_regression_loss(
        hazard_logits=hazard_logits,
        tau=tau,
        T=T,
        E=E,
        loss_type=time_loss_type,
        huber_delta=huber_delta,
        time_scale_mode=time_scale_mode,
    )

    total_loss = loss_nll + lambda_rank * loss_rank + lambda_time * loss_time

    parts = {
        "loss_total": float(total_loss.detach().item()),
        "loss_nll": float(loss_nll.detach().item()),
        "loss_rank": float(loss_rank.detach().item()),
        "loss_time": float(loss_time.detach().item()),
    }
    return total_loss, parts


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_rank: float = 0.2,
    lambda_time: float = 0.1,
    time_loss_type: str = "huber",
    huber_delta: float = 1.0,
):
    model.train()
    total_parts = {
        "loss_total": 0.0,
        "loss_nll": 0.0,
        "loss_rank": 0.0,
        "loss_time": 0.0,
    }

    for batch in loader:
        x_time = batch["x_time"].to(device)
        x_vals = batch["x_vals"].to(device)
        x_mask = batch["x_mask"].to(device)
        tau = batch["tau"].to(device)
        y_event = batch["y_event"].to(device)
        at_risk = batch["at_risk"].to(device)
        T = batch["T"].to(device)
        E = batch["E"].to(device)

        logits = model(x_time, x_vals, x_mask, tau)
        loss, parts = compute_total_survival_loss(
            hazard_logits=logits,
            y_event=y_event,
            at_risk=at_risk,
            T=T,
            E=E,
            tau=tau,
            lambda_rank=lambda_rank,
            lambda_time=lambda_time,
            time_loss_type=time_loss_type,
            huber_delta=huber_delta,
            time_scale_mode="max",
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k in total_parts:
            total_parts[k] += parts[k]

    n_batches = max(1, len(loader))
    for k in total_parts:
        total_parts[k] /= n_batches
    return total_parts


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_cv_summary(out_dir: str, df_folds: pd.DataFrame):
    c_mean, c_std = df_folds["cindex"].mean(), df_folds["cindex"].std(ddof=1)
    i_mean, i_std = df_folds["ibs"].mean(), df_folds["ibs"].std(ddof=1)

    txt_path = os.path.join(out_dir, "cv_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("5-Fold Cross-Validation Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(df_folds.to_string(index=False) + "\n\n")
        f.write(f"C-index: mean={c_mean:.4f}, std={c_std:.4f}\n")
        f.write(f"IBS    : mean={i_mean:.4f}, std={i_std:.4f}\n")

    js_path = os.path.join(out_dir, "cv_summary.json")
    summary = {
        "cindex_mean": float(c_mean),
        "cindex_std": float(c_std) if np.isfinite(c_std) else None,
        "ibs_mean": float(i_mean),
        "ibs_std": float(i_std) if np.isfinite(i_std) else None,
    }
    pd.Series(summary).to_json(js_path, indent=2, force_ascii=False)


def get_elapsed_time(start_time=None, return_str=False):
    if start_time is None:
        return time.perf_counter()

    elapsed = time.perf_counter() - start_time

    if return_str:
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    return elapsed


def _km_censoring(T: torch.Tensor, E: torch.Tensor):
    device = T.device
    C = 1.0 - E.float()

    order = torch.argsort(T)
    T_sorted = T[order]
    C_sorted = C[order]

    unique_times = torch.unique(T_sorted)
    K = unique_times.numel()

    if K == 0:
        return torch.empty(0, device=device), torch.empty(0, device=device)

    G = torch.ones(K, device=device, dtype=torch.float32)
    prev_G = torch.tensor(1.0, device=device, dtype=torch.float32)

    for k, t in enumerate(unique_times):
        at_risk = (T_sorted >= t)
        Y_k = at_risk.sum().float()

        d_k = ((T_sorted == t) & (C_sorted == 1)).sum().float()

        if Y_k > 0:
            prev_G = prev_G * (1.0 - d_k / Y_k)

        G[k] = prev_G

    return unique_times.float(), G.float()


def _G_at_times(
    ts: torch.Tensor,
    G_times: torch.Tensor,
    G_vals: torch.Tensor,
):
    device = ts.device
    out = torch.ones_like(ts, dtype=torch.float32, device=device)

    if G_times.numel() == 0:
        return out

    G_times = G_times.to(device)
    G_vals = G_vals.to(device)

    for i, t in enumerate(ts):
        idx = torch.nonzero(G_times <= t, as_tuple=False).squeeze(-1)
        out[i] = G_vals[idx.max()] if idx.numel() > 0 else 1.0

    return out


def ipcw_td_auc(
    T: torch.Tensor,
    E: torch.Tensor,
    eta: torch.Tensor,
    t_grid: torch.Tensor,
    gmin: float = 1e-3,
    gmin_for_window: float = 0.05,
    use_ge: bool = True,
    G_times: torch.Tensor = None,
    G_vals: torch.Tensor = None,
):
    device = T.device
    t_eval = t_grid.to(device).float()
    T = T.to(device).float()
    E = E.to(device).int()
    eta = eta.to(device).float()

    if G_times is None or G_vals is None:
        G_times_local, G_vals_local = _km_censoring(T, E)
    else:
        G_times_local = G_times.to(device).float()
        G_vals_local = G_vals.to(device).float()

    G_t_raw = _G_at_times(t_eval, G_times_local, G_vals_local)
    keep = G_t_raw >= gmin_for_window
    if keep.sum() == 0:
        keep = torch.ones_like(G_t_raw, dtype=torch.bool)

    t_eval = t_eval[keep]
    G_t = G_t_raw[keep].clamp_min(gmin)
    G_T = _G_at_times(T, G_times_local, G_vals_local).clamp_min(gmin)

    T_np = T.detach().cpu().numpy().astype(np.float64)
    E_np = E.detach().cpu().numpy().astype(np.int32)
    eta_np = eta.detach().cpu().numpy().astype(np.float64)

    G_T_np = G_T.detach().cpu().numpy().astype(np.float64)
    G_t_np = G_t.detach().cpu().numpy().astype(np.float64)
    t_eval_np = t_eval.detach().cpu().numpy().astype(np.float64)

    aucs = np.full(len(t_eval_np), np.nan, dtype=np.float64)

    for k, t in enumerate(t_eval_np):
        case_mask = (E_np == 1) & (T_np <= t)
        ctrl_mask = (T_np >= t) if use_ge else (T_np > t)

        n_case = int(case_mask.sum())
        n_ctrl = int(ctrl_mask.sum())
        if n_case == 0 or n_ctrl == 0:
            continue

        r_case = eta_np[case_mask]
        r_ctrl = eta_np[ctrl_mask]

        w_case = 1.0 / G_T_np[case_mask]
        w_ctrl_const = 1.0 / G_t_np[k]

        order = np.argsort(r_ctrl)
        r_ctrl_sorted = r_ctrl[order]
        w_ctrl_sorted = np.full_like(r_ctrl_sorted, w_ctrl_const, dtype=np.float64)

        cumsum_w = np.cumsum(w_ctrl_sorted)
        W_ctrl_total = cumsum_w[-1]

        left = np.searchsorted(r_ctrl_sorted, r_case, side="left")
        right = np.searchsorted(r_ctrl_sorted, r_case, side="right")

        def prefix_sum(idx_arr):
            out = np.zeros_like(idx_arr, dtype=np.float64)
            m = idx_arr > 0
            out[m] = cumsum_w[idx_arr[m] - 1]
            return out

        W_less = prefix_sum(left)
        W_leq = prefix_sum(right)
        W_equal = W_leq - W_less

        numerator = np.sum(w_case * (W_less + 0.5 * W_equal))
        W_case_total = np.sum(w_case)
        denom = W_case_total * W_ctrl_total

        if denom > 0:
            aucs[k] = numerator / denom

    finite = np.isfinite(aucs)
    if finite.sum() == 0:
        return t_eval, torch.empty(0, device=device)

    t_eval_f = torch.as_tensor(
        t_eval_np[finite], device=device, dtype=torch.float32
    )
    auc_t_f = torch.as_tensor(
        aucs[finite], device=device, dtype=torch.float32
    )
    return t_eval_f, auc_t_f


def ipcw_iauc(
    T: torch.Tensor,
    E: torch.Tensor,
    eta: torch.Tensor,
    t_grid: torch.Tensor,
    reduction: str = "trapz",
    gmin: float = 1e-3,
    gmin_for_window: float = 0.05,
    use_ge: bool = True,
    G_times: torch.Tensor = None,
    G_vals: torch.Tensor = None,
):
    t_eval_f, auc_t_f = ipcw_td_auc(
        T=T,
        E=E,
        eta=eta,
        t_grid=t_grid,
        gmin=gmin,
        gmin_for_window=gmin_for_window,
        use_ge=use_ge,
        G_times=G_times,
        G_vals=G_vals,
    )

    if auc_t_f.numel() == 0:
        return float("nan"), t_eval_f, auc_t_f

    if reduction == "none":
        return auc_t_f.mean().item(), t_eval_f, auc_t_f

    K = auc_t_f.numel()
    if K >= 2:
        area = torch.trapz(auc_t_f, t_eval_f)
        span = (t_eval_f[-1] - t_eval_f[0]).clamp_min(1e-6)
        iauc = (area / span).item()
        return iauc, t_eval_f, auc_t_f
    elif K == 1:
        return auc_t_f.item(), t_eval_f, auc_t_f
    else:
        return float("nan"), t_eval_f, auc_t_f
