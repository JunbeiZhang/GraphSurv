# dataset.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas.api.types as ptypes


def make_time_grid(tte: np.ndarray, K: int = 50, mode: str = "quantile") -> np.ndarray:
    tte = np.asarray(tte)
    tte = tte[np.isfinite(tte)]
    tmax = float(np.max(tte))
    if mode == "linear":
        grid = np.linspace(0.0, tmax, K + 1)[1:]
    else:
        qs = np.linspace(0.0, 1.0, K + 1)[1:]
        grid = np.quantile(tte, qs)
        grid = np.unique(grid)
        if len(grid) < K:
            grid = np.linspace(0.0, tmax, K + 1)[1:]
    return grid.astype(np.float32)


def build_discrete_event_targets(tte: float, event: int, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = len(grid)
    tau_prev = np.concatenate([[0.0], grid[:-1]])
    tau_curr = grid

    at_risk = (tte > tau_prev).astype(np.float32)

    y_event = np.zeros((K,), dtype=np.float32)
    if event == 1:
        k = int(np.searchsorted(tau_curr, tte, side="left"))
        if 0 <= k < K:
            y_event[k] = 1.0
    return y_event, at_risk


@dataclass
class PatientSeq:
    id: object
    t: torch.Tensor        # (N,)
    x: torch.Tensor        # (N,D) float with NaN allowed
    tte: float
    event: int


class GraphSurvDataset(torch.utils.data.Dataset):
    """
    Required columns: id, time, tte, event, features...
    """
    def __init__(
        self,
        df: pd.DataFrame,
        id_col="id",
        time_col="time",
        tte_col="tte",
        event_col="event",
        feature_cols: List[str] = None,
        grid: np.ndarray = None
    ):
        self.id_col = id_col
        self.time_col = time_col
        self.tte_col = tte_col
        self.event_col = event_col

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in [id_col, time_col, tte_col, event_col]]
        self.feature_cols = feature_cols

        df = df.copy()

        # encode features (numeric keep; non-numeric -> categorical codes, missing -> NaN)
        for c in self.feature_cols:
            s = df[c]
            if ptypes.is_numeric_dtype(s):
                df[c] = pd.to_numeric(s, errors="coerce")
                df.loc[~np.isfinite(df[c].to_numpy(dtype=float)), c] = np.nan
                continue

            s = s.replace(["", " ", "NA", "N/A", "nan", "NaN", "None", "NULL"], np.nan)
            cat = s.astype("category")
            codes = cat.cat.codes.astype(np.float32)  # missing -> -1
            codes[codes < 0] = np.nan
            df[c] = codes

        self.D = len(self.feature_cols)

        # group by patient
        self.patients: List[PatientSeq] = []
        for pid, g in df.groupby(id_col):
            g = g.sort_values(time_col)
            t = torch.tensor(g[time_col].values, dtype=torch.float32)
            x = torch.tensor(g[self.feature_cols].values, dtype=torch.float32)  # may contain NaN
            tte = float(g[tte_col].iloc[0])
            event = int(g[event_col].iloc[0])
            self.patients.append(PatientSeq(id=pid, t=t, x=x, tte=tte, event=event))

        if grid is None:
            ttes = np.array([p.tte for p in self.patients], dtype=np.float32)
            grid = make_time_grid(ttes, K=50, mode="quantile")
        self.grid = grid.astype(np.float32)
        self.K = len(self.grid)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        y_event, at_risk = build_discrete_event_targets(p.tte, p.event, self.grid)
        return {
            # Keep the original subject identifier through the DataLoader.  This
            # is essential for subject-level OOF auditing; positional row
            # indices are not stable across folds.
            "id": p.id,
            "t": p.t,
            "x": p.x,
            "y_event": torch.tensor(y_event, dtype=torch.float32),
            "at_risk": torch.tensor(at_risk, dtype=torch.float32),
            "tte": torch.tensor(p.tte, dtype=torch.float32),
            "event": torch.tensor(p.event, dtype=torch.long),
        }


def collate_graphsurv_batch(batch: List[Dict], grid: torch.Tensor) -> Dict[str, torch.Tensor]:
    t_list, x_list, m_list = [], [], []
    y_list, r_list = [], []
    T_list, E_list, id_list = [], [], []

    for item in batch:
        t = item["t"]
        x = item["x"]
        m = torch.isfinite(x)
        x0 = torch.nan_to_num(x, nan=0.0)

        t_list.append(t)
        x_list.append(x0)
        m_list.append(m)

        y_list.append(item["y_event"])
        r_list.append(item["at_risk"])

        T_list.append(item["tte"])
        E_list.append(item["event"])
        id_list.append(item["id"])

    x_time = pad_sequence(t_list, batch_first=True, padding_value=0.0)      # (B,L)
    x_vals = pad_sequence(x_list, batch_first=True, padding_value=0.0)      # (B,L,D)
    x_mask = pad_sequence(m_list, batch_first=True, padding_value=False)    # (B,L,D)

    y_event = torch.stack(y_list, dim=0)  # (B,K)
    at_risk = torch.stack(r_list, dim=0)  # (B,K)

    T = torch.stack(T_list, dim=0).float().view(-1)  # (B,)
    E = torch.stack(E_list, dim=0).long().view(-1)   # (B,)

    return {
        "id": id_list,
        "x_time": x_time,
        "x_vals": x_vals,
        "x_mask": x_mask,
        "tau": grid,
        "y_event": y_event,
        "at_risk": at_risk,
        "T": T,
        "E": E,
    }
