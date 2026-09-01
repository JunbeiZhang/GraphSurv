# model.py  (Unified configurable GraphSurv model)
import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multi-head Attention Block: query attends to key/value."""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor):
        """
        q: (B,1,H)
        kv: (B,S,H)
        key_padding_mask: (B,S), True = ignore
        """
        all_masked = key_padding_mask.all(dim=1)  # (B,)
        if all_masked.any():
            out = q.clone()
            good = ~all_masked
            if good.any():
                attn_out, _ = self.attn(
                    q[good], kv[good], kv[good],
                    key_padding_mask=key_padding_mask[good]
                )
                x = self.ln(q[good] + attn_out)
                x = x + self.ff(x)
                out[good] = x
            return out

        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        x = self.ln(q + attn_out)
        x = x + self.ff(x)
        return x


class GraphSurv(nn.Module):
    """
    Discrete hazard + event node with GraphSurv bipartite message passing,
    with unified configurable switches.

    Switches:
      - use_event_time:  EVENT node also attends over time nodes each layer (Version A component).
      - use_event_gru:   EVENT node is a persistent risk state updated by GRUCell using contexts (Version B component).
      - use_gap:         Add feature-specific gap (log1p) as edge feature.
      - use_has_history: Add has_history indicator as edge feature (recommended when use_gap=True).

    Edge features (base):
      [value, observed_indicator]
    Additional (optional):
      + [log1p(gap)]
      + [has_history]
    """
    def __init__(
        self,
        D: int,
        d_model: int = 128,
        n_layers: int = 2,
        nhead: int = 4,
        dropout: float = 0.0,
        use_event_time: bool = False,
        use_event_gru: bool = False,
        use_gap: bool = False,
        use_has_history: bool = False,
        gap_scale: float = 1.0,
        gap_clip: float | None = None,
    ):
        super().__init__()
        self.D = int(D)
        self.d_model = int(d_model)

        self.use_event_time = bool(use_event_time)
        self.use_event_gru = bool(use_event_gru)
        self.use_gap = bool(use_gap)
        self.use_has_history = bool(use_has_history)

        self.gap_scale = float(gap_scale)
        self.gap_clip = gap_clip if gap_clip is None else float(gap_clip)

        self.channel_emb = nn.Embedding(self.D + 1, self.d_model)  # last index is EVENT

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        # decide edge feature dim
        edge_dim = 2
        if self.use_gap:
            edge_dim += 1
            if self.use_has_history:
                edge_dim += 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.time_update = nn.ModuleList([MAB(self.d_model, nhead, dropout=dropout) for _ in range(n_layers)])
        self.chan_update = nn.ModuleList([MAB(self.d_model, nhead, dropout=dropout) for _ in range(n_layers)])

        if self.use_event_gru:
            self.event_gru = nn.GRUCell(self.d_model, self.d_model)

        self.hazard_head = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )

    @staticmethod
    def _batched_mab(mab: MAB, q: torch.Tensor, kv: torch.Tensor, kpm: torch.Tensor) -> torch.Tensor:
        """
        Vectorized attention call by flattening a leading axis.

        q:  (B, QN, 1, H)
        kv: (B, QN, S, H)
        kpm:(B, QN, S)  True=ignore
        returns: (B, QN, 1, H)
        """
        B, QN, _, H = q.shape
        S = kv.shape[2]
        qf = q.reshape(B * QN, 1, H)
        kvf = kv.reshape(B * QN, S, H)
        kpmf = kpm.reshape(B * QN, S)
        out = mab(qf, kvf, key_padding_mask=kpmf)
        return out.reshape(B, QN, 1, H)

    @staticmethod
    def _feature_gap_and_history(x_time: torch.Tensor, x_mask: torch.Tensor):
        """
        Compute feature-specific gap and has_history.

        x_time: (B,L) float, preferably non-decreasing within patient.
        x_mask: (B,L,D) bool, True=observed

        Returns:
          gap: (B,L,D) float, >=0, 0 if no previous observation
          has_hist: (B,L,D) float(0/1), whether there exists a previous observation
        """
        B, L = x_time.shape
        D = x_mask.shape[2]
        device = x_time.device
        dtype = x_time.dtype

        t = x_time.unsqueeze(-1).expand(B, L, D)  # (B,L,D)
        neg_inf = torch.tensor(-1e30, device=device, dtype=dtype)
        obs_time = torch.where(x_mask, t, neg_inf)  # (B,L,D)

        cummax_time, _ = torch.cummax(obs_time, dim=1)  # max up to i
        last_time = torch.roll(cummax_time, shifts=1, dims=1)
        last_time[:, 0, :] = neg_inf

        has_hist = (last_time > neg_inf / 2).to(dtype=dtype)  # 1 if valid last_time
        gap = t - last_time
        gap = torch.where(has_hist.bool(), gap, torch.zeros_like(gap))
        gap = torch.relu(gap)  # safety if time isn't perfectly sorted

        return gap, has_hist

    def _make_edge_features(self, x_time, x_vals, x_mask):
        """
        Build edge input tensor: (B,L,D,edge_dim)
        """
        # base: value + observed indicator
        obs_ind = x_mask.to(dtype=x_vals.dtype)  # (B,L,D)
        feats = [x_vals, obs_ind]

        if self.use_gap:
            gap, has_hist = self._feature_gap_and_history(x_time, x_mask)

            # scale + log1p
            s = max(self.gap_scale, 1e-8)
            log_gap = torch.log1p(gap / s).to(dtype=x_vals.dtype)

            if self.gap_clip is not None:
                log_gap = torch.clamp(log_gap, 0.0, self.gap_clip)

            feats.append(log_gap)

            if self.use_has_history:
                feats.append(has_hist.to(dtype=x_vals.dtype))

        edge_in = torch.stack(feats, dim=-1)  # (B,L,D,edge_dim)
        return edge_in

    def forward(self, x_time, x_vals, x_mask, tau):
        """
        x_time: (B,L)
        x_vals: (B,L,D)
        x_mask: (B,L,D) bool, True=observed
        tau: (K,)
        Returns:
          hazard_logits: (B,K)
        """
        device = x_time.device
        B, L, D = x_vals.shape
        assert D == self.D, f"Expected D={self.D}, got D={D}"
        K = tau.shape[0]

        # ---- embeddings ----
        chan_ids = torch.arange(D + 1, device=device)
        Hc = self.channel_emb(chan_ids).unsqueeze(0).expand(B, -1, -1)  # (B,D+1,H)
        Ht = self.time_mlp(x_time.unsqueeze(-1))  # (B,L,H)

        # ---- edge embeddings ----
        edge_in = self._make_edge_features(x_time, x_vals, x_mask)  # (B,L,D,edge_dim)
        He = self.edge_mlp(edge_in)  # (B,L,D,H)

        # masks: True=ignore
        pad_time = ~x_mask                         # (B,L,D)
        pad_chan = (~x_mask).permute(0, 2, 1)      # (B,D,L)

        # ---- message passing ----
        for l in range(len(self.time_update)):
            Hc_base = Hc[:, :D, :]  # (B,D,H)

            # time update: time node attends over channels
            kv_time = Hc_base.unsqueeze(1) + He     # (B,L,D,H)
            q_time = Ht.unsqueeze(2)                # (B,L,1,H)
            Ht = self._batched_mab(self.time_update[l], q_time, kv_time, pad_time).squeeze(2)  # (B,L,H)

            # channel update: channel node attends over times
            kv_chan = (Ht.unsqueeze(2) + He).permute(0, 2, 1, 3).contiguous()  # (B,D,L,H)
            q_chan = Hc[:, :D, :].unsqueeze(2)                                  # (B,D,1,H)
            Hc_real = self._batched_mab(self.chan_update[l], q_chan, kv_chan, pad_chan).squeeze(2)  # (B,D,H)

            # EVENT update
            if not self.use_event_gru:
                # base: event <- channels
                event_q = Hc[:, D:D+1, :]  # (B,1,H)
                event_kv_c = Hc_real       # (B,D,H)
                event_kpm_c = torch.zeros((B, D), device=device, dtype=torch.bool)
                Hc_event = self.chan_update[l](event_q, event_kv_c, key_padding_mask=event_kpm_c)  # (B,1,H)

                # optional: event <- times (Version A)
                if self.use_event_time:
                    event_kv_t = Ht
                    event_kpm_t = torch.zeros((B, L), device=device, dtype=torch.bool)
                    Hc_event = self.time_update[l](Hc_event, event_kv_t, key_padding_mask=event_kpm_t)  # (B,1,H)
            else:
                # GRU risk state
                r = Hc[:, D, :]  # (B,H)

                # ctx from channels
                event_q = r.unsqueeze(1)  # (B,1,H)
                event_kv_c = Hc_real
                event_kpm_c = torch.zeros((B, D), device=device, dtype=torch.bool)
                ctx_c = self.chan_update[l](event_q, event_kv_c, key_padding_mask=event_kpm_c).squeeze(1)  # (B,H)
                r = self.event_gru(ctx_c, r)

                if self.use_event_time:
                    # ctx from times
                    event_q2 = r.unsqueeze(1)
                    event_kv_t = Ht
                    event_kpm_t = torch.zeros((B, L), device=device, dtype=torch.bool)
                    ctx_t = self.time_update[l](event_q2, event_kv_t, key_padding_mask=event_kpm_t).squeeze(1)
                    r = self.event_gru(ctx_t, r)

                Hc_event = r.unsqueeze(1)  # (B,1,H)

            Hc = torch.cat([Hc_real, Hc_event], dim=1)  # (B,D+1,H)

        # ---- query tau -> hazard logits ----
        Htau = self.time_mlp(tau.view(1, K, 1).expand(B, -1, -1))  # (B,K,H)
        Hevent = Hc[:, D:D+1, :].expand(B, K, -1)                  # (B,K,H)
        z = torch.cat([Htau, Hevent], dim=-1)                      # (B,K,2H)
        logits = self.hazard_head(z).squeeze(-1)                   # (B,K)
        return logits
