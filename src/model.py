"""CATransformer architecture (decoder-only, causal, absolute PE by default).

Copied from the repo-root `models.py` used for training so the installable
package has no dependency on the training scripts. Parameter names are
identical, so training checkpoints load with strict key matching.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class RoPEAttention(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention with Rotary Position Embedding
    applied to Q and K. Same constructor + forward signature so CATransformerLayer
    can swap it in without other changes. Self-attention only (q_in == k_in == v_in).

    Parameter names match nn.MultiheadAttention's layout (`in_proj_weight`,
    `in_proj_bias`, `out_proj.weight/bias`), so absolute-PE checkpoints can be
    surgically remapped if needed — but RoPE runs should be trained from scratch.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True,
                 max_seq_len=2048, rope_base=10000.0):
        super().__init__()
        assert batch_first, "RoPEAttention requires batch_first=True"
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.dropout = dropout

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * embed_dim))
        nn.init.xavier_uniform_(self.in_proj_weight)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Precompute rotary cos/sin tables (T, head_dim/2).
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float()
                                         / self.head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
        self.register_buffer('rope_cos', freqs.cos(), persistent=False)
        self.register_buffer('rope_sin', freqs.sin(), persistent=False)

    def _apply_rope(self, x):
        # x: (B, H, T, head_dim). Pair-rotate every 2 dims.
        T = x.size(2)
        cos = self.rope_cos[:T][None, None, :, :]   # (1, 1, T, D/2)
        sin = self.rope_sin[:T][None, None, :, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.stack((rx1, rx2), dim=-1).flatten(-2)

    def forward(self, q_in, k_in, v_in, attn_mask=None, key_padding_mask=None,
                is_causal=False, need_weights=False, **kwargs):
        B, T, D = q_in.shape
        qkv = F.linear(q_in, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        # Use is_causal=True path (Flash-compatible). attn_mask not supported with RoPE.
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out), None


def build_2d_pos_lookup(L1, L2, T, use_row_sep=True):
    """For a 2D-CA token sequence (cells with optional row/time separators),
    build per-pair lookup tensors used for relative positional bias.

    Returns (sep_mask, dt_idx, dr_idx, dc_idx), each shape (seq_len, seq_len) int/bool.
        sep_mask[i,j] = True if either token i or j is a SEP (ROW_SEP or TIME_SEP).
        dt_idx[i,j]   = (t_i - t_j) clamped to [0, T-1]. Causal mask zeros out (i,j) with t_i < t_j.
        dr_idx[i,j]   = (r_i - r_j) + (L1-1), in [0, 2L1-2]. 0 for SEP-involved pairs (unused).
        dc_idx[i,j]   = (c_i - c_j) + (L2-1), in [0, 2L2-2]. 0 for SEP-involved pairs (unused).
    seq_len = T*tokens_per_step + (T-1) - 1 (matches data_generate_2d.compute_seq_len).
    """
    metas = []  # list of (t, r, c, is_sep)
    for t in range(T):
        for r in range(L1):
            for c in range(L2):
                metas.append((t, r, c, False))
            if use_row_sep and r < L1 - 1:
                metas.append((t, -1, -1, True))   # ROW_SEP
        if t < T - 1:
            metas.append((t, -1, -1, True))       # TIME_SEP
    metas = metas[:-1]   # input = tokens[:-1], so seq_len = total_tokens - 1
    seq_len = len(metas)

    t_arr = torch.tensor([m[0] for m in metas], dtype=torch.long)
    r_arr = torch.tensor([m[1] for m in metas], dtype=torch.long)
    c_arr = torch.tensor([m[2] for m in metas], dtype=torch.long)
    is_sep = torch.tensor([m[3] for m in metas], dtype=torch.bool)

    dt = t_arr.unsqueeze(1) - t_arr.unsqueeze(0)   # (S, S): t_q - t_k
    dr = r_arr.unsqueeze(1) - r_arr.unsqueeze(0)
    dc = c_arr.unsqueeze(1) - c_arr.unsqueeze(0)

    sep_mask = is_sep.unsqueeze(1) | is_sep.unsqueeze(0)
    dt_idx = dt.clamp(0, T - 1)
    dr_idx = (dr + (L1 - 1)).clamp(0, 2 * (L1 - 1))
    dc_idx = (dc + (L2 - 1)).clamp(0, 2 * (L2 - 1))
    # SEP-involved pairs have meaningless dr/dc; force to 0 (unused — masked by sep_mask).
    dr_idx = torch.where(sep_mask, torch.zeros_like(dr_idx), dr_idx)
    dc_idx = torch.where(sep_mask, torch.zeros_like(dc_idx), dc_idx)

    return sep_mask, dt_idx, dr_idx, dc_idx


class CATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, use_mlp=True,
                 pos_bias_T=None, pos_bias_L1=None, pos_bias_L2=None,
                 use_rope=False, rope_max_seq_len=2048,
                 mlp_depth=2):
        """
        mlp_depth: number of Linear layers in the MLP block (default 2 = standard
            `Linear -> act -> Linear`). For mlp_depth > 2, insert
            (mlp_depth - 2) extra Linear(ffn_dim, ffn_dim) layers (each with
            activation) between linear1 and linear2. mlp_depth=2 keeps the
            existing state_dict layout (no `mid_layers`).
        """
        super().__init__()
        self.use_rope = use_rope
        if use_rope:
            assert pos_bias_T is None, "RoPE and pos_bias are mutually exclusive."
            self.self_attn = RoPEAttention(d_model, nhead, dropout=dropout,
                                            batch_first=True,
                                            max_seq_len=rope_max_seq_len)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.use_mlp = use_mlp
        self.nhead = nhead
        # Optional 2D relative positional bias (off by default for backward compat).
        # When all three params are provided, register learnable bias tables.
        self._has_pos_bias = (pos_bias_T is not None and pos_bias_L1 is not None
                              and pos_bias_L2 is not None)
        if self._has_pos_bias:
            self.pos_bias_cell = nn.Parameter(torch.zeros(
                nhead, pos_bias_T, 2 * pos_bias_L1 - 1, 2 * pos_bias_L2 - 1))
            self.pos_bias_sep = nn.Parameter(torch.zeros(nhead, pos_bias_T))

        # Only register MLP-block params when use_mlp=True. Under DDP, unused
        # params trigger a "did not receive grad" reduction assert; skipping the
        # registration is cleaner than passing find_unused_parameters=True.
        # Old ckpts trained pre-fix have unused MLP keys — resume tolerates them
        # via strict=False (see train.py resume path).
        self.mlp_depth = mlp_depth
        assert mlp_depth >= 2, "mlp_depth must be >= 2"
        if use_mlp:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            # Extra hidden-dim Linear layers for deeper MLP (only when mlp_depth > 2).
            if mlp_depth > 2:
                self.mid_layers = nn.ModuleList([
                    nn.Linear(dim_feedforward, dim_feedforward)
                    for _ in range(mlp_depth - 2)
                ])
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, temperature=1.0,
                pos_bias_lookup=None):
        """
        Args:
            src: (B, T, D)
            src_mask: (T, T) causal mask (only used in mixed-L / no-pos-bias case)
            src_key_padding_mask: (B, T) bool, True = ignore (pad position).
                                  None for fixed-L (no padding).
            temperature: scalar > 0. Scales Q by 1/temperature so attention logits
                         are scaled by 1/temperature. temperature=1.0 (default)
                         gives standard scaled-dot-product attention; lower values
                         (e.g., 0.1) yield sharper / more discrete attention.
            pos_bias_lookup: optional 4-tuple of (sep_mask, dt_idx, dr_idx, dc_idx)
                             buffers from build_2d_pos_lookup. Only used when this
                             layer was init'd with pos_bias_T/L1/L2. None disables.
        """
        x = self.norm1(src)
        # Scale Q only; equivalent to softmax(QK^T / (τ * √d_k)).
        # When temperature == 1.0 we keep q = x (identity) for backward-compat.
        q = x if temperature == 1.0 else x / temperature

        if self._has_pos_bias and pos_bias_lookup is not None:
            # New code path: float attn_mask = causal + relative pos bias. No Flash.
            sep_mask, dt_idx, dr_idx, dc_idx = pos_bias_lookup
            T_seq = x.size(1)
            sep_mask = sep_mask[:T_seq, :T_seq]
            dt_idx = dt_idx[:T_seq, :T_seq]
            dr_idx = dr_idx[:T_seq, :T_seq]
            dc_idx = dc_idx[:T_seq, :T_seq]
            # (H, T_seq, T_seq) per-head bias via advanced indexing.
            cell_bias = self.pos_bias_cell[:, dt_idx, dr_idx, dc_idx]
            sep_bias = self.pos_bias_sep[:, dt_idx]
            bias = torch.where(sep_mask.unsqueeze(0), sep_bias, cell_bias)
            if temperature != 1.0:
                bias = bias / temperature   # so logits stay consistent under τ scaling
            # Causal as float mask (-inf above diagonal).
            causal = torch.zeros(T_seq, T_seq, device=x.device, dtype=bias.dtype)
            causal.masked_fill_(
                torch.triu(torch.ones(T_seq, T_seq, dtype=torch.bool, device=x.device), diagonal=1),
                float('-inf'),
            )
            attn_mask = causal.unsqueeze(0) + bias  # (H, T, T)
            B_size = x.size(0)
            # nn.MultiheadAttention wants per-head per-batch mask: (B*H, T, T).
            attn_mask = attn_mask.unsqueeze(0).expand(B_size, -1, -1, -1)\
                                              .reshape(B_size * self.nhead, T_seq, T_seq).contiguous()
            attn_out, _ = self.self_attn(
                q, x, x,
                attn_mask=attn_mask,
                need_weights=False,
            )
        elif src_key_padding_mask is None:
            # Existing fixed-L path with Flash-friendly is_causal hint (UNCHANGED).
            attn_out, _ = self.self_attn(
                q, x, x,
                attn_mask=src_mask,
                is_causal=True,
                need_weights=False,
            )
        else:
            # Existing mixed-L path (UNCHANGED).
            attn_out, _ = self.self_attn(
                q, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )
        x = src + self.dropout1(attn_out)

        if self.use_mlp:
            y = self.norm2(x)
            y = self.activation(self.linear1(y))
            if self.mlp_depth > 2:
                for mid in self.mid_layers:
                    y = self.activation(mid(y))
            y = self.linear2(self.dropout(y))
            x = x + self.dropout2(y)
        return x


class CATransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, seq_len,
                 num_heads_per_layer=None, use_mlp_per_layer=None,
                 ffn_dim_per_layer=None,
                 emb_dropout=0.1, dropout=0.1,
                 pos_bias_T=None, pos_bias_L1=None, pos_bias_L2=None,
                 pos_bias_use_row_sep=True,
                 use_rope=False,
                 mlp_depth_per_layer=None,
                 # Legacy kwarg aliases (deprecated). Callers using old names
                 # still work; do not pass both an old and a new name for the
                 # same argument.
                 heads_list=None, use_mlp_list=None, ffn_dim_list=None,
                 mlp_depth_list=None):
        # Resolve legacy aliases (silently — code migration is in flight).
        def _pick(new_val, old_val, name):
            if new_val is not None and old_val is not None:
                raise TypeError(
                    f"CATransformer: pass only one of the new-name / legacy "
                    f"aliases for '{name}'.")
            return new_val if new_val is not None else old_val
        num_heads_per_layer = _pick(num_heads_per_layer, heads_list, "num_heads_per_layer/heads_list")
        use_mlp_per_layer = _pick(use_mlp_per_layer, use_mlp_list, "use_mlp_per_layer/use_mlp_list")
        ffn_dim_per_layer = _pick(ffn_dim_per_layer, ffn_dim_list, "ffn_dim_per_layer/ffn_dim_list")
        mlp_depth_per_layer = _pick(mlp_depth_per_layer, mlp_depth_list, "mlp_depth_per_layer/mlp_depth_list")
        assert num_heads_per_layer is not None, (
            "CATransformer requires num_heads_per_layer (or legacy heads_list).")
        """
        Args:
            vocab_size: 3 for fixed-L (0, 1, SEP=2), 4 for mixed-L (+ PAD=3)
            output_size: same as vocab_size
            seq_len: max sequence length (for positional embedding and causal mask)
            ffn_dim_list: per-layer MLP hidden dim. Defaults to 4*hidden_size for each layer.
            pos_bias_T/L1/L2: enable 2D relative positional bias. Set all three to
                positive ints to activate. None (default) → no pos_bias (backward compat).
            pos_bias_use_row_sep: whether the data has row separators (used to build
                the (Δt, Δr, Δc) lookup). Default True.
        """
        super().__init__()
        self.use_rope = use_rope
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.final_dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # Absolute PE only when not using RoPE (RoPE injects positions in attn instead).
        if not use_rope:
            self.positional_embedding = nn.Embedding(seq_len, hidden_size)
        self.register_buffer('position_ids', torch.arange(seq_len))

        if use_mlp_per_layer is None:
            use_mlp_per_layer = [True] * len(num_heads_per_layer)
        if ffn_dim_per_layer is None:
            ffn_dim_per_layer = [4 * hidden_size] * len(num_heads_per_layer)
        if mlp_depth_per_layer is None:
            mlp_depth_per_layer = [2] * len(num_heads_per_layer)
        assert len(mlp_depth_per_layer) == len(num_heads_per_layer)

        self._has_pos_bias = (pos_bias_T is not None and pos_bias_L1 is not None
                              and pos_bias_L2 is not None)
        if self._has_pos_bias:
            sep_mask, dt_idx, dr_idx, dc_idx = build_2d_pos_lookup(
                pos_bias_L1, pos_bias_L2, pos_bias_T, use_row_sep=pos_bias_use_row_sep)
            self.register_buffer("pos_bias_sep_mask", sep_mask)
            self.register_buffer("pos_bias_dt_idx", dt_idx)
            self.register_buffer("pos_bias_dr_idx", dr_idx)
            self.register_buffer("pos_bias_dc_idx", dc_idx)

        self.transformer_layers = nn.ModuleList([
            CATransformerLayer(
                d_model=hidden_size,
                nhead=h,
                dim_feedforward=fd,
                dropout=dropout,
                use_mlp=m,
                pos_bias_T=pos_bias_T,
                pos_bias_L1=pos_bias_L1,
                pos_bias_L2=pos_bias_L2,
                use_rope=use_rope,
                rope_max_seq_len=seq_len,
                mlp_depth=md,
            ) for h, m, fd, md in zip(num_heads_per_layer, use_mlp_per_layer,
                                       ffn_dim_per_layer, mlp_depth_per_layer)
        ])
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

        # Causal mask: position i can only see positions <= i
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x, lengths=None, attn_temperature=1.0):
        """
        Args:
            x: (B, T) input token ids
            lengths: (B,) actual sequence lengths before padding.
                     None for fixed-L mode (backward compatible).
            attn_temperature: scalar > 0 to scale attention logits by 1/temperature.
                              Default 1.0 = standard attention (no change).
                              Also accepts a sequence (list/tuple/tensor) of
                              length num_layers for per-layer temperatures —
                              temperature[i] scales layer i's attention softmax
                              only. Scalar and [1.0,...,1.0] both take the
                              per-layer identity shortcut in the layer's
                              forward, so backward-compat is exact.
        Returns:
            logits: (B, T, output_size)
        """
        n_layers = len(self.transformer_layers)
        if isinstance(attn_temperature, (list, tuple)):
            _temps = [float(t) for t in attn_temperature]
        elif hasattr(attn_temperature, "__len__") and not isinstance(attn_temperature, str):
            _temps = [float(t) for t in attn_temperature]
        else:
            _temps = [float(attn_temperature)] * n_layers
        if len(_temps) != n_layers:
            raise ValueError(
                f"attn_temperature list length {len(_temps)} != num layers {n_layers}")
        B, T = x.shape
        x = self.embedding(x)

        if not self.use_rope:
            positions = self.position_ids[:T]
            pos_emb = self.positional_embedding(positions).unsqueeze(0)
            x = x + pos_emb
        x = self.emb_dropout(x)
        # Causal mask
        mask = self.causal_mask[:T, :T] if T < self.seq_len else self.causal_mask

        # Padding mask: True at positions that should be ignored
        # Only used in mixed-L mode; None preserves original behavior
        pad_mask = None
        if lengths is not None:
            pad_mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]  # (B, T)

        # Build pos_bias_lookup once (shared across layers) when enabled.
        pos_bias_lookup = None
        if self._has_pos_bias:
            pos_bias_lookup = (self.pos_bias_sep_mask, self.pos_bias_dt_idx,
                               self.pos_bias_dr_idx, self.pos_bias_dc_idx)

        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, src_mask=mask, src_key_padding_mask=pad_mask,
                      temperature=_temps[i],
                      pos_bias_lookup=pos_bias_lookup)
        x = self.final_dropout(x)
        return self.fc(x)
