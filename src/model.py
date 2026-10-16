"""CATransformer: decoder-only causal transformer with learned absolute
positional embeddings, pre-LayerNorm blocks and an optional MLP per layer.

Parameter names match the training checkpoints, so released checkpoints load
with strict key matching.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class CATransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, use_mlp=True,
                 mlp_depth=2):
        """
        mlp_depth: number of Linear layers in the MLP block (default 2 = standard
            `Linear -> act -> Linear`). For mlp_depth > 2, insert
            (mlp_depth - 2) extra Linear(ffn_dim, ffn_dim) layers (each with
            activation) between linear1 and linear2. mlp_depth=2 keeps the
            existing state_dict layout (no `mid_layers`).
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.use_mlp = use_mlp
        self.nhead = nhead

        # Only register MLP-block params when use_mlp=True. Under DDP, unused
        # params trigger a "did not receive grad" reduction assert; skipping the
        # registration is cleaner than passing find_unused_parameters=True.
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

    def forward(self, src, src_mask=None):
        """
        Args:
            src: (B, T, D)
            src_mask: (T, T) bool causal mask (True = masked)
        """
        x = self.norm1(src)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            is_causal=True,
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
                 num_heads_per_layer, use_mlp_per_layer=None,
                 ffn_dim_per_layer=None,
                 emb_dropout=0.1, dropout=0.1,
                 mlp_depth_per_layer=None):
        """
        Args:
            vocab_size: number of token ids (cell states + separators)
            output_size: same as vocab_size
            seq_len: max sequence length (for positional embedding and causal mask)
            num_heads_per_layer: attention heads per layer; its length sets the depth.
            use_mlp_per_layer: per-layer MLP on/off. Defaults to all on.
            ffn_dim_per_layer: per-layer MLP hidden dim. Defaults to 4*hidden_size.
            mlp_depth_per_layer: per-layer number of MLP Linear layers. Defaults to 2.
        """
        super().__init__()
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.final_dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(seq_len, hidden_size)
        self.register_buffer('position_ids', torch.arange(seq_len))

        if use_mlp_per_layer is None:
            use_mlp_per_layer = [True] * len(num_heads_per_layer)
        if ffn_dim_per_layer is None:
            ffn_dim_per_layer = [4 * hidden_size] * len(num_heads_per_layer)
        if mlp_depth_per_layer is None:
            mlp_depth_per_layer = [2] * len(num_heads_per_layer)
        for name, lst in (("use_mlp_per_layer", use_mlp_per_layer),
                          ("ffn_dim_per_layer", ffn_dim_per_layer),
                          ("mlp_depth_per_layer", mlp_depth_per_layer)):
            if len(lst) != len(num_heads_per_layer):
                raise ValueError(f"{name} has {len(lst)} entries but num_heads_per_layer has "
                                 f"{len(num_heads_per_layer)} (one entry per layer)")

        self.transformer_layers = nn.ModuleList([
            CATransformerLayer(
                d_model=hidden_size,
                nhead=h,
                dim_feedforward=fd,
                dropout=dropout,
                use_mlp=m,
                mlp_depth=md,
            ) for h, m, fd, md in zip(num_heads_per_layer, use_mlp_per_layer,
                                       ffn_dim_per_layer, mlp_depth_per_layer)
        ])
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

        # Causal mask: position i can only see positions <= i
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        """
        Args:
            x: (B, T) input token ids
        Returns:
            logits: (B, T, output_size)
        """
        B, T = x.shape
        x = self.embedding(x)
        x = x + self.positional_embedding(self.position_ids[:T]).unsqueeze(0)
        x = self.emb_dropout(x)
        mask = self.causal_mask[:T, :T]

        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask)
        x = self.final_dropout(x)
        return self.fc(x)
