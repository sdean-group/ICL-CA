"""Neural network architectures for ECA transformer and baseline models."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EcaLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, use_mlp: bool = True) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.use_mlp = use_mlp

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, src: Tensor, src_mask: Tensor | None = None,
                src_key_padding_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            src: (B, T, D)
            src_mask: (T, T) causal mask
            src_key_padding_mask: (B, T) bool, True = ignore (pad position).
                                  None for fixed-L (no padding).
        """
        x = self.norm1(src)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = src + self.dropout1(attn_out)

        if self.use_mlp:
            y = self.norm2(x)
            y = self.linear2(self.dropout(self.activation(self.linear1(y))))
            x = x + self.dropout2(y)
        return x


class VSimpleTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, output_size: int,
                 seq_len: int, heads_list: list[int],
                 use_mlp_list: list[bool] | None = None,
                 ffn_dim_list: list[int] | None = None,
                 emb_dropout: float = 0.1, dropout: float = 0.1) -> None:
        """
        Args:
            vocab_size: 3 for fixed-L (0, 1, SEP=2), 4 for mixed-L (+ PAD=3)
            output_size: same as vocab_size
            seq_len: max sequence length (for positional embedding and causal mask)
            ffn_dim_list: per-layer MLP hidden dim. Defaults to 4*hidden_size for each layer.
        """
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(seq_len, hidden_size)
        self.register_buffer('position_ids', torch.arange(seq_len))

        if use_mlp_list is None:
            use_mlp_list = [True] * len(heads_list)
        if ffn_dim_list is None:
            ffn_dim_list = [4 * hidden_size] * len(heads_list)

        self.transformer_layers = nn.ModuleList([
            EcaLayer(
                d_model=hidden_size,
                nhead=h,
                dim_feedforward=fd,
                dropout=dropout,
                use_mlp=m,
            ) for h, m, fd in zip(heads_list, use_mlp_list, ffn_dim_list)
        ])
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

        # Causal mask: position i can only see positions <= i
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, T) input token ids
            lengths: (B,) actual sequence lengths before padding.
                     None for fixed-L mode (backward compatible).
        Returns:
            logits: (B, T, output_size)
        """
        B, T = x.shape
        x = self.embedding(x)

        positions = self.position_ids[:T]
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb

        # Causal mask
        mask = self.causal_mask[:T, :T] if T < self.seq_len else self.causal_mask

        # Padding mask: True at positions that should be ignored
        # Only used in mixed-L mode; None preserves original behavior
        pad_mask = None
        if lengths is not None:
            pad_mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]  # (B, T)

        for layer in self.transformer_layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=pad_mask)

        return self.fc(x)


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        num_feats: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        seq_len: int,
        nheads: int = 4,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super(SimpleTransformer, self).__init__()
        self.seq_len = seq_len
        self.emb_dropout = emb_dropout

        self.input_proj = nn.Linear(num_feats, hidden_size)
        self.positional_embedding = nn.Embedding(seq_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nheads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            dim_feedforward=1024,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, -1)
        x = self.input_proj(x)

        positions = torch.arange(self.seq_len, device=x.device)
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb

        if self.training and self.emb_dropout > 0:
            mask = torch.bernoulli(
                torch.full((B, T, 1), 1 - self.emb_dropout, device=x.device)
            )
            x = x * mask / (1 - self.emb_dropout)

        x = self.transformer(x)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.fc(pooled)


class CLSSimpleTransformer(nn.Module):
    def __init__(
        self,
        num_feats: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        seq_len: int,
        nheads: int = 4,
        emb_dropout: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super(CLSSimpleTransformer, self).__init__()
        self.seq_len = seq_len
        self.emb_dropout = emb_dropout

        self.input_proj = nn.Linear(num_feats, hidden_size)
        self.positional_embedding = nn.Embedding(
            seq_len + 1, hidden_size
        )
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nheads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            dim_feedforward=1024,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, -1)
        x = self.input_proj(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_size)

        positions = torch.arange(self.seq_len + 1, device=x.device)
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb

        if self.training and self.emb_dropout > 0:
            mask = torch.bernoulli(
                torch.full((B, T + 1, 1), 1 - self.emb_dropout, device=x.device)
            )
            x = x * mask / (1 - self.emb_dropout)

        x = self.transformer(x)
        cls_output = x[:, 0, :]  # (B, hidden_size)
        return self.fc(cls_output)


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int) -> None:
        super(MLP, self).__init__()
        layers = [
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.Tanh(),
        ]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(
                nn.Tanh(),
            )
        layers.append(nn.Linear(hidden_size, output_size, bias=False))
        self.model = nn.Sequential(*layers)

        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="tanh")

        self.model.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = x.reshape(B, -1)
        return self.model(x)