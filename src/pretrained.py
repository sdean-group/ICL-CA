"""Download a released checkpoint, build the right architecture, load weights."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import torch

from .model import CATransformer
from .registry import get_spec, registry_entry, _raw_registry
from .types import ModelSpec

# Buffers rebuilt in __init__; their absence from a checkpoint is harmless.
_REBUILT_BUFFERS = ("position_ids", "causal_mask")


@dataclass
class ModelBundle:
    spec: ModelSpec
    model: CATransformer
    device: torch.device
    checkpoint_path: str
    checkpoint_config: dict

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return (f"ModelBundle({self.spec.describe()}; {n / 1e6:.2f}M params; "
                f"device={self.device})")


def resolve_device(device: Union[str, torch.device] = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_checkpoint(key: str, cache_dir: Optional[str] = None,
                        revision: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    entry = registry_entry(key)
    filename = entry.get("checkpoint_file",
                         _raw_registry()["defaults"]["checkpoint_file"])
    return hf_hub_download(entry["repo_id"], filename, cache_dir=cache_dir,
                           revision=revision)


def _normalize_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k = k.replace("_orig_mod.", "", 1).replace("transformer.layers.", "transformer_layers.")
        out[k] = v
    return out


def _check_against_registry(spec: ModelSpec, cfg: dict, sd: dict) -> None:
    """Fail loudly if the registry disagrees with the checkpoint."""
    problems = []
    if cfg["hidden_size"] != spec.d_model:
        problems.append(f"d_model {cfg['hidden_size']} != registry {spec.d_model}")
    if list(cfg["heads_list"]) != list(spec.heads):
        problems.append(f"heads {cfg['heads_list']} != registry {list(spec.heads)}")
    mlp = cfg.get("use_mlp_list") or [1] * len(cfg["heads_list"])
    if [int(m) for m in mlp] != list(spec.mlp):
        problems.append(f"mlp {mlp} != registry {list(spec.mlp)}")
    vocab = sd["embedding.weight"].shape[0]
    if vocab != spec.vocab_size:
        problems.append(f"vocab {vocab} != registry-implied {spec.vocab_size}")
    n_pos = sd["positional_embedding.weight"].shape[0] if "positional_embedding.weight" in sd else None
    if n_pos is not None and n_pos != spec.seq_len:
        problems.append(f"positional table {n_pos} != registry-implied seq_len {spec.seq_len} "
                        f"(check grid_shape / T)")
    if problems:
        raise RuntimeError(f"registry/checkpoint mismatch for {spec.key}: " + "; ".join(problems))


def _build_model(spec: ModelSpec, cfg: dict) -> CATransformer:
    if cfg.get("use_rope") or cfg.get("pos_bias_T") is not None:
        raise NotImplementedError("checkpoint uses RoPE / 2D positional bias, which this "
                                  "release does not support (absolute positions only)")
    return CATransformer(
        vocab_size=spec.vocab_size,
        hidden_size=cfg["hidden_size"],
        output_size=spec.vocab_size,
        seq_len=spec.seq_len,
        num_heads_per_layer=cfg["heads_list"],
        use_mlp_per_layer=cfg.get("use_mlp_list"),
        ffn_dim_per_layer=cfg.get("ffn_dim_list"),
        emb_dropout=0.0,
        dropout=0.0,
        mlp_depth_per_layer=cfg.get("mlp_depth_list"),
    )


def _load_state(model: CATransformer, sd: dict, name: str) -> None:
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith(_REBUILT_BUFFERS)]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint key mismatch for {name}: "
                           f"missing={missing} unexpected={unexpected}")


def load_pretrained(key: str, device: Union[str, torch.device] = "auto",
                    cache_dir: Optional[str] = None, revision: Optional[str] = None,
                    checkpoint_path: Optional[str] = None) -> ModelBundle:
    """Load one of the released models by registry key (see `list_pretrained()`).

    Downloads only that model's checkpoint from the Hugging Face Hub (cached
    after the first call). Pass `checkpoint_path` to load a local file instead.
    """
    spec = get_spec(key)
    dev = resolve_device(device)
    path = checkpoint_path or download_checkpoint(key, cache_dir, revision)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    sd = _normalize_state_dict(ckpt["model_state_dict"])
    _check_against_registry(spec, cfg, sd)

    model = _build_model(spec, cfg)
    _load_state(model, sd, key)
    model.to(dev).eval()
    return ModelBundle(spec, model, dev, path, cfg)


def load_checkpoint(path: str, device: Union[str, torch.device] = "auto") -> ModelBundle:
    """Load any checkpoint written by src.train (it stores its task spec), or a
    released one whose registry key you pass through `load_pretrained`."""
    import dataclasses
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "task" not in ckpt:
        raise ValueError(f"{path} has no task description (written by the original train.py?); "
                         f"use load_pretrained(<registry key>, checkpoint_path=...) instead")
    fields = dict(ckpt["task"])
    fields["grid_shape"] = tuple(fields["grid_shape"])
    fields["heads"], fields["mlp"] = tuple(fields["heads"]), tuple(fields["mlp"])
    spec = ModelSpec(**{f.name: fields[f.name] for f in dataclasses.fields(ModelSpec)})
    cfg = dict(ckpt["model_config"])
    sd = _normalize_state_dict(ckpt["model_state_dict"])
    dev = resolve_device(device)
    model = _build_model(spec, cfg)
    _load_state(model, sd, path)
    model.to(dev).eval()
    return ModelBundle(spec, model, dev, path, cfg)
