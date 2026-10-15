"""The registry of released checkpoints (configs/pretrained_models.yaml at the repo root)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

from .types import ModelSpec

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "configs" / "pretrained_models.yaml"


@lru_cache(maxsize=None)
def _raw_registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text())


def _spec_from_entry(key: str, entry: dict) -> ModelSpec:
    task, model, tokens = entry["task"], entry["model"], entry["tokens"]
    return ModelSpec(
        key=key,
        repo_id=entry["repo_id"],
        dimension=int(task["dimension"]),
        stochastic=bool(task["stochastic"]),
        V=int(task["V"]),
        K=int(task["K"]),
        grid_shape=tuple(int(s) for s in task["grid_shape"]),
        T=int(task["T"]),
        M=int(task["M"]),
        neighborhood=task["neighborhood"],
        architecture=model["architecture"],
        d_model=int(model["d_model"]),
        heads=tuple(model["heads"]),
        mlp=tuple(model["mlp"]),
        time_sep=int(tokens["time_sep"]),
        row_sep=None if tokens["row_sep"] is None else int(tokens["row_sep"]),
        rule_pool=task["rule_pool"],
        train_data=task.get("train_data", ""),
    )


@lru_cache(maxsize=None)
def specs() -> Dict[str, ModelSpec]:
    return {k: _spec_from_entry(k, v) for k, v in _raw_registry()["models"].items()}


def get_spec(key: str) -> ModelSpec:
    all_specs = specs()
    if key not in all_specs:
        import difflib
        close = difflib.get_close_matches(key, list(all_specs), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"unknown model '{key}'.{hint} See list_pretrained().")
    return all_specs[key]


def registry_entry(key: str) -> dict:
    get_spec(key)
    return _raw_registry()["models"][key]


class PretrainedTable(list):
    """List of row dicts that prints as a table in terminals and notebooks."""

    COLUMNS = ("key", "task", "grid", "T", "M", "architecture", "d_model")

    def _rows(self) -> List[List[str]]:
        return [[str(r[c]) for c in self.COLUMNS] for r in self]

    def __repr__(self) -> str:
        rows = [list(self.COLUMNS)] + self._rows()
        widths = [max(len(r[i]) for r in rows) for i in range(len(self.COLUMNS))]
        lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows]
        lines.insert(1, "  ".join("-" * w for w in widths))
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        head = "".join(f"<th style='text-align:left'>{c}</th>" for c in self.COLUMNS)
        body = "".join("<tr>" + "".join(f"<td style='text-align:left'>{v}</td>" for v in r)
                       + "</tr>" for r in self._rows())
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def list_pretrained() -> PretrainedTable:
    """All released models, one row each."""
    rows = PretrainedTable()
    for key, s in specs().items():
        rows.append({
            "key": key,
            "task": f"{s.dimension}D {'stoch' if s.stochastic else 'det'} V={s.V} K={s.K}",
            "grid": "x".join(map(str, s.grid_shape)),
            "T": s.T,
            "M": s.M,
            "architecture": s.architecture,
            "d_model": s.d_model,
            "repo_id": s.repo_id,
        })
    return rows


def training_config(key: str) -> dict:
    """training_defaults merged with the model's own `training:` overrides."""
    raw = _raw_registry()
    cfg = dict(raw.get("training_defaults", {}))
    cfg.update(registry_entry(key).get("training") or {})
    return cfg


def task_names() -> List[str]:
    """Task keys, e.g. '1d-det-v2k3' (a model key without its architecture)."""
    return sorted({k.rsplit("-", 1)[0] for k in specs()})


def task_spec(task: str) -> ModelSpec:
    """Spec of any released model on `task` (only its task fields are meaningful)."""
    for key, s in specs().items():
        if key.rsplit("-", 1)[0] == task:
            return s
    raise KeyError(f"unknown task '{task}'; choose from {task_names()}")
