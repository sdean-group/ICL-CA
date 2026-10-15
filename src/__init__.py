"""CA-Transformer: pretrained in-context cellular-automaton models.

Quick start:
    from src import list_pretrained, load_pretrained, generate_example, predict
    bundle = load_pretrained("1d-det-v2k3-shared")
    example = generate_example(bundle.spec, seed=0)
    result = predict(bundle, example)
    result.metrics
"""

from .data import check_context, continue_trajectory, generate_example, generate_examples
from .inference import evaluate, predict, rollout
from .pretrained import ModelBundle, load_pretrained
from .registry import get_spec, list_pretrained
from .types import Example, ModelSpec, PredictionResult, RolloutResult
from .visualization import plot_ar_accuracy, plot_example, plot_prediction, plot_rollout

__all__ = [
    "list_pretrained", "load_pretrained", "get_spec",
    "generate_example", "generate_examples", "continue_trajectory", "check_context",
    "predict", "rollout", "evaluate",
    "plot_example", "plot_prediction", "plot_rollout", "plot_ar_accuracy",
    "ModelBundle", "ModelSpec", "Example", "PredictionResult", "RolloutResult",
]
__version__ = "0.1.0"
