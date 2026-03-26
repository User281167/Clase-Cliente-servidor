from typing import Literal

from .binary_cross_entropy import BinaryCrossEntropy
from .cost_function import CostFunction
from .cross_entropy import CrossEntropy
from .mean_squared_error import MeanSquaredError

# ──────────────────────────────────────────────
# Registro
# ──────────────────────────────────────────────
COSTS = {
    "cross_entropy": CrossEntropy,
    "mse": MeanSquaredError,
    "binary_cross_entropy": BinaryCrossEntropy,
}

CostName = Literal["cross_entropy", "mse", "binary_cross_entropy"]


def get_cost(name: CostName) -> CostFunction:
    if name not in COSTS:
        raise ValueError(f"Unknown cost '{name}'. Available: {list(COSTS.keys())}")
    return COSTS[name]()


__all__ = [
    "get_cost",
    "CostFunction",
    "CrossEntropy",
    "MeanSquaredError",
    "BinaryCrossEntropy",
]
