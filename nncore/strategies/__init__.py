from .batch_avarage import BatchAvgStrategy
from .final_avarage import FinalAvgStrategy
from .full_batch import FullBatchStrategy
from .gradient_avarage import GradAvarageStrategy
from .mini_batch import MiniBatchStrategy
from .parallel_weight_avarage import ParallelWeightAvgStrategy
from .training_strategy import TrainingStrategy
from .weight_avarage import WeightAvgStrategy

__all__ = [
    "BatchAvgStrategy",
    "FinalAvgStrategy",
    "FullBatchStrategy",
    "GradAvarageStrategy",
    "MiniBatchStrategy",
    "ParallelWeightAvgStrategy",
    "TrainingStrategy",
    "WeightAvgStrategy",
]
