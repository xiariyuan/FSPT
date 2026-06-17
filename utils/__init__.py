"""
FSPT Utils Package
"""

from .tensor_cache import DiskTensorCache

__all__ = [
    "DiskTensorCache",
]


def __getattr__(name):
    """
    Lazy-load heavier training utilities on demand.

    Importing `utils` should stay lightweight because low-level modules such as
    `models.semantic_encoder` only need `tensor_cache`. Eagerly importing the
    full training stack here creates expensive import-time side effects and can
    stall model construction.
    """
    if name == "DataPrefetcher":
        from .data_prefetcher import DataPrefetcher

        return DataPrefetcher
    if name == "CUDAGraphWrapper":
        from .data_prefetcher import CUDAGraphWrapper

        return CUDAGraphWrapper
    if name == "create_prefetched_dataloader":
        from .data_prefetcher import create_prefetched_dataloader

        return create_prefetched_dataloader
    if name == "ExperimentLogger":
        from .wandb_logger import ExperimentLogger

        return ExperimentLogger
    if name == "create_experiment_logger":
        from .wandb_logger import create_experiment_logger

        return create_experiment_logger
    if name == "WANDB_AVAILABLE":
        from .wandb_logger import WANDB_AVAILABLE

        return WANDB_AVAILABLE
    if name == "ModelEMA":
        from .ema import ModelEMA

        return ModelEMA
    if name == "AveragedModel":
        from .ema import AveragedModel

        return AveragedModel
    if name == "create_ema":
        from .ema import create_ema

        return create_ema
    if name == "EarlyStopping":
        from .early_stopping import EarlyStopping

        return EarlyStopping
    if name == "ReduceLROnPlateau":
        from .early_stopping import ReduceLROnPlateau

        return ReduceLROnPlateau
    if name == "create_early_stopping":
        from .early_stopping import create_early_stopping

        return create_early_stopping
    if name == "LRFinder":
        from .lr_finder import LRFinder

        return LRFinder
    if name == "find_optimal_lr":
        from .lr_finder import find_optimal_lr

        return find_optimal_lr
    if name == "setup_distributed":
        from .distributed import setup_distributed

        return setup_distributed
    if name == "cleanup_distributed":
        from .distributed import cleanup_distributed

        return cleanup_distributed
    if name == "get_device":
        from .distributed import get_device

        return get_device
    if name == "wrap_model_ddp":
        from .distributed import wrap_model_ddp

        return wrap_model_ddp
    if name == "reduce_tensor":
        from .distributed import reduce_tensor

        return reduce_tensor
    if name == "is_main_process":
        from .distributed import is_main_process

        return is_main_process
    if name == "get_world_size":
        from .distributed import get_world_size

        return get_world_size
    if name == "get_rank":
        from .distributed import get_rank

        return get_rank
    if name == "barrier":
        from .distributed import barrier

        return barrier
    if name == "DistributedMetricLogger":
        from .distributed import DistributedMetricLogger

        return DistributedMetricLogger
    if name == "ProgressiveStage":
        from .advanced_training import ProgressiveStage

        return ProgressiveStage
    if name == "ProgressiveTrainingScheduler":
        from .advanced_training import ProgressiveTrainingScheduler

        return ProgressiveTrainingScheduler
    if name == "create_progressive_stages":
        from .advanced_training import create_progressive_stages

        return create_progressive_stages
    if name == "ConfidenceWeightedLoss":
        from .advanced_training import ConfidenceWeightedLoss

        return ConfidenceWeightedLoss
    if name == "PseudoLabelTrainer":
        from .advanced_training import PseudoLabelTrainer

        return PseudoLabelTrainer
    if name == "TemporalConsistencyLoss":
        from .advanced_training import TemporalConsistencyLoss

        return TemporalConsistencyLoss
    if name == "DomainAdaptationLoss":
        from .advanced_training import DomainAdaptationLoss

        return DomainAdaptationLoss
    if name == "MultiIterationLoss":
        from .advanced_training import MultiIterationLoss

        return MultiIterationLoss
    if name == "create_advanced_training_components":
        from .advanced_training import create_advanced_training_components

        return create_advanced_training_components
    raise AttributeError(f"module 'utils' has no attribute {name!r}")
