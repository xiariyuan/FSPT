"""
FSPT (Frequency-Semantic Point Tracking) Models

核心模块:
- LearnableFrequencyDecomposition: 可学习频率分解 (从FM-Track复用)
- SemanticEnhancedLFD: 语义增强的频率分解
- FrequencyAwareOcclusionPredictor: 频率感知遮挡预测
- FSPTTracker: 完整的点追踪模型
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

_AVAILABLE_MODULES: Dict[str, bool] = {
    "lfd": False,
    "ftt": False,
    "semantic": False,
    "fusion": False,
    "occlusion": False,
    "tracker": False,
    "cotracker_refiner": False,
}

_EXPORTS = {
    "LearnableFrequencyDecomposition": ("learnable_freq_decomposition", "LearnableFrequencyDecomposition"),
    "LearnableFrequencyFilter": ("learnable_freq_decomposition", "LearnableFrequencyFilter"),
    "FrequencyAwarePositionalEncoding": ("freq_temporal_transformer", "FrequencyAwarePositionalEncoding"),
    "BandSpecificTemporalAttention": ("freq_temporal_transformer", "BandSpecificTemporalAttention"),
    "FrequencyTemporalTransformer": ("freq_temporal_transformer", "FrequencyTemporalTransformer"),
    "SemanticEncoder": ("semantic_encoder", "SemanticEncoder"),
    "SemanticEnhancedLFD": ("freq_semantic_fusion", "SemanticEnhancedLFD"),
    "FrequencyAwareOcclusionPredictor": ("occlusion_predictor", "FrequencyAwareOcclusionPredictor"),
    "FSPTTracker": ("point_tracker", "FSPTTracker"),
    "CoTrackerFSPTRefiner": ("cotracker_refiner", "CoTrackerFSPTRefiner"),
}

_GROUP_EXPORTS = {
    "lfd": ["LearnableFrequencyDecomposition", "LearnableFrequencyFilter"],
    "ftt": [
        "FrequencyAwarePositionalEncoding",
        "BandSpecificTemporalAttention",
        "FrequencyTemporalTransformer",
    ],
    "semantic": ["SemanticEncoder"],
    "fusion": ["SemanticEnhancedLFD"],
    "occlusion": ["FrequencyAwareOcclusionPredictor"],
    "tracker": ["FSPTTracker"],
    "cotracker_refiner": ["CoTrackerFSPTRefiner"],
}


def _try_import(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f".{module_name}", __name__)
    attr = getattr(module, attr_name)
    if module_name == "learnable_freq_decomposition":
        _AVAILABLE_MODULES["lfd"] = True
    elif module_name == "freq_temporal_transformer":
        _AVAILABLE_MODULES["ftt"] = True
    elif module_name == "semantic_encoder":
        _AVAILABLE_MODULES["semantic"] = True
    elif module_name == "freq_semantic_fusion":
        _AVAILABLE_MODULES["fusion"] = True
    elif module_name == "occlusion_predictor":
        _AVAILABLE_MODULES["occlusion"] = True
    elif module_name == "point_tracker":
        _AVAILABLE_MODULES["tracker"] = True
    elif module_name == "cotracker_refiner":
        _AVAILABLE_MODULES["cotracker_refiner"] = True
    return attr


def _prime_module_status() -> None:
    """
    Probe optional modules without importing the whole package chain eagerly.

    Each probe is isolated so a failure in one module does not prevent the rest
    of the package from remaining importable.
    """
    probes = [
        ("learnable_freq_decomposition", "LearnableFrequencyDecomposition", "lfd"),
        ("freq_temporal_transformer", "FrequencyTemporalTransformer", "ftt"),
        ("semantic_encoder", "SemanticEncoder", "semantic"),
        ("freq_semantic_fusion", "SemanticEnhancedLFD", "fusion"),
        ("occlusion_predictor", "FrequencyAwareOcclusionPredictor", "occlusion"),
        ("point_tracker", "FSPTTracker", "tracker"),
        ("cotracker_refiner", "CoTrackerFSPTRefiner", "cotracker_refiner"),
    ]
    for module_name, attr_name, key in probes:
        try:
            importlib.import_module(f".{module_name}", __name__)
            _AVAILABLE_MODULES[key] = True
        except Exception as exc:
            _AVAILABLE_MODULES[key] = False
            logger.debug("Optional model module unavailable: %s.%s (%s)", module_name, attr_name, exc)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = _try_import(module_name, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def get_available_modules():
    """返回可用模块列表"""
    _prime_module_status()
    return {k: v for k, v in _AVAILABLE_MODULES.items() if v}


def get_module_status():
    """返回模块可用性状态（包含可用与不可用项）"""
    _prime_module_status()
    return dict(_AVAILABLE_MODULES)


def check_dependencies():
    """检查所有依赖是否可用"""
    _prime_module_status()
    print("=" * 50)
    print("FSPT Module Availability")
    print("=" * 50)

    modules = {
        "lfd": "LearnableFrequencyDecomposition (FM-Track)",
        "ftt": "FrequencyTemporalTransformer (FM-Track)",
        "semantic": "SemanticEncoder (new)",
        "fusion": "SemanticEnhancedLFD (new)",
        "occlusion": "FrequencyAwareOcclusionPredictor (new)",
        "tracker": "FSPTTracker (new)",
        "cotracker_refiner": "CoTrackerFSPTRefiner (hybrid, optional)",
    }

    for key, name in modules.items():
        status = "✓" if _AVAILABLE_MODULES.get(key, False) else "✗"
        print(f"  {status} {name}")

    print("=" * 50)

    required = ["semantic", "fusion", "occlusion", "tracker"]
    return all(_AVAILABLE_MODULES.get(key, False) for key in required)


__all__ = ["get_available_modules", "get_module_status", "check_dependencies"]
for key in _GROUP_EXPORTS:
    if _AVAILABLE_MODULES.get(key, False):
        __all__ += _GROUP_EXPORTS[key]
