#!/usr/bin/env python3
"""
FSPT (Frequency-Semantic Point Tracking) 训练脚本

使用方法:
    python train.py --config configs/fspt_base.yaml
    python train.py --config configs/fspt_base.yaml --resume checkpoints/fspt_base/latest.pth
"""

import os
import ast
import sys
import argparse
import logging
import json
import time
import random
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# 配置
from omegaconf import OmegaConf

# 日志和可视化
from tqdm import tqdm
from utils.wandb_logger import create_experiment_logger
from utils.ema import create_ema
from utils.early_stopping import create_early_stopping

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _configure_hf_download_endpoint() -> None:
    """
    Prefer a domestic Hugging Face mirror when the user has not configured one.

    This keeps timm / huggingface_hub backbone downloads from repeatedly failing
    on direct public endpoints in restricted environments.
    """
    if os.environ.get("HF_ENDPOINT") or os.environ.get("HF_HUB_ENDPOINT"):
        return
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"


def _safe_bce_prob(input_prob: torch.Tensor, target_prob: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """
    Compute BCE on probability inputs in fp32 to avoid AMP autocast runtime errors.

    PyTorch 2.x marks F.binary_cross_entropy/BCELoss as unsafe under autocast on CUDA.
    Keep model outputs as probabilities (sigmoid path) but evaluate BCE in fp32.
    """
    eps = 1e-6
    with autocast(enabled=False):
        input_prob = torch.nan_to_num(input_prob.float(), nan=0.5, posinf=1.0, neginf=0.0)
        target_prob = torch.nan_to_num(target_prob.float(), nan=0.0, posinf=1.0, neginf=0.0)
        input_prob = input_prob.clamp(eps, 1.0 - eps)
        target_prob = target_prob.clamp(0.0, 1.0)
        return F.binary_cross_entropy(
            input_prob,
            target_prob,
            reduction=reduction,
        )


def _compute_reappearance_mask_for_loss(
    base_visibility: torch.Tensor,
    query_t: torch.Tensor,
    *,
    min_occlusion_len: int,
    frames_after: int,
    vis_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Build a train-time mask for frames right after a long occlusion run ends.

    This mirrors the Route A "reappearance" intuition: the hard frames are not
    the whole trajectory, but the visible frames immediately following a long
    occlusion where relocalization is expected to matter.
    """
    if not isinstance(base_visibility, torch.Tensor) or base_visibility.dim() != 3:
        raise ValueError(
            f"base_visibility must be a torch.Tensor with shape (B,N,T), got {type(base_visibility)}"
        )
    if not isinstance(query_t, torch.Tensor) or query_t.dim() != 2:
        raise ValueError(f"query_t must be a torch.Tensor with shape (B,N), got {type(query_t)}")

    bsz, num_points, num_frames = base_visibility.shape
    if query_t.shape != (bsz, num_points):
        raise ValueError(
            f"query_t shape mismatch: expected {(bsz, num_points)}, got {tuple(query_t.shape)}"
        )

    min_len = max(int(min_occlusion_len), 1)
    after = max(int(frames_after), 1)
    vis_thr = float(vis_threshold)

    device = base_visibility.device
    t_idx = torch.arange(num_frames, device=device).view(1, 1, num_frames)
    after_query = t_idx >= query_t.clamp(0, num_frames - 1).unsqueeze(-1)

    visible = (base_visibility >= vis_thr) & after_query
    occluded = (~visible) & after_query

    run = torch.zeros((bsz, num_points), device=device, dtype=torch.int32)
    run_len = torch.zeros((bsz, num_points, num_frames), device=device, dtype=torch.int32)
    for t in range(num_frames):
        occ_t = occluded[:, :, t]
        run = torch.where(occ_t, run + 1, torch.zeros_like(run))
        run_len[:, :, t] = run

    run_before = torch.cat(
        [
            torch.zeros((bsz, num_points, 1), device=device, dtype=run_len.dtype),
            run_len[:, :, :-1],
        ],
        dim=-1,
    )
    reappear = visible & (run_before >= min_len)

    if after <= 1:
        return reappear

    out = torch.zeros_like(reappear)
    countdown = torch.zeros((bsz, num_points), device=device, dtype=torch.int32)
    for t in range(num_frames):
        start = reappear[:, :, t]
        countdown = torch.where(start, torch.full_like(countdown, after), countdown)
        out[:, :, t] = (countdown > 0) & visible[:, :, t]
        countdown = torch.where(countdown > 0, countdown - 1, countdown)
        countdown = torch.where(visible[:, :, t], countdown, torch.zeros_like(countdown))
    return out


def _is_distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed(config):
    """初始化分布式训练"""
    hardware_cfg = getattr(config, 'hardware', None)
    dist_cfg = getattr(hardware_cfg, 'distributed', None)
    enabled = bool(dist_cfg and getattr(dist_cfg, 'enabled', False))
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        try:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        except ValueError:
            world_size = 1
        if not enabled and world_size > 1:
            logger.warning(
                "Distributed env detected but hardware.distributed.enabled=false; "
                "enabling DDP in train.py. Consider using scripts/train_distributed.py for LR scaling."
            )
            enabled = True
    if not enabled:
        return False, 0, 1, 0
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        logger.warning("Distributed enabled but env vars missing; running single process.")
        return False, 0, 1, 0

    backend = getattr(dist_cfg, 'backend', 'nccl')
    init_method = getattr(dist_cfg, 'init_method', 'env://')
    if 'LOCAL_RANK' not in os.environ:
        logger.warning("LOCAL_RANK not set; defaulting to 0.")
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    if backend == 'nccl' and dist.is_available() and not dist.is_nccl_available():
        logger.warning("NCCL backend not available; falling back to gloo.")
        backend = 'gloo'
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        if backend == 'nccl':
            logger.warning("CUDA not available; falling back to gloo backend.")
            backend = 'gloo'

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    return True, rank, world_size, local_rank


def cleanup_distributed():
    if _is_distributed_available():
        dist.destroy_process_group()


def _resolve_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str)
    return str(path if path.is_absolute() else base_dir / path)


def _parse_gpu_ids(value) -> list:
    """解析GPU ID配置，支持list/tuple/字符串/单整数"""
    if value is None:
        return []
    try:
        if OmegaConf.is_list(value):
            return [int(v) for v in list(value)]
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        ids = []
        for v in value:
            try:
                ids.append(int(v))
            except (TypeError, ValueError):
                continue
        return ids
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                return [int(v) for v in parsed]
            if isinstance(parsed, int):
                return [parsed]
        except (ValueError, SyntaxError):
            pass
        parts = [p.strip() for p in s.split(',') if p.strip()]
        ids = []
        for p in parts:
            try:
                ids.append(int(p))
            except ValueError:
                continue
        return ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = 0
    trainable = 0
    for p in model.parameters():
        n = int(p.numel())
        total += n
        if bool(p.requires_grad):
            trainable += n
    return total, trainable


def _append_jsonl(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to append jsonl {path}: {exc}")


def _cfg_to_plain(value):
    if value is None:
        return None
    try:
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _normalize_name_list(value) -> List[str]:
    value = _cfg_to_plain(value)
    if value is None:
        return []
    if isinstance(value, str):
        parts = [v.strip() for v in value.split(",")]
        return [v for v in parts if v]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def _normalize_param_name(name: str) -> str:
    normalized = str(name)
    while normalized.startswith("module."):
        normalized = normalized[len("module."):]
    return normalized


def _matches_prefix(name: str, prefixes: List[str]) -> bool:
    normalized = _normalize_param_name(name)
    for prefix in prefixes:
        p = str(prefix).strip()
        if not p:
            continue
        if normalized == p or normalized.startswith(p + ".") or normalized.startswith(p):
            return True
    return False


def _resolve_training_phase_schedule(config) -> List[Dict[str, Any]]:
    training_cfg = getattr(config, "training", None)
    phase_cfg = getattr(training_cfg, "phase_schedule", None) if training_cfg is not None else None
    if phase_cfg is None or not bool(getattr(phase_cfg, "enabled", False)):
        return []
    phases = getattr(phase_cfg, "phases", None)
    if phases is None:
        return []

    resolved: List[Dict[str, Any]] = []
    for idx, raw_phase in enumerate(list(phases)):
        phase = _cfg_to_plain(raw_phase)
        if not isinstance(phase, dict):
            continue
        start_epoch = int(phase.get("start_epoch", idx * 1000) or 0)
        resolved.append(
            {
                "name": str(phase.get("name", f"phase_{idx}")),
                "start_epoch": start_epoch,
                "freeze_modules": _normalize_name_list(phase.get("freeze_modules")),
                "unfreeze_modules": _normalize_name_list(phase.get("unfreeze_modules")),
                "trainable_modules": _normalize_name_list(phase.get("trainable_modules")),
                "lr_scale_overrides": _cfg_to_plain(phase.get("lr_scale_overrides")),
            }
        )
    resolved.sort(key=lambda item: int(item.get("start_epoch", 0)))
    return resolved


def _find_active_phase(phases: List[Dict[str, Any]], epoch: int) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    active_idx: Optional[int] = None
    for idx, phase in enumerate(phases):
        if int(phase.get("start_epoch", 0)) <= int(epoch):
            active_idx = idx
        else:
            break
    if active_idx is None:
        return None, None
    return active_idx, phases[active_idx]


def _apply_phase_trainability(model: nn.Module, phase: Dict[str, Any]) -> Dict[str, int]:
    freeze_modules = _normalize_name_list(phase.get("freeze_modules"))
    unfreeze_modules = _normalize_name_list(phase.get("unfreeze_modules"))
    trainable_modules = _normalize_name_list(phase.get("trainable_modules"))

    total = 0
    trainable = 0
    changed = 0

    for name, param in model.named_parameters():
        current = bool(param.requires_grad)
        target = current
        if trainable_modules:
            target = _matches_prefix(name, trainable_modules)
        else:
            if unfreeze_modules and _matches_prefix(name, unfreeze_modules):
                target = True
            if freeze_modules and _matches_prefix(name, freeze_modules):
                target = False

        if current != target:
            param.requires_grad = target
            changed += int(param.numel())
        total += int(param.numel())
        if target:
            trainable += int(param.numel())

    return {
        "changed_params": changed,
        "total_params": total,
        "trainable_params": trainable,
    }


def _apply_phase_lr_multipliers(optimizer, phase: Dict[str, Any]) -> bool:
    raw = phase.get("lr_scale_overrides", None)
    if raw is None:
        return False
    if not isinstance(raw, dict):
        logger.warning(f"Ignoring lr_scale_overrides with invalid type: {type(raw)}")
        return False

    overrides: Dict[str, float] = {}
    for key, value in raw.items():
        tag = str(key).strip().lower()
        if not tag:
            continue
        try:
            overrides[tag] = float(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid lr_scale_overrides[{key}]={value}, skipping.")
    if not overrides:
        return False

    for group in optimizer.param_groups:
        tag = str(group.get("group_tag", "other")).strip().lower()
        group["phase_lr_multiplier"] = float(overrides.get(tag, 1.0))
    return True


def _collect_module_status(config, model: Optional[nn.Module] = None) -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    try:
        from models import get_module_status

        status["available_modules"] = get_module_status()
    except Exception as exc:
        status["available_modules_error"] = str(exc)

    if model is not None:
        model_unwrapped = unwrap_model(model)
        status["model_type"] = model_unwrapped.__class__.__name__
        status["use_semantic"] = bool(getattr(model_unwrapped, "use_semantic", False))
        status["use_frequency"] = bool(getattr(model_unwrapped, "use_frequency", False))
        status["use_occlusion"] = bool(getattr(model_unwrapped, "use_occlusion", False))
        status["num_refinement_iters"] = int(getattr(model_unwrapped, "num_refinement_iters", 1) or 1)
        freq_module = getattr(model_unwrapped, "freq_semantic", None)
        if freq_module is not None:
            status["frequency_module"] = {
                "class": freq_module.__class__.__name__,
                "lfd_backend": getattr(freq_module, "lfd_backend", None),
                "strict_lfd": bool(getattr(freq_module, "strict_lfd", False)),
            }
        semantic_encoder = getattr(model_unwrapped, "semantic_encoder", None)
        if semantic_encoder is not None:
            status["semantic_encoder"] = {
                "backend": str(getattr(semantic_encoder, "clip_model_name", "unknown")),
                "freeze": bool(getattr(semantic_encoder, "freeze", True)),
                "output_dim": int(getattr(semantic_encoder, "output_dim", 0) or 0),
            }

    train_cfg = getattr(getattr(config, "data", None), "train", None)
    if train_cfg is not None:
        status["data_train"] = {
            "dataset": str(getattr(train_cfg, "dataset", "")),
            "root": str(getattr(train_cfg, "root", "")),
            "backend": str(getattr(train_cfg, "backend", "auto")),
            "use_tfds": bool(getattr(train_cfg, "use_tfds", False)),
            "annotation_file": getattr(train_cfg, "annotation_file", None),
        }
    return status


def _validate_required_modules(config, module_status: Dict[str, Any]) -> None:
    model_cfg = getattr(config, "model", None)
    strict_modules = bool(getattr(model_cfg, "strict_modules", False)) if model_cfg is not None else False
    if not strict_modules:
        return

    required = _normalize_name_list(getattr(model_cfg, "required_modules", None))
    if not required:
        required = ["semantic", "fusion", "occlusion", "tracker"]
        freq_cfg = getattr(model_cfg, "frequency", None)
        if freq_cfg is not None and bool(getattr(freq_cfg, "enabled", False)):
            required.append("lfd")

    available = module_status.get("available_modules", {})
    missing = [name for name in required if not bool(available.get(name, False))]
    if missing:
        raise RuntimeError(
            "Missing required modules under strict mode: "
            + ", ".join(missing)
            + ". Set model.strict_modules=false to allow degraded fallback."
        )


def _get_refiner_stage_cfg(config, stage_idx: Optional[int]):
    if stage_idx is None:
        return None
    try:
        idx = int(stage_idx)
    except Exception:
        return None
    refiner_cfg = getattr(getattr(config, "model", None), "refiner", None)
    stages = getattr(refiner_cfg, "stages", None) if refiner_cfg is not None else None
    if stages is None:
        return None
    try:
        stages_list = list(stages)
    except Exception:
        return None
    if 0 <= idx < len(stages_list):
        stage = stages_list[idx]
        if isinstance(stage, dict):
            return stage
        try:
            # OmegaConf nodes may behave like dicts.
            return dict(stage)
        except Exception:
            return None
    return None


def _get_amp_dtype(config) -> torch.dtype:
    training_cfg = getattr(config, 'training', None)
    amp_cfg = getattr(training_cfg, 'amp', None) if training_cfg is not None else None
    dtype = str(getattr(amp_cfg, 'dtype', 'float16')).lower()
    if dtype in ['bf16', 'bfloat16']:
        if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported'):
            if not torch.cuda.is_bf16_supported():
                logger.warning("bfloat16 not supported on this GPU; falling back to float16.")
                return torch.float16
        return torch.bfloat16
    return torch.float16


def parse_args():
    parser = argparse.ArgumentParser(description='FSPT Training')
    parser.add_argument('--config', type=str, default='configs/fspt_base.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with reduced data')
    args, unknown = parser.parse_known_args()
    return args, unknown


def _merge_with_defaults(config, config_dir: Path):
    defaults = config.pop('defaults', None)
    if not defaults:
        return config

    merged = OmegaConf.create()
    for entry in defaults:
        if isinstance(entry, str):
            if entry == '_self_':
                continue
            base_path = (config_dir / entry)
            if base_path.suffix == '':
                base_path = base_path.with_suffix('.yaml')
            if not base_path.exists():
                raise FileNotFoundError(f"Default config not found: {base_path}")
            base_cfg = OmegaConf.load(base_path)
            base_cfg = _merge_with_defaults(base_cfg, base_path.parent)
            merged = OmegaConf.merge(merged, base_cfg)
        elif isinstance(entry, dict):
            for key, value in entry.items():
                if key == '_self_':
                    continue
                if value is None:
                    base_path = (config_dir / key)
                else:
                    base_path = (config_dir / key / str(value))
                if base_path.suffix == '':
                    base_path = base_path.with_suffix('.yaml')
                if not base_path.exists():
                    raise FileNotFoundError(f"Default config not found: {base_path}")
                base_cfg = OmegaConf.load(base_path)
                base_cfg = _merge_with_defaults(base_cfg, base_path.parent)
                merged = OmegaConf.merge(merged, base_cfg)

    return OmegaConf.merge(merged, config)


def load_config(config_path):
    """加载配置文件（支持defaults继承）"""
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    return _merge_with_defaults(config, config_path.parent)


def create_model(config):
    """
    创建FSPT模型
    
    根据配置决定使用哪些模块
    """
    from models import check_dependencies
    check_dependencies()
    
    # 动态导入可用模块
    model_config = config.model
    model_type = str(getattr(model_config, 'type', 'fspt') or 'fspt').lower().strip()
    
    if model_type in ['cotracker_refiner', 'cotracker_fspt_refiner', 'hybrid_cotracker']:
        try:
            from models.cotracker_refiner import CoTrackerFSPTRefiner
            model = CoTrackerFSPTRefiner(model_config)
            logger.info("Created CoTrackerFSPTRefiner model")
        except ImportError as exc:
            logger.warning(f"CoTrackerFSPTRefiner not available: {exc}. Falling back to FSPTTracker.")
            model_type = 'fspt'

    if model_type in ['fspt', 'default', 'baseline']:
        # 尝试导入完整模型
        try:
            from models.point_tracker import FSPTTracker
            model = FSPTTracker(model_config)
            logger.info("Created FSPTTracker model")
        except ImportError:
            # 使用简化模型
            logger.warning("FSPTTracker not available, using simplified model")
            model = create_simplified_model(model_config)

    loss_cfg = getattr(config, "loss", None)
    verifier_cfg = getattr(loss_cfg, "relocal_acceptor", None)
    if verifier_cfg is None:
        verifier_cfg = getattr(loss_cfg, "verifier", None)
    if verifier_cfg is not None:
        eval_threshold = float(
            getattr(
                verifier_cfg,
                "eval_threshold",
                getattr(
                    verifier_cfg,
                    "threshold",
                    getattr(model, "relocal_acceptor_threshold", 0.5),
                ),
            )
            or getattr(model, "relocal_acceptor_threshold", 0.5)
        )
        eval_threshold = max(0.0, min(1.0, eval_threshold))
        setattr(model, "relocal_acceptor_eval_threshold", eval_threshold)
        setattr(model, "relocal_acceptor_eval_margin", float(getattr(verifier_cfg, "margin", 0.0) or 0.0))
        setattr(
            model,
            "relocal_acceptor_eval_base_error_threshold",
            float(getattr(verifier_cfg, "base_error_threshold", 0.0) or 0.0),
        )
        setattr(model, "verifier_eval_threshold", eval_threshold)
        setattr(model, "verifier_eval_margin", getattr(model, "relocal_acceptor_eval_margin", 0.0))
        setattr(
            model,
            "verifier_eval_base_error_threshold",
            getattr(model, "relocal_acceptor_eval_base_error_threshold", 0.0),
        )

    return model


def create_simplified_model(config):
    """
    创建简化版模型（用于初期调试）
    """
    class SimplifiedFSPT(nn.Module):
        def __init__(self, config):
            super().__init__()
            temporal_cfg = getattr(config, 'temporal', None)
            self.dim = int(getattr(temporal_cfg, 'dim', 256) or 256)
            
            # 简单的特征提取器
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, self.dim, 3, padding=1),
            )
            
            # 时序建模
            self.temporal_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.dim,
                    nhead=8,
                    dim_feedforward=self.dim * 4,
                    dropout=0.1,
                    batch_first=True,
                ),
                num_layers=4,
            )
            
            # 预测头
            self.position_head = nn.Linear(self.dim, 2)
            self.visibility_head = nn.Linear(self.dim, 1)
            
        def forward(self, video, query_points):
            """
            Args:
                video: (B, T, 3, H, W)
                query_points: (B, N, 3) [t, y, x]
                
            Returns:
                tracks: (B, N, T, 2)
                visibility: (B, N, T)
            """
            B, T, C, H, W = video.shape
            N = query_points.shape[1]
            
            # 提取特征
            video_flat = video.reshape(B * T, C, H, W)
            features = self.feature_extractor(video_flat)  # (B*T, dim, H', W')
            _, dim, H_out, W_out = features.shape
            features = features.reshape(B, T, dim, H_out, W_out)
            
            # 在查询点位置采样特征
            # 简化：使用平均池化
            query_features = features.mean(dim=(3, 4))  # (B, T, dim)
            
            # 时序建模
            temporal_features = self.temporal_encoder(query_features)  # (B, T, dim)
            
            # 扩展到每个点
            temporal_features = temporal_features.unsqueeze(1).expand(-1, N, -1, -1)
            temporal_features = temporal_features.reshape(B * N * T, dim)
            
            # 预测
            position_delta = self.position_head(temporal_features)
            visibility_logits = self.visibility_head(temporal_features)
            
            position_delta = position_delta.reshape(B, N, T, 2)
            visibility = visibility_logits.reshape(B, N, T).sigmoid()
            
            # 计算轨迹
            init_positions = query_points[:, :, 1:3].unsqueeze(2)  # (B, N, 1, 2)
            tracks = init_positions + position_delta
            
            return tracks, visibility
    
    return SimplifiedFSPT(config)


def _disable_temporal_augmentation_cfg(augmentation_cfg):
    """Disable dataset-level temporal augmentation while keeping spatial/color augmentation intact."""
    if augmentation_cfg is None or isinstance(augmentation_cfg, bool):
        return augmentation_cfg

    cfg = augmentation_cfg
    try:
        if OmegaConf is not None and OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        cfg = augmentation_cfg

    if isinstance(cfg, dict):
        out = dict(cfg)
    else:
        try:
            out = dict(cfg)
        except Exception:
            return augmentation_cfg

    # Remove legacy keys that can force-enable temporal augmentation via key presence.
    for key in ("random_reverse", "random_speed", "random_start"):
        out.pop(key, None)

    temporal = out.get("temporal", None)
    if isinstance(temporal, dict):
        temporal_out = dict(temporal)
    else:
        temporal_out = {}
    temporal_out["enabled"] = False
    out["temporal"] = temporal_out
    return out


def _collect_optional_dataset_kwargs(cfg, keys):
    collected = {}
    if cfg is None:
        return collected
    for key in keys:
        if hasattr(cfg, key):
            value = getattr(cfg, key)
            if value is not None:
                collected[key] = value
    return collected


def create_dataloaders(
    config,
    debug: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    is_main_process: bool = True,
):
    """创建数据加载器"""
    from datasets import get_dataloader

    data_loading_cfg = getattr(config.training, 'data_loading', None)
    pin_memory = True
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'pin_memory'):
        pin_memory = bool(data_loading_cfg.pin_memory)
    persistent_workers = False
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'persistent_workers'):
        persistent_workers = bool(data_loading_cfg.persistent_workers)
    prefetch_factor = None
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'prefetch_factor'):
        prefetch_factor = getattr(data_loading_cfg, 'prefetch_factor')
    multiprocessing_context = None
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'multiprocessing_context'):
        multiprocessing_context = getattr(data_loading_cfg, 'multiprocessing_context')
    
    # 训练数据
    train_config = config.data.train
    train_dataset_name = str(train_config.dataset).lower()
    extra_train_args = {}
    if 'kubric' in train_dataset_name or 'kinetics' in train_dataset_name:
        if hasattr(train_config, 'backend'):
            backend = getattr(train_config, 'backend')
            if backend is not None:
                extra_train_args['backend'] = str(backend)
        if hasattr(train_config, 'allow_synthetic_tracks'):
            extra_train_args['allow_synthetic_tracks'] = train_config.allow_synthetic_tracks
        if hasattr(train_config, 'annotation_file'):
            extra_train_args['annotation_file'] = train_config.annotation_file
        if hasattr(train_config, 'use_tfds'):
            extra_train_args['use_tfds'] = train_config.use_tfds
        if hasattr(train_config, 'tfds_name'):
            tfds_name = getattr(train_config, 'tfds_name')
            if tfds_name:
                extra_train_args['tfds_name'] = str(tfds_name)
        if hasattr(train_config, 'shuffle_buffer'):
            shuffle_buffer = getattr(train_config, 'shuffle_buffer')
            if shuffle_buffer is not None:
                extra_train_args['shuffle_buffer'] = int(shuffle_buffer)
        if hasattr(train_config, 'shuffle_files'):
            shuffle_files = getattr(train_config, 'shuffle_files')
            if shuffle_files is not None:
                extra_train_args['shuffle_files'] = bool(shuffle_files)
        if hasattr(train_config, 'tfds_low_memory'):
            extra_train_args['tfds_low_memory'] = bool(getattr(train_config, 'tfds_low_memory'))
        if hasattr(train_config, 'tfds_read_buffer_size'):
            tfds_read_buffer_size = getattr(train_config, 'tfds_read_buffer_size')
            if tfds_read_buffer_size is not None:
                extra_train_args['tfds_read_buffer_size'] = int(tfds_read_buffer_size)
        if hasattr(train_config, 'deterministic_sampling'):
            extra_train_args['deterministic_sampling'] = bool(train_config.deterministic_sampling)
            if extra_train_args['deterministic_sampling']:
                if hasattr(train_config, 'deterministic_seed') and train_config.deterministic_seed is not None:
                    extra_train_args['deterministic_seed'] = int(train_config.deterministic_seed)
                else:
                    extra_train_args['deterministic_seed'] = int(getattr(config.experiment, 'seed', 0) or 0)
    train_num_frames = getattr(train_config, 'num_frames', None)
    if 'kinetics' in train_dataset_name:
        if train_num_frames is not None and int(train_num_frames) > 0:
            extra_train_args['max_frames'] = int(train_num_frames)
        train_num_frames = None
    elif 'davis' in train_dataset_name:
        train_num_frames = None
    train_resolution = tuple(train_config.resolution) if getattr(train_config, 'resolution', None) is not None else None

    # If two-view temporal regularization is enabled, avoid double temporal augmentation at the dataset level.
    two_view_cfg = getattr(getattr(config, 'loss', None), 'two_view_consistency', None)
    two_view_enabled = bool(two_view_cfg is not None and getattr(two_view_cfg, 'enabled', False))
    two_view_temporal_cfg = getattr(two_view_cfg, 'temporal', None) if two_view_cfg is not None else None
    two_view_temporal_enabled = bool(
        two_view_temporal_cfg is not None and getattr(two_view_temporal_cfg, 'enabled', False)
    )
    auto_disable_dataset_temporal = bool(
        two_view_cfg is None or getattr(two_view_cfg, 'auto_disable_dataset_temporal', True)
    )
    train_augmentation = config.data.augmentation
    if two_view_enabled and two_view_temporal_enabled and auto_disable_dataset_temporal:
        train_augmentation = _disable_temporal_augmentation_cfg(train_augmentation)
        if is_main_process:
            logger.info(
                "two_view_consistency.temporal enabled: disabled dataset temporal augmentation "
                "(auto_disable_dataset_temporal=true)."
            )
    train_kwargs = dict(
        name=train_config.dataset,
        root=train_config.root,
        batch_size=config.training.batch_size if not debug else 2,
        split='train',
        num_points=train_config.num_points,
        resolution=train_resolution,
        augmentation=train_augmentation,
        num_workers=config.training.num_workers if not debug else 0,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        seed=config.experiment.seed,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        **extra_train_args,
    )
    if (
        ('kubric' in train_dataset_name or 'kinetics' in train_dataset_name)
        and hasattr(train_config, 'sampling')
        and train_config.sampling is not None
    ):
        train_kwargs['sampling'] = train_config.sampling
    train_kwargs.update(
        _collect_optional_dataset_kwargs(
            train_config,
            (
                "annotation_file",
                "split_strategy",
                "val_fraction",
                "bidirectional",
                "depth_tolerance",
                "max_samples",
            ),
        )
    )
    if train_num_frames is not None:
        train_kwargs['num_frames'] = train_num_frames
    if hasattr(train_config, 'base_tracks_dir') and train_config.base_tracks_dir is not None:
        train_kwargs['base_tracks_dir'] = train_config.base_tracks_dir
        if hasattr(train_config, 'base_tracks_strict'):
            train_kwargs['base_tracks_strict'] = bool(train_config.base_tracks_strict)
        if hasattr(train_config, 'base_tracks_query_tol') and train_config.base_tracks_query_tol is not None:
            try:
                train_kwargs['base_tracks_query_tol'] = float(train_config.base_tracks_query_tol)
            except Exception:
                pass
    train_loader = get_dataloader(**train_kwargs)
    
    # 验证数据
    val_config = config.data.val
    val_extra_args = {}
    val_dataset_name = str(val_config.dataset).lower()
    if 'kubric' in val_dataset_name:
        if hasattr(val_config, 'backend'):
            backend = getattr(val_config, 'backend')
            if backend is not None:
                val_extra_args['backend'] = str(backend)
        if hasattr(val_config, 'allow_synthetic_tracks'):
            val_extra_args['allow_synthetic_tracks'] = val_config.allow_synthetic_tracks
        if hasattr(val_config, 'annotation_file'):
            val_extra_args['annotation_file'] = val_config.annotation_file
        if hasattr(val_config, 'use_tfds'):
            val_extra_args['use_tfds'] = val_config.use_tfds
        if hasattr(val_config, 'tfds_name'):
            tfds_name = getattr(val_config, 'tfds_name')
            if tfds_name:
                val_extra_args['tfds_name'] = str(tfds_name)
        if hasattr(val_config, 'shuffle_buffer'):
            shuffle_buffer = getattr(val_config, 'shuffle_buffer')
            if shuffle_buffer is not None:
                val_extra_args['shuffle_buffer'] = int(shuffle_buffer)
        if hasattr(val_config, 'shuffle_files'):
            shuffle_files = getattr(val_config, 'shuffle_files')
            if shuffle_files is not None:
                val_extra_args['shuffle_files'] = bool(shuffle_files)
        if hasattr(val_config, 'tfds_low_memory'):
            val_extra_args['tfds_low_memory'] = bool(getattr(val_config, 'tfds_low_memory'))
        if hasattr(val_config, 'tfds_read_buffer_size'):
            tfds_read_buffer_size = getattr(val_config, 'tfds_read_buffer_size')
            if tfds_read_buffer_size is not None:
                val_extra_args['tfds_read_buffer_size'] = int(tfds_read_buffer_size)
        if hasattr(val_config, 'deterministic_sampling'):
            val_extra_args['deterministic_sampling'] = bool(val_config.deterministic_sampling)
            if val_extra_args['deterministic_sampling']:
                if hasattr(val_config, 'deterministic_seed') and val_config.deterministic_seed is not None:
                    val_extra_args['deterministic_seed'] = int(val_config.deterministic_seed)
                else:
                    val_extra_args['deterministic_seed'] = int(getattr(config.experiment, 'seed', 0) or 0)
        if hasattr(val_config, 'sampling') and val_config.sampling is not None:
            val_extra_args['sampling'] = val_config.sampling
    if hasattr(val_config, 'resolution') and val_config.resolution is not None:
        val_extra_args['resolution'] = tuple(val_config.resolution)
    if hasattr(val_config, 'augmentation'):
        val_extra_args['augmentation'] = val_config.augmentation
        if is_main_process:
            val_aug = val_config.augmentation
            val_aug_enabled = False
            if isinstance(val_aug, bool):
                val_aug_enabled = val_aug
            elif OmegaConf.is_config(val_aug):
                val_aug_enabled = bool(getattr(val_aug, 'enabled', True))
            elif isinstance(val_aug, dict):
                val_aug_enabled = bool(val_aug.get('enabled', True))
            else:
                val_aug_enabled = True
            if val_aug_enabled:
                logger.warning("Validation augmentation is enabled; metrics may be distorted.")
    if hasattr(val_config, 'num_points'):
        val_extra_args['num_points'] = val_config.num_points
    if 'davis' in val_dataset_name or 'kinetics' in val_dataset_name:
        query_mode = getattr(val_config, 'query_mode', None) if hasattr(val_config, 'query_mode') else None
        if query_mode is None and hasattr(config, 'evaluation') and hasattr(config.evaluation, 'query_mode'):
            query_mode = getattr(config.evaluation, 'query_mode')
        if query_mode is not None and str(query_mode).strip().lower() not in ('', 'none', 'null'):
            val_extra_args['query_mode'] = str(query_mode).strip()
        if hasattr(val_config, 'query_stride') and getattr(val_config, 'query_stride') is not None:
            try:
                val_extra_args['query_stride'] = max(1, int(val_config.query_stride))
            except Exception:
                pass
        if hasattr(val_config, 'points_order') and getattr(val_config, 'points_order') is not None:
            val_extra_args['points_order'] = str(val_config.points_order)
    if 'kinetics' in val_dataset_name:
        if hasattr(val_config, 'max_frames'):
            val_extra_args['max_frames'] = val_config.max_frames
        elif hasattr(val_config, 'num_frames') and val_config.num_frames is not None:
            if int(val_config.num_frames) > 0:
                val_extra_args['max_frames'] = int(val_config.num_frames)
    elif 'davis' not in val_dataset_name:
        if hasattr(val_config, 'num_frames') and val_config.num_frames is not None:
            val_extra_args['num_frames'] = val_config.num_frames
    if hasattr(val_config, 'base_tracks_dir') and val_config.base_tracks_dir is not None:
        val_extra_args['base_tracks_dir'] = val_config.base_tracks_dir
        if hasattr(val_config, 'base_tracks_strict'):
            val_extra_args['base_tracks_strict'] = bool(val_config.base_tracks_strict)
        if hasattr(val_config, 'base_tracks_query_tol') and val_config.base_tracks_query_tol is not None:
            try:
                val_extra_args['base_tracks_query_tol'] = float(val_config.base_tracks_query_tol)
            except Exception:
                pass
    val_extra_args.update(
        _collect_optional_dataset_kwargs(
            val_config,
            (
                "annotation_file",
                "split_strategy",
                "val_fraction",
                "bidirectional",
                "depth_tolerance",
                "max_samples",
            ),
        )
    )

    val_num_workers = config.training.num_workers if not debug else 0
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'num_workers'):
        val_num_workers = config.evaluation.num_workers if not debug else 0
    val_loader = None
    if (not distributed) or is_main_process:
        val_loader = get_dataloader(
            name=val_config.dataset,
            root=val_config.root,
            batch_size=1,
            split='val',  # 明确指定验证集
            num_workers=val_num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            seed=config.experiment.seed,
            **val_extra_args,
        )
    
    return train_loader, val_loader


def create_pseudo_dataloader(config, debug=False, distributed: bool = False, rank: int = 0, world_size: int = 1):
    """创建伪标签训练数据加载器（可选）"""
    pseudo_cfg = getattr(getattr(config, 'data', None), 'pseudo', None)
    if pseudo_cfg is None:
        return None
    if not hasattr(pseudo_cfg, 'dataset') or not hasattr(pseudo_cfg, 'root'):
        logger.warning("Pseudo labeling enabled but data.pseudo.dataset/root missing; skipping.")
        return None

    from datasets import get_dataloader

    data_loading_cfg = getattr(config.training, 'data_loading', None)
    pin_memory = True
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'pin_memory'):
        pin_memory = bool(data_loading_cfg.pin_memory)
    persistent_workers = False
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'persistent_workers'):
        persistent_workers = bool(data_loading_cfg.persistent_workers)
    prefetch_factor = None
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'prefetch_factor'):
        prefetch_factor = getattr(data_loading_cfg, 'prefetch_factor')
    multiprocessing_context = None
    if data_loading_cfg is not None and hasattr(data_loading_cfg, 'multiprocessing_context'):
        multiprocessing_context = getattr(data_loading_cfg, 'multiprocessing_context')

    extra_args = {}
    if 'kubric' in str(pseudo_cfg.dataset).lower():
        if hasattr(pseudo_cfg, 'backend'):
            backend = getattr(pseudo_cfg, 'backend')
            if backend is not None:
                extra_args['backend'] = str(backend)
        if hasattr(pseudo_cfg, 'allow_synthetic_tracks'):
            extra_args['allow_synthetic_tracks'] = pseudo_cfg.allow_synthetic_tracks
        if hasattr(pseudo_cfg, 'annotation_file'):
            extra_args['annotation_file'] = pseudo_cfg.annotation_file
        if hasattr(pseudo_cfg, 'use_tfds'):
            extra_args['use_tfds'] = pseudo_cfg.use_tfds
        if hasattr(pseudo_cfg, 'tfds_name'):
            tfds_name = getattr(pseudo_cfg, 'tfds_name')
            if tfds_name:
                extra_args['tfds_name'] = str(tfds_name)
        if hasattr(pseudo_cfg, 'shuffle_buffer'):
            shuffle_buffer = getattr(pseudo_cfg, 'shuffle_buffer')
            if shuffle_buffer is not None:
                extra_args['shuffle_buffer'] = int(shuffle_buffer)
        if hasattr(pseudo_cfg, 'shuffle_files'):
            shuffle_files = getattr(pseudo_cfg, 'shuffle_files')
            if shuffle_files is not None:
                extra_args['shuffle_files'] = bool(shuffle_files)
        if hasattr(pseudo_cfg, 'tfds_low_memory'):
            extra_args['tfds_low_memory'] = bool(getattr(pseudo_cfg, 'tfds_low_memory'))
        if hasattr(pseudo_cfg, 'tfds_read_buffer_size'):
            tfds_read_buffer_size = getattr(pseudo_cfg, 'tfds_read_buffer_size')
            if tfds_read_buffer_size is not None:
                extra_args['tfds_read_buffer_size'] = int(tfds_read_buffer_size)
        if hasattr(pseudo_cfg, 'deterministic_sampling'):
            extra_args['deterministic_sampling'] = bool(pseudo_cfg.deterministic_sampling)
            if extra_args['deterministic_sampling']:
                if hasattr(pseudo_cfg, 'deterministic_seed') and pseudo_cfg.deterministic_seed is not None:
                    extra_args['deterministic_seed'] = int(pseudo_cfg.deterministic_seed)
                else:
                    extra_args['deterministic_seed'] = int(getattr(config.experiment, 'seed', 0) or 0)

    pseudo_dataset_name = str(pseudo_cfg.dataset).lower()
    resolution = getattr(pseudo_cfg, 'resolution', None)
    if resolution is None and hasattr(config.data.train, 'resolution'):
        resolution = config.data.train.resolution
    resolution = tuple(resolution) if resolution is not None else None
    split = getattr(pseudo_cfg, 'split', 'train')
    num_frames = getattr(pseudo_cfg, 'num_frames', None)
    if num_frames is None and hasattr(config.data.train, 'num_frames'):
        num_frames = config.data.train.num_frames
    if 'kinetics' in pseudo_dataset_name:
        if num_frames is not None and int(num_frames) > 0:
            extra_args['max_frames'] = int(num_frames)
        num_frames = None
    elif 'davis' in pseudo_dataset_name:
        num_frames = None
    num_points = getattr(pseudo_cfg, 'num_points', None)
    if num_points is None and hasattr(config.data.train, 'num_points'):
        num_points = config.data.train.num_points
    augmentation = getattr(pseudo_cfg, 'augmentation', None)
    if augmentation is None and hasattr(config.data, 'augmentation'):
        augmentation = config.data.augmentation

    # If two-view temporal regularization is enabled, avoid double temporal augmentation for pseudo data.
    two_view_cfg = getattr(getattr(config, 'loss', None), 'two_view_consistency', None)
    two_view_enabled = bool(two_view_cfg is not None and getattr(two_view_cfg, 'enabled', False))
    two_view_temporal_cfg = getattr(two_view_cfg, 'temporal', None) if two_view_cfg is not None else None
    two_view_temporal_enabled = bool(
        two_view_temporal_cfg is not None and getattr(two_view_temporal_cfg, 'enabled', False)
    )
    auto_disable_dataset_temporal = bool(
        two_view_cfg is None or getattr(two_view_cfg, 'auto_disable_dataset_temporal', True)
    )
    if two_view_enabled and two_view_temporal_enabled and auto_disable_dataset_temporal:
        augmentation = _disable_temporal_augmentation_cfg(augmentation)
        if rank == 0:
            logger.info(
                "two_view_consistency.temporal enabled: disabled pseudo dataset temporal augmentation "
                "(auto_disable_dataset_temporal=true)."
            )
    try:
        pseudo_kwargs = dict(
            name=pseudo_cfg.dataset,
            root=pseudo_cfg.root,
            batch_size=config.training.batch_size if not debug else 2,
            split=split,
            num_points=num_points,
            resolution=resolution,
            augmentation=augmentation,
            num_workers=config.training.num_workers if not debug else 0,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
            seed=config.experiment.seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            **extra_args,
        )
        if (
            ('kubric' in pseudo_dataset_name or 'kinetics' in pseudo_dataset_name)
            and hasattr(pseudo_cfg, 'sampling')
            and pseudo_cfg.sampling is not None
        ):
            pseudo_kwargs['sampling'] = pseudo_cfg.sampling
        elif (
            ('kubric' in pseudo_dataset_name or 'kinetics' in pseudo_dataset_name)
            and hasattr(config.data.train, 'sampling')
            and config.data.train.sampling is not None
        ):
            pseudo_kwargs['sampling'] = config.data.train.sampling
        pseudo_kwargs.update(
            _collect_optional_dataset_kwargs(
                pseudo_cfg,
                (
                    "annotation_file",
                    "split_strategy",
                    "val_fraction",
                    "bidirectional",
                    "depth_tolerance",
                    "max_samples",
                ),
            )
        )
        if num_frames is not None:
            pseudo_kwargs['num_frames'] = num_frames
        pseudo_loader = get_dataloader(**pseudo_kwargs)
    except Exception as exc:
        logger.warning(f"Failed to create pseudo dataloader: {exc}. Disabling pseudo-labeling.")
        return None
    if pseudo_loader is None:
        return None
    if len(pseudo_loader) == 0:
        logger.warning("Pseudo dataloader is empty; disabling pseudo-labeling.")
        return None
    return pseudo_loader


def _set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        scale = group.get('lr_scale', 1.0) * group.get('phase_lr_multiplier', 1.0)
        group['lr'] = lr * scale


def _update_scheduler_base_lrs(scheduler, lr: float) -> None:
    if scheduler is None:
        return
    base_lrs = None
    if hasattr(scheduler, 'optimizer') and hasattr(scheduler.optimizer, 'param_groups'):
        base_lrs = [
            lr * group.get('lr_scale', 1.0) * group.get('phase_lr_multiplier', 1.0)
            for group in scheduler.optimizer.param_groups
        ]
    if base_lrs is None and hasattr(scheduler, 'base_lrs'):
        base_lrs = [lr for _ in scheduler.base_lrs]
    if base_lrs is not None and hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = list(base_lrs)
    if isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR):
        for sub in scheduler._schedulers:
            if base_lrs is not None and hasattr(sub, 'base_lrs'):
                sub.base_lrs = list(base_lrs)


def _get_scheduler_type(config) -> Optional[str]:
    sched_cfg = getattr(getattr(config, 'training', None), 'scheduler', None)
    if sched_cfg is None:
        return None
    sched_type = getattr(sched_cfg, 'type', None)
    if sched_type is None:
        return None
    return str(sched_type).lower()


def _scheduler_allows_lr_update(config, scheduler) -> bool:
    if scheduler is None:
        return True
    sched_type = _get_scheduler_type(config)
    return sched_type != 'onecyclelr'


def _apply_progressive_stage(stage_cfg, config, allow_lr_batch_update: bool = True) -> None:
    config.data.train.resolution = list(stage_cfg['resolution'])
    config.data.train.num_frames = int(stage_cfg['num_frames'])
    config.data.train.num_points = int(stage_cfg['num_points'])
    if allow_lr_batch_update:
        config.training.batch_size = int(stage_cfg['batch_size'])
        config.training.optimizer.lr = float(stage_cfg['learning_rate'])


def create_optimizer(model, config):
    """创建优化器"""
    opt_config = config.training.optimizer
    opt_type = str(getattr(opt_config, 'type', 'AdamW')).lower()
    betas = getattr(opt_config, 'betas', None)
    if betas is not None:
        betas = tuple(betas)
    clip_lr_scale_raw = getattr(opt_config, 'clip_lr_scale', 0.1)
    if clip_lr_scale_raw is None:
        clip_lr_scale = 0.1
    else:
        try:
            clip_lr_scale = float(clip_lr_scale_raw)
        except (TypeError, ValueError):
            logger.warning(f"Invalid clip_lr_scale={clip_lr_scale_raw}, fallback to 0.1")
            clip_lr_scale = 0.1
    if clip_lr_scale < 0:
        logger.warning(f"clip_lr_scale < 0 ({clip_lr_scale}); fallback to 0.1")
        clip_lr_scale = 0.1

    semantic_lr_scale_raw = getattr(opt_config, 'semantic_lr_scale', None)
    if semantic_lr_scale_raw is None:
        semantic_lr_scale = 1.0
    else:
        try:
            semantic_lr_scale = float(semantic_lr_scale_raw)
        except (TypeError, ValueError):
            logger.warning(f"Invalid semantic_lr_scale={semantic_lr_scale_raw}, fallback to 1.0")
            semantic_lr_scale = 1.0
    if semantic_lr_scale < 0:
        logger.warning(f"semantic_lr_scale < 0 ({semantic_lr_scale}); fallback to 1.0")
        semantic_lr_scale = 1.0

    backbone_lr_scale_raw = getattr(opt_config, 'backbone_lr_scale', None)
    if backbone_lr_scale_raw is None:
        backbone_lr_scale = 1.0
    else:
        try:
            backbone_lr_scale = float(backbone_lr_scale_raw)
        except (TypeError, ValueError):
            logger.warning(f"Invalid backbone_lr_scale={backbone_lr_scale_raw}, fallback to 1.0")
            backbone_lr_scale = 1.0
    if backbone_lr_scale < 0:
        logger.warning(f"backbone_lr_scale < 0 ({backbone_lr_scale}); fallback to 1.0")
        backbone_lr_scale = 1.0
    
    def _is_no_decay(name: str, param: torch.Tensor) -> bool:
        if param.ndim == 1:
            return True
        lname = name.lower()
        if lname.endswith('bias'):
            return True
        if 'norm' in lname or 'bn' in lname:
            return True
        return False

    # 分组参数（可选：对CLIP使用不同学习率，同时对norm/bias禁用weight decay）
    # When staged unfreezing is enabled, keep frozen params in optimizer so
    # that later unfreezing updates them without rebuilding optimizers.
    include_frozen = False
    try:
        refiner_cfg = getattr(getattr(config, 'model', None), 'refiner', None)
        stages = getattr(refiner_cfg, 'stages', None) if refiner_cfg is not None else None
        if stages is not None:
            try:
                include_frozen = len(stages) > 0
            except TypeError:
                include_frozen = True
    except Exception:
        include_frozen = False
    try:
        phase_schedule = _resolve_training_phase_schedule(config)
        if phase_schedule:
            include_frozen = True
    except Exception:
        pass

    semantic_decay = []
    semantic_no_decay = []
    clip_decay = []
    clip_no_decay = []
    backbone_decay = []
    backbone_no_decay = []
    other_decay = []
    other_no_decay = []
    for name, param in model.named_parameters():
        if (not include_frozen) and (not param.requires_grad):
            continue
        lname = str(name).lower()
        normalized = lname
        while normalized.startswith('module.'):
            normalized = normalized[len('module.'):]
        if normalized.startswith('base_tracker.'):
            # Route A: never optimize CoTracker base tracker.
            continue

        is_clip = ('semantic_encoder.clip_model' in normalized) or ('.clip_model.' in normalized)
        is_semantic = ('semantic_encoder.' in normalized) and (not is_clip)
        is_backbone = normalized.startswith('geo_backbone.backbone.')
        if _is_no_decay(name, param):
            if is_clip:
                clip_no_decay.append(param)
            elif is_semantic:
                semantic_no_decay.append(param)
            elif is_backbone:
                backbone_no_decay.append(param)
            else:
                other_no_decay.append(param)
        else:
            if is_clip:
                clip_decay.append(param)
            elif is_semantic:
                semantic_decay.append(param)
            elif is_backbone:
                backbone_decay.append(param)
            else:
                other_decay.append(param)

    params = []
    if other_decay:
        params.append({
            'params': other_decay,
            'lr': opt_config.lr,
            'lr_scale': 1.0,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'other',
            'weight_decay': opt_config.weight_decay,
        })
    if other_no_decay:
        params.append({
            'params': other_no_decay,
            'lr': opt_config.lr,
            'lr_scale': 1.0,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'other',
            'weight_decay': 0.0,
        })
    if backbone_decay:
        params.append({
            'params': backbone_decay,
            'lr': opt_config.lr * backbone_lr_scale,
            'lr_scale': backbone_lr_scale,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'backbone',
            'weight_decay': opt_config.weight_decay,
        })
    if backbone_no_decay:
        params.append({
            'params': backbone_no_decay,
            'lr': opt_config.lr * backbone_lr_scale,
            'lr_scale': backbone_lr_scale,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'backbone',
            'weight_decay': 0.0,
        })
    if semantic_decay:
        params.append({
            'params': semantic_decay,
            'lr': opt_config.lr * semantic_lr_scale,
            'lr_scale': semantic_lr_scale,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'semantic',
            'weight_decay': opt_config.weight_decay,
        })
    if semantic_no_decay:
        params.append({
            'params': semantic_no_decay,
            'lr': opt_config.lr * semantic_lr_scale,
            'lr_scale': semantic_lr_scale,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'semantic',
            'weight_decay': 0.0,
        })
    if clip_decay:
        params.append({
            'params': clip_decay,
            'lr': opt_config.lr * clip_lr_scale,
            'lr_scale': clip_lr_scale,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'clip',
            'weight_decay': opt_config.weight_decay,
        })
    if clip_no_decay:
        params.append({
            'params': clip_no_decay,
            'lr': opt_config.lr * clip_lr_scale,
            'lr_scale': clip_lr_scale,
            'phase_lr_multiplier': 1.0,
            'group_tag': 'clip',
            'weight_decay': 0.0,
        })
    
    if opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=opt_config.lr,
            weight_decay=0.0,
            betas=betas if betas is not None else (0.9, 0.999),
        )
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=opt_config.lr,
            weight_decay=0.0,
            betas=betas if betas is not None else (0.9, 0.999),
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config.type}")
    
    return optimizer


def create_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    sched_config = getattr(getattr(config, 'training', None), 'scheduler', None)
    if sched_config is None:
        return None
    sched_type = _get_scheduler_type(config)
    
    if sched_type == 'cosineannealinglr':
        warmup_epochs = int(getattr(sched_config, 'warmup_epochs', 0) or 0)
        main_t_max = max(1, int(sched_config.T_max) - warmup_epochs)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=main_t_max,
            eta_min=sched_config.eta_min,
        )
        if warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=warmup_epochs,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = main_scheduler
    elif sched_type == 'onecyclelr':
        max_lr = config.training.optimizer.lr
        if len(optimizer.param_groups) > 1:
            max_lr = [max_lr * group.get('lr_scale', 1.0) for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=num_training_steps,
            pct_start=0.1,
        )
    else:
        scheduler = None
    
    return scheduler


class PointTrackingLoss(nn.Module):
    """
    点追踪损失函数
    """
    def __init__(self, config):
        super().__init__()
        self.position_weight = config.loss.position.weight
        self.occlusion_weight = config.loss.occlusion.weight
        self.freq_ortho_weight = config.loss.frequency_ortho.weight
        freq_recon_cfg = getattr(config.loss, 'frequency_reconstruction', None)
        self.freq_reconstruction_weight = float(
            getattr(freq_recon_cfg, 'weight', 0.0) if freq_recon_cfg is not None else 0.0
        )
        freq_sep_cfg = getattr(config.loss, 'frequency_separation', None)
        self.freq_separation_weight = float(
            getattr(freq_sep_cfg, 'weight', 0.0) if freq_sep_cfg is not None else 0.0
        )
        self.semantic_weight = config.loss.semantic_consistency.weight
        self.position_type = str(getattr(config.loss.position, 'type', 'l1')).lower()
        self.occlusion_type = str(getattr(config.loss.occlusion, 'type', 'bce')).lower()
        pos_hard_cfg = getattr(getattr(config.loss, 'position', None), 'hard_mining', None)
        self.pos_hard_enabled = bool(pos_hard_cfg is not None and getattr(pos_hard_cfg, 'enabled', False))
        self.pos_hard_tau = float(getattr(pos_hard_cfg, 'tau', 0.015625) or 0.015625) if pos_hard_cfg is not None else 0.015625
        self.pos_hard_power = float(getattr(pos_hard_cfg, 'power', 1.0) or 1.0) if pos_hard_cfg is not None else 1.0
        self.pos_hard_min_weight = float(getattr(pos_hard_cfg, 'min_weight', 1.0) or 1.0) if pos_hard_cfg is not None else 1.0
        self.pos_hard_max_weight = float(getattr(pos_hard_cfg, 'max_weight', 5.0) or 5.0) if pos_hard_cfg is not None else 5.0
        self.pos_hard_detach_base = bool(getattr(pos_hard_cfg, 'detach_base', True)) if pos_hard_cfg is not None else True

        # Route A / Refiner training: selective supervision w.r.t base error.
        # Only compute position loss on points where base tracker is sufficiently wrong.
        # This reduces "jittering" already-correct points, which can destroy AJ at tight thresholds.
        pos_selective_cfg = getattr(getattr(config.loss, 'position', None), 'selective', None)
        self.pos_selective_enabled = bool(
            pos_selective_cfg is not None and getattr(pos_selective_cfg, 'enabled', False)
        )
        self.pos_selective_base_error_threshold = float(
            getattr(pos_selective_cfg, 'base_error_threshold', 0.0) or 0.0
        ) if pos_selective_cfg is not None else 0.0
        self.pos_selective_min_points = int(
            getattr(pos_selective_cfg, 'min_points', 256) or 256
        ) if pos_selective_cfg is not None else 256
        self.pos_selective_detach_base = bool(
            getattr(pos_selective_cfg, 'detach_base', True)
        ) if pos_selective_cfg is not None else True

        # Route A: "do no harm" regularizer w.r.t the base tracker.
        # This is different from base_track_consistency:
        # - base_track_consistency penalizes any deviation from base, even if it improves GT error.
        # - no_harm_vs_base only penalizes *being worse than base* (hinge), allowing improvements.
        pos_noharm_cfg = getattr(getattr(config.loss, 'position', None), 'no_harm_vs_base', None)
        self.pos_noharm_enabled = bool(
            pos_noharm_cfg is not None and getattr(pos_noharm_cfg, 'enabled', False)
        )
        self.pos_noharm_weight = float(
            getattr(pos_noharm_cfg, 'weight', 0.0) if pos_noharm_cfg is not None else 0.0
        )
        self.pos_noharm_margin = float(
            getattr(pos_noharm_cfg, 'margin', 0.0) if pos_noharm_cfg is not None else 0.0
        )
        self.pos_noharm_detach_base = bool(
            getattr(pos_noharm_cfg, 'detach_base', True) if pos_noharm_cfg is not None else True
        )
        # Optional: only apply no-harm hinge on points where the base is "meaningfully wrong".
        # This avoids forcing the refiner to jitter already-correct tracks just to satisfy
        # a positive margin.
        self.pos_noharm_base_error_threshold = float(
            getattr(pos_noharm_cfg, 'base_error_threshold', 0.0) if pos_noharm_cfg is not None else 0.0
        )
        self.pos_noharm_min_points = int(
            getattr(pos_noharm_cfg, 'min_points', 0) or 0
        ) if pos_noharm_cfg is not None else 0

        # Route A: explicit delta supervision (teach the refiner to predict GT - base).
        # This is often more stable than absolute supervision when the base tracker is strong.
        delta_sup_cfg = getattr(getattr(config.loss, "position", None), "delta_supervision", None)
        self.delta_sup_weight = float(
            getattr(delta_sup_cfg, "weight", 0.0) if delta_sup_cfg is not None else 0.0
        )
        delta_sup_enabled = getattr(delta_sup_cfg, "enabled", None) if delta_sup_cfg is not None else None
        if delta_sup_enabled is None:
            delta_sup_enabled = self.delta_sup_weight > 0
        self.delta_sup_enabled = bool(delta_sup_cfg is not None and delta_sup_enabled)
        self.delta_sup_type = str(
            getattr(delta_sup_cfg, "type", "l1") if delta_sup_cfg is not None else "l1"
        ).lower()
        self.delta_sup_detach_base = bool(
            getattr(delta_sup_cfg, "detach_base", True) if delta_sup_cfg is not None else True
        )
        self.delta_sup_clamp = float(
            getattr(delta_sup_cfg, "clamp", 0.0) if delta_sup_cfg is not None else 0.0
        )
        self.delta_sup_base_error_threshold = float(
            getattr(delta_sup_cfg, "base_error_threshold", 0.0) if delta_sup_cfg is not None else 0.0
        )
        self.delta_sup_min_points = int(
            getattr(delta_sup_cfg, "min_points", 0) or 0
        ) if delta_sup_cfg is not None else 0
        focus_sup_cfg = getattr(getattr(config.loss, "position", None), "focus_supervision", None)
        self.focus_sup_enabled = bool(
            focus_sup_cfg is not None and getattr(focus_sup_cfg, "enabled", False)
        )
        self.focus_sup_weight = float(
            getattr(focus_sup_cfg, "weight", 0.0) if focus_sup_cfg is not None else 0.0
        )
        self.focus_sup_mode = str(
            getattr(focus_sup_cfg, "mode", "relocal_mask") if focus_sup_cfg is not None else "relocal_mask"
        ).lower()
        self.focus_sup_base_error_threshold = float(
            getattr(focus_sup_cfg, "base_error_threshold", 0.0) if focus_sup_cfg is not None else 0.0
        )
        self.focus_sup_min_points = int(
            getattr(focus_sup_cfg, "min_points", 0) or 0
        ) if focus_sup_cfg is not None else 0
        self.focus_sup_min_occlusion_len = int(
            getattr(focus_sup_cfg, "min_occlusion_len", 30) or 30
        ) if focus_sup_cfg is not None else 30
        self.focus_sup_frames_after = int(
            getattr(focus_sup_cfg, "frames_after", 3) or 3
        ) if focus_sup_cfg is not None else 3
        self.focus_sup_vis_threshold = float(
            getattr(focus_sup_cfg, "vis_threshold", 0.5) if focus_sup_cfg is not None else 0.5
        )
        self.focus_sup_visibility_source = str(
            getattr(focus_sup_cfg, "visibility_source", "base") if focus_sup_cfg is not None else "base"
        ).lower().strip()
        if self.focus_sup_visibility_source in ("", "none", "null"):
            self.focus_sup_visibility_source = "base"
        self.smooth_l1_beta = float(getattr(config.loss.position, 'smooth_l1_beta', 1.0) or 1.0)
        self.focal_gamma = float(getattr(config.loss.occlusion, 'focal_gamma', 2.0) or 2.0)
        self.focal_alpha = float(getattr(config.loss.occlusion, 'focal_alpha', 0.25) or 0.25)
        self.semantic_num_pairs = int(
            getattr(config.loss.semantic_consistency, 'num_pairs', 256) or 256
        )
        # 时序平滑损失权重
        self.temporal_smooth_weight = float(
            getattr(config.loss, 'temporal_smooth_weight', 0.0) or 0.0
        )
        # 时序一致性正则化
        temporal_consistency_cfg = getattr(config.loss, 'temporal_consistency', None)
        self.temporal_consistency_enabled = bool(
            temporal_consistency_cfg is not None and getattr(temporal_consistency_cfg, 'enabled', False)
        )
        self.temporal_consistency_order = int(getattr(temporal_consistency_cfg, 'order', 2) or 2)
        self.temporal_consistency_weight = float(getattr(temporal_consistency_cfg, 'weight', 0.0) or 0.0)
        if self.temporal_consistency_enabled and self.temporal_smooth_weight > 0:
            logger.warning("Both temporal_smooth_weight and temporal_consistency are enabled; losses will be additive.")
        # 多迭代监督
        multi_iter_cfg = getattr(config.loss, 'multi_iteration', None)
        self.multi_iter_enabled = bool(multi_iter_cfg is not None and getattr(multi_iter_cfg, 'enabled', False))
        self.multi_iter_gamma = float(getattr(multi_iter_cfg, 'gamma', 0.8) or 0.8)
        self._warned_multi_iter = False
        # 置信度加权损失
        confidence_cfg = getattr(config.loss, 'confidence_weighted', None)
        self.confidence_weighted_enabled = bool(
            confidence_cfg is not None and getattr(confidence_cfg, 'enabled', False)
        )
        self.confidence_weight = float(getattr(confidence_cfg, 'weight', 0.0) or 0.0)
        self.min_confidence = float(getattr(confidence_cfg, 'min_confidence', 0.1) or 0.1)

        # Optional policy gate supervision (Route A): learn when to trust the refiner vs base.
        policy_gate_cfg = getattr(getattr(config, 'loss', None), 'policy_gate', None)
        self.policy_gate_enabled = bool(
            policy_gate_cfg is not None and getattr(policy_gate_cfg, 'enabled', False)
        )
        self.policy_gate_weight = float(
            getattr(policy_gate_cfg, 'weight', 0.0) if policy_gate_cfg is not None else 0.0
        )
        self.policy_gate_margin = float(
            getattr(policy_gate_cfg, 'margin', 0.0) if policy_gate_cfg is not None else 0.0
        )
        self.policy_gate_base_error_threshold = float(
            getattr(policy_gate_cfg, 'base_error_threshold', 0.0) if policy_gate_cfg is not None else 0.0
        )
        self.policy_gate_min_points = int(
            getattr(policy_gate_cfg, 'min_points', 0) or 0
        ) if policy_gate_cfg is not None else 0
        self.policy_gate_detach_pred = bool(
            getattr(policy_gate_cfg, 'detach_pred', True) if policy_gate_cfg is not None else True
        )
        self.policy_gate_detach_base = bool(
            getattr(policy_gate_cfg, 'detach_base', True) if policy_gate_cfg is not None else True
        )
        self.policy_gate_loss_type = str(
            getattr(policy_gate_cfg, 'type', 'bce') if policy_gate_cfg is not None else 'bce'
        ).lower()
        self.policy_gate_use_pre = bool(
            getattr(policy_gate_cfg, 'use_pre_gate_tracks', True) if policy_gate_cfg is not None else True
        )

        # Relocalization acceptor: learn whether a relocalized candidate should
        # replace the base tracker on hard re-appearance frames.
        relocal_accept_cfg = getattr(getattr(config, 'loss', None), 'relocal_acceptor', None)
        if relocal_accept_cfg is None:
            relocal_accept_cfg = getattr(getattr(config, 'loss', None), 'verifier', None)
        self.relocal_acceptor_enabled = bool(
            relocal_accept_cfg is not None and getattr(relocal_accept_cfg, 'enabled', False)
        )
        self.relocal_acceptor_weight = float(
            getattr(relocal_accept_cfg, 'weight', 0.0) if relocal_accept_cfg is not None else 0.0
        )
        self.relocal_acceptor_margin = float(
            getattr(relocal_accept_cfg, 'margin', 0.0) if relocal_accept_cfg is not None else 0.0
        )
        self.relocal_acceptor_base_error_threshold = float(
            getattr(relocal_accept_cfg, 'base_error_threshold', 0.0) if relocal_accept_cfg is not None else 0.0
        )
        self.relocal_acceptor_min_points = int(
            getattr(relocal_accept_cfg, 'min_points', 0) or 0
        ) if relocal_accept_cfg is not None else 0
        self.relocal_acceptor_detach_pred = bool(
            getattr(relocal_accept_cfg, 'detach_pred', True) if relocal_accept_cfg is not None else True
        )
        self.relocal_acceptor_detach_base = bool(
            getattr(relocal_accept_cfg, 'detach_base', True) if relocal_accept_cfg is not None else True
        )
        self.relocal_acceptor_loss_type = str(
            getattr(relocal_accept_cfg, 'type', 'bce') if relocal_accept_cfg is not None else 'bce'
        ).lower()
        self.relocal_acceptor_eval_threshold = float(
            getattr(
                relocal_accept_cfg,
                'eval_threshold',
                getattr(
                    relocal_accept_cfg,
                    'threshold',
                    0.5,
                ) if relocal_accept_cfg is not None else 0.5,
            ) if relocal_accept_cfg is not None else 0.5
        )
        self.relocal_acceptor_eval_threshold = max(0.0, min(1.0, self.relocal_acceptor_eval_threshold))

        # Optional framewise init supervision (TAPIR-style).
        frame_sup_cfg = getattr(getattr(config, 'loss', None), 'framewise_init_supervision', None)
        self.frame_sup_enabled = bool(
            frame_sup_cfg is not None and getattr(frame_sup_cfg, 'enabled', False)
        )
        self.frame_sup_weight = float(
            getattr(frame_sup_cfg, 'weight', 0.0) if frame_sup_cfg is not None else 0.0
        )
        self.frame_sup_type = str(
            getattr(frame_sup_cfg, 'type', 'l1') if frame_sup_cfg is not None else 'l1'
        ).lower()
        self.frame_sup_base_error_threshold = float(
            getattr(frame_sup_cfg, 'base_error_threshold', 0.0) if frame_sup_cfg is not None else 0.0
        )
        self.frame_sup_min_points = int(
            getattr(frame_sup_cfg, 'min_points', 0) or 0
        ) if frame_sup_cfg is not None else 0
        self.frame_sup_detach_init = bool(
            getattr(frame_sup_cfg, 'detach_init', True) if frame_sup_cfg is not None else True
        )

        # Base tracker visibility consistency (Route A / cached base visibility)
        base_vis_cfg = getattr(getattr(config, 'loss', None), 'base_visibility_consistency', None)
        self.base_visibility_weight = float(
            getattr(base_vis_cfg, 'weight', 0.0) if base_vis_cfg is not None else 0.0
        )
        self.base_visibility_type = str(
            getattr(base_vis_cfg, 'type', 'bce') if base_vis_cfg is not None else 'bce'
        ).lower()
        self.base_visibility_mask = str(
            getattr(base_vis_cfg, 'mask', 'all') if base_vis_cfg is not None else 'all'
        ).lower()
        self._warned_base_vis_shape = False

        # Base tracker track consistency (optional distillation / regularization).
        base_track_cfg = getattr(getattr(config, 'loss', None), 'base_track_consistency', None)
        self.base_track_weight = float(
            getattr(base_track_cfg, 'weight', 0.0) if base_track_cfg is not None else 0.0
        )
        self.base_track_type = str(
            getattr(base_track_cfg, 'type', 'l2') if base_track_cfg is not None else 'l2'
        ).lower()
        self.base_track_mask = str(
            getattr(base_track_cfg, 'mask', 'gt_visible') if base_track_cfg is not None else 'gt_visible'
        ).lower()
        self.base_track_smooth_l1_beta = float(
            getattr(base_track_cfg, 'smooth_l1_beta', self.smooth_l1_beta)
            if base_track_cfg is not None
            else self.smooth_l1_beta
        )
        # Optional: error-aware base-track regularization.
        # When enabled, we down-weight the "stay close to base" penalty if the base tracker
        # is far from ground-truth (so the refiner has room to correct).
        self.base_track_adaptive_enabled = False
        self.base_track_adaptive_tau = 0.02
        self.base_track_adaptive_power = 2.0
        self.base_track_adaptive_min_weight = 0.0
        adaptive_cfg = getattr(base_track_cfg, "adaptive", None) if base_track_cfg is not None else None
        if adaptive_cfg is not None:
            try:
                self.base_track_adaptive_enabled = bool(getattr(adaptive_cfg, "enabled", False))
                self.base_track_adaptive_tau = float(getattr(adaptive_cfg, "tau", 0.02) or 0.02)
                self.base_track_adaptive_power = float(getattr(adaptive_cfg, "power", 2.0) or 2.0)
                self.base_track_adaptive_min_weight = float(
                    getattr(adaptive_cfg, "min_weight", 0.0) or 0.0
                )
            except Exception:
                logger.warning("Invalid loss.base_track_consistency.adaptive config; disabling.")
                self.base_track_adaptive_enabled = False
        self._warned_base_track_shape = False

        # Optional: epoch-based weight schedules (paper-friendly ablation knob).
        self._current_epoch = 0
        self._total_epochs: Optional[int] = None
        self._warned_unknown_weight_schedule: set[str] = set()
        self._base_visibility_schedule = self._parse_weight_schedule(
            base_vis_cfg, default_weight=self.base_visibility_weight
        )
        self._base_track_schedule = self._parse_weight_schedule(
            base_track_cfg, default_weight=self.base_track_weight
        )

        # Optional: explicit supervision for local correlation logits (Route A).
        corr_sup_cfg = getattr(getattr(config, "loss", None), "correlation_supervision", None)
        self.corr_sup_weight = float(
            getattr(corr_sup_cfg, "weight", 0.0) if corr_sup_cfg is not None else 0.0
        )
        corr_sup_enabled = getattr(corr_sup_cfg, "enabled", None) if corr_sup_cfg is not None else None
        if corr_sup_enabled is None:
            corr_sup_enabled = self.corr_sup_weight > 0
        self.corr_sup_enabled = bool(corr_sup_cfg is not None and corr_sup_enabled)
        self.corr_sup_ignore_oow = bool(
            getattr(corr_sup_cfg, "ignore_out_of_window", True) if corr_sup_cfg is not None else True
        )
        self.corr_sup_mask = str(
            getattr(corr_sup_cfg, "mask", "gt_visible") if corr_sup_cfg is not None else "gt_visible"
        ).lower()

    @staticmethod
    def _cfg_get(cfg, key: str, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def set_epoch(self, epoch: int, total_epochs: Optional[int] = None) -> None:
        """Provide epoch context for scheduled loss weights."""
        try:
            self._current_epoch = int(epoch)
        except (TypeError, ValueError):
            self._current_epoch = 0
        if total_epochs is not None:
            try:
                self._total_epochs = int(total_epochs)
            except (TypeError, ValueError):
                self._total_epochs = None

    def _parse_weight_schedule(self, cfg, default_weight: float) -> Optional[Dict[str, object]]:
        sched_cfg = self._cfg_get(cfg, "schedule", None)
        if sched_cfg is None:
            return None
        enabled = self._cfg_get(sched_cfg, "enabled", True)
        if enabled is not None and not bool(enabled):
            return None
        sched_type = str(self._cfg_get(sched_cfg, "type", "none")).lower().strip()
        if sched_type in ("none", "", "off", "disabled", "false"):
            return None

        start_epoch = int(self._cfg_get(sched_cfg, "start_epoch", 0) or 0)
        end_epoch = self._cfg_get(sched_cfg, "end_epoch", None)
        if end_epoch is not None:
            end_epoch = int(end_epoch)
        else:
            warmup_epochs = self._cfg_get(sched_cfg, "warmup_epochs", None)
            duration_epochs = self._cfg_get(sched_cfg, "duration_epochs", None)
            if warmup_epochs is not None:
                end_epoch = start_epoch + int(warmup_epochs)
            elif duration_epochs is not None:
                end_epoch = start_epoch + int(duration_epochs)
            else:
                end_epoch = None

        start_weight = float(self._cfg_get(sched_cfg, "start_weight", 0.0) or 0.0)
        end_weight = float(self._cfg_get(sched_cfg, "end_weight", default_weight) or default_weight)
        return {
            "type": sched_type,
            "start_epoch": int(start_epoch),
            "end_epoch": end_epoch,
            "start_weight": float(start_weight),
            "end_weight": float(end_weight),
        }

    def _scheduled_weight(self, base_weight: float, schedule: Optional[Dict[str, object]], key: str) -> float:
        if schedule is None:
            return float(base_weight)

        epoch = int(getattr(self, "_current_epoch", 0) or 0)
        total_epochs = getattr(self, "_total_epochs", None)
        start_epoch = int(schedule.get("start_epoch", 0) or 0)
        end_epoch = schedule.get("end_epoch", None)
        if end_epoch is None:
            end_epoch = (int(total_epochs) - 1) if isinstance(total_epochs, int) and total_epochs > 0 else start_epoch
        else:
            end_epoch = int(end_epoch)
            if end_epoch < 0 and isinstance(total_epochs, int) and total_epochs > 0:
                end_epoch = total_epochs + end_epoch

        if end_epoch < start_epoch:
            end_epoch = start_epoch

        start_weight = float(schedule.get("start_weight", 0.0) or 0.0)
        end_weight = float(schedule.get("end_weight", base_weight) or base_weight)

        if epoch <= start_epoch:
            return start_weight
        if epoch >= end_epoch:
            return end_weight

        denom = max(int(end_epoch) - int(start_epoch), 1)
        t = float(epoch - start_epoch) / float(denom)

        sched_type = str(schedule.get("type", "linear")).lower().strip()
        if sched_type in ("linear", "lerp", "ramp"):
            blend = t
        elif sched_type in ("cosine", "cos"):
            blend = 0.5 - 0.5 * math.cos(math.pi * t)
        elif sched_type in ("step", "constant"):
            blend = 1.0
        else:
            if key not in self._warned_unknown_weight_schedule:
                logger.warning(f"Unknown weight schedule type {sched_type!r} for {key}; falling back to linear.")
                self._warned_unknown_weight_schedule.add(key)
            blend = t

        return float(start_weight + (end_weight - start_weight) * float(blend))

    def _get_base_visibility_weight(self) -> float:
        return self._scheduled_weight(self.base_visibility_weight, self._base_visibility_schedule, "base_visibility")

    def _get_base_track_weight(self) -> float:
        return self._scheduled_weight(self.base_track_weight, self._base_track_schedule, "base_tracks")
          
    def _semantic_consistency_loss(
        self,
        semantic_feat: torch.Tensor,
        pred_visibility: torch.Tensor,
        gt_visibility: torch.Tensor,
    ) -> torch.Tensor:
        """
        基于语义相似度约束可见性一致性
        Args:
            semantic_feat: (B, T, N, C)
            pred_visibility: (B, N, T)
            gt_visibility: (B, N, T)
        """
        if semantic_feat is None or self.semantic_weight <= 0:
            return torch.tensor(0.0, device=pred_visibility.device)

        B, T, N, C = semantic_feat.shape
        if N < 2:
            return torch.tensor(0.0, device=pred_visibility.device)

        semantic_feat = F.normalize(semantic_feat, dim=-1)
        vis_pred = pred_visibility.permute(0, 2, 1)  # (B, T, N)
        vis_gt = gt_visibility.permute(0, 2, 1)  # (B, T, N)

        BT = B * T
        sem_flat = semantic_feat.reshape(BT, N, C)
        vis_pred_flat = vis_pred.reshape(BT, N)
        vis_gt_flat = vis_gt.reshape(BT, N)

        num_pairs = min(self.semantic_num_pairs, N * N)
        if num_pairs <= 0:
            return torch.tensor(0.0, device=pred_visibility.device)

        idx_i = torch.randint(0, N, (BT, num_pairs), device=semantic_feat.device)
        idx_j = torch.randint(0, N, (BT, num_pairs), device=semantic_feat.device)
        batch_idx = torch.arange(BT, device=semantic_feat.device).unsqueeze(1)

        feat_i = sem_flat[batch_idx, idx_i]
        feat_j = sem_flat[batch_idx, idx_j]
        sim = (feat_i * feat_j).sum(dim=-1).clamp(min=0.0)

        vis_i = vis_pred_flat[batch_idx, idx_i]
        vis_j = vis_pred_flat[batch_idx, idx_j]
        mask = vis_gt_flat[batch_idx, idx_i] & vis_gt_flat[batch_idx, idx_j]
        mask = mask.float()

        loss = (vis_i - vis_j) ** 2
        loss = loss * sim * mask
        denom = mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    def _compute_position_error(
        self,
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
    ) -> torch.Tensor:
        if self.position_type == 'l1':
            return torch.abs(pred_tracks - gt_tracks)
        if self.position_type in ['l2', 'mse']:
            return (pred_tracks - gt_tracks) ** 2
        if self.position_type == 'smooth_l1':
            return F.smooth_l1_loss(
                pred_tracks, gt_tracks, reduction='none', beta=self.smooth_l1_beta
            )
        raise ValueError(f"Unknown position loss type: {self.position_type}")

    def _compute_position_loss(
        self,
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        base_tracks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        visible_mask = gt_visibility.unsqueeze(-1).float()
        position_error = self._compute_position_error(pred_tracks, gt_tracks)
        if (
            self.pos_selective_enabled
            and self.pos_selective_base_error_threshold > 0
            and base_tracks is not None
            and isinstance(base_tracks, torch.Tensor)
            and base_tracks.shape == gt_tracks.shape
        ):
            base = base_tracks
            if base.device != pred_tracks.device:
                base = base.to(pred_tracks.device)
            base = base.to(dtype=pred_tracks.dtype)
            gt = gt_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype)
            base_err = torch.norm((base - gt), dim=-1)  # (B,N,T)
            if self.pos_selective_detach_base:
                base_err = base_err.detach()
            sel = base_err > float(self.pos_selective_base_error_threshold)
            sel = sel.unsqueeze(-1).float()
            masked = visible_mask * sel
            if masked.sum() >= float(max(int(self.pos_selective_min_points), 1)):
                visible_mask = masked
        weight = None
        if (
            self.pos_hard_enabled
            and base_tracks is not None
            and isinstance(base_tracks, torch.Tensor)
            and base_tracks.shape == gt_tracks.shape
        ):
            base = base_tracks
            if base.device != pred_tracks.device:
                base = base.to(pred_tracks.device)
            base = base.to(dtype=pred_tracks.dtype)
            base_err = torch.norm((base - gt_tracks), dim=-1)  # (B,N,T)
            if self.pos_hard_detach_base:
                base_err = base_err.detach()
            tau = max(float(self.pos_hard_tau), 1.0e-6)
            weight = 1.0 + (base_err / tau) ** float(self.pos_hard_power)
            weight = weight.clamp(min=float(self.pos_hard_min_weight), max=float(self.pos_hard_max_weight))
            weight = weight.unsqueeze(-1)  # (B,N,T,1)

        if weight is None:
            denom = visible_mask.sum() + 1e-6
            return (position_error * visible_mask).sum() / denom

        weighted_mask = visible_mask * weight
        denom = weighted_mask.sum() + 1e-6
        return (position_error * weighted_mask).sum() / denom

    def _compute_delta_error(self, pred_delta: torch.Tensor, gt_delta: torch.Tensor) -> torch.Tensor:
        if self.delta_sup_type == 'l1':
            return torch.abs(pred_delta - gt_delta)
        if self.delta_sup_type in ['l2', 'mse']:
            return (pred_delta - gt_delta) ** 2
        if self.delta_sup_type in ['smooth_l1', 'huber']:
            return F.smooth_l1_loss(
                pred_delta, gt_delta, reduction='none', beta=self.smooth_l1_beta
            )
        raise ValueError(f"Unknown delta_supervision loss type: {self.delta_sup_type}")

    def _compute_framewise_error(self, pred_tracks: torch.Tensor, init_tracks: torch.Tensor) -> torch.Tensor:
        if self.frame_sup_type == 'l1':
            return torch.abs(pred_tracks - init_tracks)
        if self.frame_sup_type in ['l2', 'mse']:
            return (pred_tracks - init_tracks) ** 2
        if self.frame_sup_type in ['smooth_l1', 'huber']:
            return F.smooth_l1_loss(
                pred_tracks, init_tracks, reduction='none', beta=self.smooth_l1_beta
            )
        raise ValueError(f"Unknown framewise_init_supervision type: {self.frame_sup_type}")

    def _delta_supervision_loss(
        self,
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        base_tracks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.delta_sup_enabled or self.delta_sup_weight <= 0:
            return torch.tensor(0.0, device=pred_tracks.device)
        if base_tracks is None or not isinstance(base_tracks, torch.Tensor):
            return torch.tensor(0.0, device=pred_tracks.device)
        if base_tracks.shape != gt_tracks.shape or pred_tracks.shape != gt_tracks.shape:
            return torch.tensor(0.0, device=pred_tracks.device)

        base = base_tracks
        if base.device != pred_tracks.device:
            base = base.to(pred_tracks.device)
        base = base.to(dtype=pred_tracks.dtype)
        if self.delta_sup_detach_base:
            base = base.detach()

        pred_delta = pred_tracks - base
        gt_delta = gt_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype) - base
        clamp = float(self.delta_sup_clamp)
        if clamp > 0:
            gt_delta = gt_delta.clamp(min=-clamp, max=clamp)

        visible_mask = gt_visibility.unsqueeze(-1).float()
        if self.delta_sup_base_error_threshold > 0:
            base_err = torch.norm((base - gt_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype)), dim=-1)  # (B,N,T)
            if self.delta_sup_detach_base:
                base_err = base_err.detach()
            sel = base_err > float(self.delta_sup_base_error_threshold)
            masked = visible_mask * sel.unsqueeze(-1).float()
            min_points = max(int(self.delta_sup_min_points), 0)
            if min_points <= 0 or masked.sum() >= float(max(min_points, 1)):
                visible_mask = masked
        delta_error = self._compute_delta_error(pred_delta, gt_delta)
        denom = visible_mask.sum() + 1e-6
        return (delta_error * visible_mask).sum() / denom

    def _focused_position_loss(
        self,
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        *,
        base_tracks: Optional[torch.Tensor] = None,
        base_visibility: Optional[torch.Tensor] = None,
        relocal_mask: Optional[torch.Tensor] = None,
        query_points: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extra position supervision focused on relocalization-relevant frames.

        Returns:
            (loss_value, active_mask_float)
        """
        zero = torch.tensor(0.0, device=pred_tracks.device)
        empty_mask = torch.zeros_like(gt_visibility, dtype=torch.float32, device=pred_tracks.device)
        if not self.focus_sup_enabled or self.focus_sup_weight <= 0:
            return zero, empty_mask

        focus_mask: Optional[torch.Tensor] = None
        mode = str(self.focus_sup_mode).lower().strip()

        if (
            mode == "relocal_mask"
            and isinstance(relocal_mask, torch.Tensor)
            and relocal_mask.shape == gt_visibility.shape
        ):
            focus_mask = relocal_mask.to(device=pred_tracks.device)
            if focus_mask.dtype != torch.bool:
                focus_mask = focus_mask > 0.5
        elif (
            mode in ("reappearance", "reappear")
            and isinstance(query_points, torch.Tensor)
            and query_points.dim() >= 3
            and query_points.shape[-1] >= 1
        ):
            vis_source = str(self.focus_sup_visibility_source).lower().strip()
            focus_visibility: Optional[torch.Tensor] = None
            if vis_source in ("gt", "ground_truth", "target", "gt_visibility"):
                if isinstance(gt_visibility, torch.Tensor):
                    focus_visibility = gt_visibility
            elif vis_source in ("base", "base_visibility", "pred", "model"):
                if isinstance(base_visibility, torch.Tensor) and base_visibility.shape == gt_visibility.shape:
                    focus_visibility = base_visibility
            elif vis_source in ("auto", "any", "fallback"):
                if isinstance(base_visibility, torch.Tensor) and base_visibility.shape == gt_visibility.shape:
                    focus_visibility = base_visibility
                elif isinstance(gt_visibility, torch.Tensor):
                    focus_visibility = gt_visibility
            else:
                if isinstance(base_visibility, torch.Tensor) and base_visibility.shape == gt_visibility.shape:
                    focus_visibility = base_visibility
                elif isinstance(gt_visibility, torch.Tensor):
                    focus_visibility = gt_visibility

            if focus_visibility is None:
                return zero, empty_mask

            query_t = query_points[..., 0].round().long().clamp(0, gt_visibility.shape[-1] - 1)
            if query_t.device != pred_tracks.device:
                query_t = query_t.to(pred_tracks.device)
            base_vis = focus_visibility.to(device=pred_tracks.device, dtype=pred_tracks.dtype)
            focus_mask = _compute_reappearance_mask_for_loss(
                base_vis,
                query_t,
                min_occlusion_len=self.focus_sup_min_occlusion_len,
                frames_after=self.focus_sup_frames_after,
                vis_threshold=self.focus_sup_vis_threshold,
            )

        if focus_mask is None:
            return zero, empty_mask

        if gt_visibility.dtype != torch.bool:
            vis_mask = gt_visibility > 0.5
        else:
            vis_mask = gt_visibility
        vis_mask = vis_mask.to(pred_tracks.device)
        active_mask = focus_mask & vis_mask

        if (
            self.focus_sup_base_error_threshold > 0
            and isinstance(base_tracks, torch.Tensor)
            and base_tracks.shape == gt_tracks.shape
        ):
            base = base_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype)
            gt = gt_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype)
            base_err = torch.norm(base - gt, dim=-1)
            active_mask = active_mask & (base_err > float(self.focus_sup_base_error_threshold))

        active_mask_f = active_mask.float()
        min_points = max(int(self.focus_sup_min_points), 0)
        if min_points > 0 and active_mask_f.sum() < float(max(min_points, 1)):
            return zero, active_mask_f

        position_error = self._compute_position_error(pred_tracks, gt_tracks)
        loss = (position_error * active_mask_f.unsqueeze(-1)).sum() / (active_mask_f.sum() + 1e-6)
        return loss, active_mask_f

    def _no_harm_vs_base_position_loss(
        self,
        pred_tracks: torch.Tensor,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        base_tracks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Hinge penalty: only penalize the refiner when it is *worse than base*.

        loss = mean( relu(err_pred - err_base + margin) ) over visible points
        where err_* is L2 distance in normalized coordinates.
        """
        if not self.pos_noharm_enabled or self.pos_noharm_weight <= 0:
            return torch.tensor(0.0, device=pred_tracks.device)
        if base_tracks is None or not isinstance(base_tracks, torch.Tensor):
            return torch.tensor(0.0, device=pred_tracks.device)
        if base_tracks.shape != gt_tracks.shape:
            return torch.tensor(0.0, device=pred_tracks.device)

        base = base_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype)
        gt = gt_tracks.to(device=pred_tracks.device, dtype=pred_tracks.dtype)

        err_pred = torch.norm(pred_tracks - gt, dim=-1)  # (B,N,T)
        err_base = torch.norm(base - gt, dim=-1)  # (B,N,T)
        if self.pos_noharm_detach_base:
            err_base = err_base.detach()

        margin = float(self.pos_noharm_margin)
        hinge = F.relu(err_pred - err_base + margin)

        if gt_visibility.dtype != torch.bool:
            vis = gt_visibility > 0.5
        else:
            vis = gt_visibility
        vis_f = vis.float()
        if self.pos_noharm_base_error_threshold > 0:
            sel = err_base > float(self.pos_noharm_base_error_threshold)
            sel_f = sel.float()
            masked = vis_f * sel_f
            min_points = max(int(self.pos_noharm_min_points), 0)
            if min_points <= 0 or masked.sum() >= float(max(min_points, 1)):
                vis_f = masked
        denom = vis_f.sum().clamp(min=1.0)
        return (hinge * vis_f).sum() / denom

    def _temporal_consistency_loss(
        self,
        pred_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
    ) -> torch.Tensor:
        if self.temporal_consistency_weight <= 0:
            return torch.tensor(0.0, device=pred_tracks.device)
        B, N, T, _ = pred_tracks.shape
        order = self.temporal_consistency_order
        if order == 1:
            if T < 2:
                return torch.tensor(0.0, device=pred_tracks.device)
            diff = pred_tracks[:, :, 1:, :] - pred_tracks[:, :, :-1, :]
            smoothness = torch.norm(diff, dim=-1)
            valid = gt_visibility[:, :, 1:] * gt_visibility[:, :, :-1]
        elif order == 2:
            if T < 3:
                return torch.tensor(0.0, device=pred_tracks.device)
            diff = pred_tracks[:, :, 2:, :] - 2 * pred_tracks[:, :, 1:-1, :] + pred_tracks[:, :, :-2, :]
            smoothness = torch.norm(diff, dim=-1)
            valid = gt_visibility[:, :, 2:] * gt_visibility[:, :, 1:-1] * gt_visibility[:, :, :-2]
        else:
            raise ValueError(f"Unsupported temporal consistency order: {order}")

        valid = valid.float()
        if valid.sum() > 0:
            loss = (smoothness * valid).sum() / valid.sum()
        else:
            # 无可见帧时不施加一致性约束
            loss = torch.tensor(0.0, device=pred_tracks.device)
        return loss * self.temporal_consistency_weight

    def _base_visibility_consistency_loss(
        self,
        pred_visibility: torch.Tensor,
        base_visibility: torch.Tensor,
        gt_visibility: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_visibility is None or self.base_visibility_weight <= 0:
            return torch.tensor(0.0, device=pred_visibility.device)
        if not isinstance(base_visibility, torch.Tensor):
            base_visibility = torch.as_tensor(base_visibility, device=pred_visibility.device)
        if base_visibility.device != pred_visibility.device:
            base_visibility = base_visibility.to(pred_visibility.device)
        if base_visibility.dtype == torch.bool:
            base_visibility = base_visibility.float()
        else:
            base_visibility = base_visibility.float()

        if base_visibility.dim() == 2:
            base_visibility = base_visibility.unsqueeze(0)
        if base_visibility.shape != pred_visibility.shape and base_visibility.dim() == 3:
            # Handle (B,T,N) -> (B,N,T)
            if (
                base_visibility.shape[0] == pred_visibility.shape[0]
                and base_visibility.shape[1] == pred_visibility.shape[2]
                and base_visibility.shape[2] == pred_visibility.shape[1]
            ):
                base_visibility = base_visibility.permute(0, 2, 1).contiguous()

        if base_visibility.shape != pred_visibility.shape:
            if not self._warned_base_vis_shape:
                logger.warning(
                    "base_visibility_consistency skipped due to shape mismatch: "
                    f"base_visibility={tuple(base_visibility.shape)} vs pred_visibility={tuple(pred_visibility.shape)}"
                )
                self._warned_base_vis_shape = True
            return torch.tensor(0.0, device=pred_visibility.device)

        valid = None
        if mask is not None:
            valid = mask
        if self.base_visibility_mask in ("gt_visible", "visible"):
            if gt_visibility is not None:
                gt = gt_visibility
                if gt.dtype != torch.bool:
                    gt = gt > 0.5
                valid = gt if valid is None else (valid & gt)
        elif self.base_visibility_mask not in ("all", "none", ""):
            if not self._warned_base_vis_shape:
                logger.warning(
                    f"Unknown loss.base_visibility_consistency.mask={self.base_visibility_mask!r}; "
                    "supported: all | gt_visible."
                )

        if self.base_visibility_type in ("bce", "binary_cross_entropy"):
            pred = pred_visibility.clamp(1e-6, 1 - 1e-6)
            loss = _safe_bce_prob(pred, base_visibility, reduction="none")
        elif self.base_visibility_type in ("mse", "l2"):
            loss = (pred_visibility - base_visibility) ** 2
        else:
            raise ValueError(f"Unknown base visibility loss type: {self.base_visibility_type}")

        if valid is not None:
            valid_f = valid.float()
            denom = valid_f.sum().clamp(min=1.0)
            return (loss * valid_f).sum() / denom
        return loss.mean()

    def _base_track_consistency_loss(
        self,
        pred_tracks: torch.Tensor,
        base_tracks: torch.Tensor,
        gt_tracks: Optional[torch.Tensor] = None,
        gt_visibility: Optional[torch.Tensor] = None,
        base_visibility: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_tracks is None or self.base_track_weight <= 0:
            return torch.tensor(0.0, device=pred_tracks.device)
        if not isinstance(base_tracks, torch.Tensor):
            base_tracks = torch.as_tensor(base_tracks, device=pred_tracks.device)
        if base_tracks.device != pred_tracks.device:
            base_tracks = base_tracks.to(pred_tracks.device)
        base_tracks = base_tracks.to(dtype=pred_tracks.dtype)

        # Normalize possible shapes to (B,N,T,2).
        if base_tracks.dim() == 3 and base_tracks.shape[-1] == 2:
            base_tracks = base_tracks.unsqueeze(0)
        if base_tracks.dim() == 4 and base_tracks.shape[-1] == 2:
            # Handle (B,T,N,2) -> (B,N,T,2).
            if (
                base_tracks.shape[0] == pred_tracks.shape[0]
                and base_tracks.shape[1] == pred_tracks.shape[2]
                and base_tracks.shape[2] == pred_tracks.shape[1]
            ):
                base_tracks = base_tracks.permute(0, 2, 1, 3).contiguous()

        if base_tracks.shape != pred_tracks.shape:
            if not self._warned_base_track_shape:
                logger.warning(
                    "base_track_consistency skipped due to shape mismatch: "
                    f"base_tracks={tuple(base_tracks.shape)} vs pred_tracks={tuple(pred_tracks.shape)}"
                )
                self._warned_base_track_shape = True
            return torch.tensor(0.0, device=pred_tracks.device)

        weight_map = None
        if self.base_track_adaptive_enabled and gt_tracks is not None:
            gt = gt_tracks
            if not isinstance(gt, torch.Tensor):
                gt = torch.as_tensor(gt, device=pred_tracks.device)
            if gt.device != pred_tracks.device:
                gt = gt.to(pred_tracks.device)
            gt = gt.to(dtype=pred_tracks.dtype)
            if gt.shape == base_tracks.shape:
                # Error-aware trust region:
                # - base close to GT -> strong constraint to not drift
                # - base far from GT -> relax constraint, allow correction
                base_err = torch.norm((base_tracks - gt), dim=-1)  # (B,N,T)
                tau = max(float(self.base_track_adaptive_tau), 1.0e-6)
                power = max(float(self.base_track_adaptive_power), 0.0)
                weight_map = 1.0 / (1.0 + (base_err / tau).pow(power))
                min_w = float(self.base_track_adaptive_min_weight)
                if min_w > 0:
                    weight_map = torch.clamp(weight_map, min=min_w)
                # Safety: keep occluded points tightly anchored to base (they have no position supervision).
                if gt_visibility is not None:
                    vis = gt_visibility
                    if not isinstance(vis, torch.Tensor):
                        vis = torch.as_tensor(vis, device=pred_tracks.device)
                    if vis.device != pred_tracks.device:
                        vis = vis.to(pred_tracks.device)
                    if vis.dtype != torch.bool:
                        vis = vis > 0.5
                    if vis.shape == weight_map.shape:
                        weight_map = torch.where(vis, weight_map, torch.ones_like(weight_map))

        valid = None
        if self.base_track_mask in ("gt_visible", "visible", "gt"):
            if gt_visibility is not None:
                gt = gt_visibility
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, device=pred_tracks.device)
                if gt.device != pred_tracks.device:
                    gt = gt.to(pred_tracks.device)
                if gt.dtype != torch.bool:
                    gt = gt > 0.5
                valid = gt
        elif self.base_track_mask in ("base_visible", "base", "base_vis"):
            if base_visibility is not None:
                bv = base_visibility
                if not isinstance(bv, torch.Tensor):
                    bv = torch.as_tensor(bv, device=pred_tracks.device)
                if bv.device != pred_tracks.device:
                    bv = bv.to(pred_tracks.device)
                if bv.dtype != torch.bool:
                    bv = bv > 0.5
                valid = bv
        elif self.base_track_mask in ("both_visible", "both", "gt_and_base"):
            gt = None
            bv = None
            if gt_visibility is not None:
                gt = gt_visibility
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, device=pred_tracks.device)
                if gt.device != pred_tracks.device:
                    gt = gt.to(pred_tracks.device)
                if gt.dtype != torch.bool:
                    gt = gt > 0.5
            if base_visibility is not None:
                bv = base_visibility
                if not isinstance(bv, torch.Tensor):
                    bv = torch.as_tensor(bv, device=pred_tracks.device)
                if bv.device != pred_tracks.device:
                    bv = bv.to(pred_tracks.device)
                if bv.dtype != torch.bool:
                    bv = bv > 0.5
            if gt is not None and bv is not None:
                valid = gt & bv
            elif gt is not None:
                valid = gt
            elif bv is not None:
                valid = bv
        elif self.base_track_mask in ("all", "none", ""):
            valid = None
        else:
            if gt_visibility is not None:
                gt = gt_visibility
                if not isinstance(gt, torch.Tensor):
                    gt = torch.as_tensor(gt, device=pred_tracks.device)
                if gt.device != pred_tracks.device:
                    gt = gt.to(pred_tracks.device)
                if gt.dtype != torch.bool:
                    gt = gt > 0.5
                valid = gt

        if mask is not None:
            m = mask
            if not isinstance(m, torch.Tensor):
                m = torch.as_tensor(m, device=pred_tracks.device)
            if m.device != pred_tracks.device:
                m = m.to(pred_tracks.device)
            if m.dtype != torch.bool:
                m = m > 0.5
            valid = m if valid is None else (valid & m)

        diff = pred_tracks - base_tracks
        if self.base_track_type in ("l1", "abs"):
            error = torch.abs(diff)
        elif self.base_track_type in ("l2", "mse"):
            error = diff ** 2
        elif self.base_track_type in ("smooth_l1", "huber"):
            error = F.smooth_l1_loss(
                pred_tracks, base_tracks, reduction="none", beta=self.base_track_smooth_l1_beta
            )
        else:
            raise ValueError(f"Unknown base track loss type: {self.base_track_type}")

        if weight_map is not None:
            error = error * weight_map.unsqueeze(-1)

        if valid is not None:
            valid_f = valid.unsqueeze(-1).float()
            denom = valid_f
            if weight_map is not None:
                denom = denom * weight_map.unsqueeze(-1)
            denom = denom.sum().clamp(min=1.0)
            return (error * valid_f).sum() / denom
        return error.mean()

    def _multi_iteration_loss(
        self,
        iter_tracks,
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        base_tracks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not iter_tracks or len(iter_tracks) < 2:
            return torch.tensor(0.0, device=gt_tracks.device)
        # 仅对中间迭代进行监督，避免重复惩罚最终预测
        tracks_list = iter_tracks[:-1]
        num_iters = len(tracks_list)
        weights = [self.multi_iter_gamma ** (num_iters - 1 - i) for i in range(num_iters)]
        weight_sum = sum(weights) if weights else 1.0
        weights = [w / weight_sum for w in weights]

        total = 0.0
        for pred_tracks, weight in zip(tracks_list, weights):
            total = total + weight * self._compute_position_loss(
                pred_tracks, gt_tracks, gt_visibility, base_tracks=base_tracks
            )
        return total

    def _correlation_supervision_loss(
        self,
        corr_logits: Optional[torch.Tensor],
        corr_center_tracks_bt: Optional[torch.Tensor],
        gt_tracks: torch.Tensor,
        gt_visibility: torch.Tensor,
        corr_feature_hw: Optional[torch.Tensor] = None,
        corr_window_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Supervise local-correlation logits with a classification target.

        The target is the integer offset (dy, dx) from the correlation-window center
        to the ground-truth track location, expressed in *feature-map pixels*.

        Args:
            corr_logits: (B, T, N, K) logits over window offsets.
            corr_center_tracks_bt: (B, T, N, 2) normalized [y,x] used as correlation center.
            gt_tracks: (B, N, T, 2) normalized [y,x] ground truth.
            gt_visibility: (B, N, T) bool visibility mask (already includes any extra masks).
            corr_feature_hw: tensor-like (2,) [H_feat, W_feat].
            corr_window_size: int window size (odd).
        """
        if not self.corr_sup_enabled or self.corr_sup_weight <= 0:
            return torch.tensor(0.0, device=gt_tracks.device)
        if corr_logits is None or corr_center_tracks_bt is None or corr_feature_hw is None:
            return torch.tensor(0.0, device=gt_tracks.device)
        if not isinstance(corr_logits, torch.Tensor) or not isinstance(corr_center_tracks_bt, torch.Tensor):
            return torch.tensor(0.0, device=gt_tracks.device)
        if corr_logits.dim() != 4:
            return torch.tensor(0.0, device=gt_tracks.device)

        B, T, N, K = corr_logits.shape
        if corr_center_tracks_bt.shape != (B, T, N, 2):
            return torch.tensor(0.0, device=corr_logits.device)
        if gt_tracks.shape != (B, N, T, 2):
            return torch.tensor(0.0, device=corr_logits.device)

        window_size = None
        if corr_window_size is not None:
            try:
                window_size = int(corr_window_size)
            except (TypeError, ValueError):
                window_size = None
        if window_size is None:
            # Infer from K.
            root = int(round(float(K) ** 0.5))
            if root * root != K:
                return torch.tensor(0.0, device=corr_logits.device)
            window_size = root
        if window_size < 1 or window_size % 2 != 1:
            return torch.tensor(0.0, device=corr_logits.device)
        if window_size * window_size != K:
            return torch.tensor(0.0, device=corr_logits.device)

        if not isinstance(corr_feature_hw, torch.Tensor):
            corr_feature_hw = torch.as_tensor(corr_feature_hw, device=corr_logits.device)
        if corr_feature_hw.numel() != 2:
            return torch.tensor(0.0, device=corr_logits.device)
        H_feat = int(corr_feature_hw.flatten()[0].item())
        W_feat = int(corr_feature_hw.flatten()[1].item())
        if H_feat < 1 or W_feat < 1:
            return torch.tensor(0.0, device=corr_logits.device)

        radius = window_size // 2

        gt_bt = gt_tracks.permute(0, 2, 1, 3).to(device=corr_logits.device, dtype=corr_center_tracks_bt.dtype)
        center_bt = corr_center_tracks_bt.detach().to(device=corr_logits.device, dtype=corr_center_tracks_bt.dtype)

        denom = torch.tensor([float(H_feat), float(W_feat)], device=corr_logits.device, dtype=gt_bt.dtype).view(
            1, 1, 1, 2
        )
        delta_feat = (gt_bt - center_bt) * denom  # (B,T,N,2) in feature pixels
        dy = torch.round(delta_feat[..., 0]).to(dtype=torch.int64)
        dx = torch.round(delta_feat[..., 1]).to(dtype=torch.int64)

        within = (dy.abs() <= radius) & (dx.abs() <= radius)
        if self.corr_sup_mask in ("gt_visible", "visible", "gtvis", "gt_vis"):
            vis = gt_visibility
            if not isinstance(vis, torch.Tensor):
                vis = torch.as_tensor(vis, device=corr_logits.device)
            if vis.dtype != torch.bool:
                vis = vis > 0.5
            if vis.shape == (B, N, T):
                vis_bt = vis.permute(0, 2, 1)
                within = within & vis_bt

        if within.sum() == 0:
            return torch.tensor(0.0, device=corr_logits.device)

        target = (dy + radius) * window_size + (dx + radius)  # (B,T,N)
        target = target.clamp(min=0, max=K - 1)

        logits_flat = corr_logits.reshape(B * T * N, K).float()
        target_flat = target.reshape(B * T * N).long()
        valid_flat = within.reshape(B * T * N)

        loss = F.cross_entropy(logits_flat[valid_flat], target_flat[valid_flat], reduction="mean")
        return loss

    def forward(
        self,
        pred_tracks,
        gt_tracks,
        pred_visibility,
        gt_visibility,
        freq_info=None,
        semantic_feat=None,
        iter_tracks=None,
        confidence=None,
        visibility_mask=None,
        base_visibility=None,
        base_tracks=None,
        pre_accept_tracks=None,
        relocal_acceptor=None,
        relocal_accept_mask=None,
        pre_gate_tracks=None,
        policy_gate=None,
        framewise_init_tracks=None,
        query_points=None,
        relocal_mask=None,
        corr_logits=None,
        corr_center_tracks_bt=None,
        corr_feature_hw=None,
        corr_window_size=None,
    ):
        """
        Args:
            pred_tracks: (B, N, T, 2) 预测轨迹
            gt_tracks: (B, N, T, 2) 真实轨迹
            pred_visibility: (B, N, T) 预测可见性
            gt_visibility: (B, N, T) 真实可见性
            freq_info: 频率分解信息（用于正交性损失）
            semantic_feat: (B, T, N, C) 语义特征（用于语义一致性损失）
            iter_tracks: 多迭代预测列表，每项为 (B, N, T, 2)
            confidence: (B, N, T) 置信度（用于置信度加权损失）
            visibility_mask: (B, N, T) 可选掩码，用于忽略不可靠点
            base_visibility: (B, N, T) 可选的base tracker可见性，用于一致性正则（Route A）
        """
        losses = {}

        # 可选掩码：用于忽略不可靠/不标注点
        mask = None
        visibility_device = gt_visibility.device if isinstance(gt_visibility, torch.Tensor) else pred_tracks.device
        if visibility_mask is not None:
            mask = visibility_mask
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, device=visibility_device)
            if mask.dtype != torch.bool:
                mask = mask > 0.5
            if mask.device != visibility_device:
                mask = mask.to(visibility_device)

        effective_visibility = gt_visibility
        if not isinstance(effective_visibility, torch.Tensor):
            effective_visibility = torch.as_tensor(effective_visibility, device=visibility_device)
        if effective_visibility.device != visibility_device:
            effective_visibility = effective_visibility.to(visibility_device)
        if effective_visibility.dtype != torch.bool:
            effective_visibility = effective_visibility > 0.5
        if mask is not None:
            effective_visibility = effective_visibility & mask
        
        # 1. 位置损失 (仅在可见点计算)
        pred_tracks_for_loss = pred_tracks
        if (
            self.policy_gate_enabled
            and self.policy_gate_use_pre
            and isinstance(pre_gate_tracks, torch.Tensor)
            and pre_gate_tracks.shape == pred_tracks.shape
        ):
            pred_tracks_for_loss = pre_gate_tracks

        position_loss = self._compute_position_loss(
            pred_tracks_for_loss, gt_tracks, effective_visibility, base_tracks=base_tracks
        )
        # Optional delta supervision (Route A): directly supervise (pred - base) to match (gt - base).
        if self.delta_sup_enabled and self.delta_sup_weight > 0 and base_tracks is not None:
            delta_loss = self._delta_supervision_loss(
                pred_tracks_for_loss,
                gt_tracks,
                effective_visibility,
                base_tracks=base_tracks,
            )
            losses["position_delta_supervision"] = delta_loss * float(self.delta_sup_weight)
        # Optional "do no harm vs base" hinge penalty (Route A).
        if self.pos_noharm_enabled and self.pos_noharm_weight > 0 and base_tracks is not None:
            noharm = self._no_harm_vs_base_position_loss(
                pred_tracks_for_loss,
                gt_tracks,
                effective_visibility,
                base_tracks=base_tracks,
            )
            losses["position_no_harm_vs_base"] = noharm * float(self.pos_noharm_weight)
        if self.focus_sup_enabled and self.focus_sup_weight > 0:
            focused_loss, focused_mask_f = self._focused_position_loss(
                pred_tracks_for_loss,
                gt_tracks,
                effective_visibility,
                base_tracks=base_tracks,
                base_visibility=base_visibility,
                relocal_mask=relocal_mask,
                query_points=query_points,
            )
            losses["position_focus_supervision"] = focused_loss * float(self.focus_sup_weight)
            losses["position_focus_mask_rate"] = focused_mask_f.mean()
            losses["position_focus_num_points"] = focused_mask_f.sum().detach()

        if (
            self.relocal_acceptor_enabled
            and self.relocal_acceptor_weight > 0
            and isinstance(relocal_acceptor, torch.Tensor)
            and isinstance(relocal_accept_mask, torch.Tensor)
            and isinstance(pre_accept_tracks, torch.Tensor)
            and isinstance(base_tracks, torch.Tensor)
            and pre_accept_tracks.shape == gt_tracks.shape
            and base_tracks.shape == gt_tracks.shape
            and relocal_acceptor.shape == gt_tracks[..., 0].shape
            and relocal_accept_mask.shape == gt_tracks[..., 0].shape
        ):
            err_pred = torch.norm(pre_accept_tracks - gt_tracks.to(pre_accept_tracks.device), dim=-1)
            err_base = torch.norm(base_tracks - gt_tracks.to(base_tracks.device), dim=-1)
            if self.relocal_acceptor_detach_pred:
                err_pred = err_pred.detach()
            if self.relocal_acceptor_detach_base:
                err_base = err_base.detach()

            margin = float(self.relocal_acceptor_margin)
            target = (err_pred + margin < err_base).float()

            accept_mask = effective_visibility & relocal_accept_mask.bool()
            if self.relocal_acceptor_base_error_threshold > 0:
                accept_mask = accept_mask & (err_base > float(self.relocal_acceptor_base_error_threshold))
            accept_mask_f = accept_mask.float()

            min_points = max(int(self.relocal_acceptor_min_points), 0)
            if min_points <= 0 or accept_mask_f.sum() >= float(max(min_points, 1)):
                accept_pred = relocal_acceptor.to(device=accept_mask_f.device, dtype=accept_mask_f.dtype).clamp(1.0e-4, 1.0 - 1.0e-4)
                if self.relocal_acceptor_loss_type in ("mse", "l2"):
                    accept_loss = (accept_pred - target.to(accept_pred.device)) ** 2
                else:
                    target_f = target.to(device=accept_pred.device, dtype=torch.float32)
                    accept_pred_f = accept_pred.to(dtype=torch.float32)
                    if accept_pred_f.is_cuda:
                        with torch.cuda.amp.autocast(enabled=False):
                            accept_loss = F.binary_cross_entropy(accept_pred_f, target_f, reduction="none")
                    else:
                        accept_loss = F.binary_cross_entropy(accept_pred_f, target_f, reduction="none")
                accept_loss = (accept_loss * accept_mask_f).sum() / (accept_mask_f.sum() + 1e-6)
                losses["relocal_acceptor"] = accept_loss * float(self.relocal_acceptor_weight)
                losses["relocal_acceptor_mean"] = (accept_pred * accept_mask_f).sum() / (accept_mask_f.sum() + 1e-6)
                pred_pos = (accept_pred >= float(self.relocal_acceptor_eval_threshold)) & accept_mask
                target_pos = (target > 0.5) & accept_mask
                valid_count = accept_mask_f.sum()
                pred_count = pred_pos.float().sum()
                target_count = target_pos.float().sum()
                tp = (pred_pos & target_pos).float().sum()
                tn = ((~pred_pos) & (~target_pos) & accept_mask).float().sum()
                losses["relocal_acceptor_precision"] = tp / (pred_count + 1.0e-6)
                losses["relocal_acceptor_recall"] = tp / (target_count + 1.0e-6)
                losses["relocal_acceptor_coverage"] = pred_count / (valid_count + 1.0e-6)
                losses["relocal_acceptor_target_rate"] = target_count / (valid_count + 1.0e-6)
                losses["relocal_acceptor_accuracy"] = (tp + tn) / (valid_count + 1.0e-6)
                losses["relocal_acceptor_num_points"] = valid_count.detach()

        if (
            self.policy_gate_enabled
            and self.policy_gate_weight > 0
            and isinstance(policy_gate, torch.Tensor)
            and isinstance(pre_gate_tracks, torch.Tensor)
            and isinstance(base_tracks, torch.Tensor)
            and pre_gate_tracks.shape == gt_tracks.shape
            and base_tracks.shape == gt_tracks.shape
            and policy_gate.shape == gt_tracks[..., 0].shape
        ):
            err_pred = torch.norm(pre_gate_tracks - gt_tracks.to(pre_gate_tracks.device), dim=-1)  # (B,N,T)
            err_base = torch.norm(base_tracks - gt_tracks.to(base_tracks.device), dim=-1)  # (B,N,T)
            if self.policy_gate_detach_pred:
                err_pred = err_pred.detach()
            if self.policy_gate_detach_base:
                err_base = err_base.detach()

            margin = float(self.policy_gate_margin)
            target = (err_pred + margin < err_base).float()

            gate_mask = effective_visibility
            if self.policy_gate_base_error_threshold > 0:
                gate_mask = gate_mask & (err_base > float(self.policy_gate_base_error_threshold))
            gate_mask_f = gate_mask.float()

            min_points = max(int(self.policy_gate_min_points), 0)
            if min_points <= 0 or gate_mask_f.sum() >= float(max(min_points, 1)):
                gate_pred = policy_gate.to(device=gate_mask_f.device, dtype=gate_mask_f.dtype).clamp(1.0e-4, 1.0 - 1.0e-4)
                if self.policy_gate_loss_type in ("mse", "l2"):
                    gate_loss = (gate_pred - target.to(gate_pred.device)) ** 2
                else:
                    # BCE is unsafe under autocast; compute in full precision.
                    target_f = target.to(device=gate_pred.device, dtype=torch.float32)
                    gate_pred_f = gate_pred.to(dtype=torch.float32)
                    if gate_pred_f.is_cuda:
                        with torch.cuda.amp.autocast(enabled=False):
                            gate_loss = F.binary_cross_entropy(gate_pred_f, target_f, reduction="none")
                    else:
                        gate_loss = F.binary_cross_entropy(gate_pred_f, target_f, reduction="none")
                gate_loss = (gate_loss * gate_mask_f).sum() / (gate_mask_f.sum() + 1e-6)
                losses["policy_gate"] = gate_loss * float(self.policy_gate_weight)
                losses["policy_gate_mean"] = (gate_pred * gate_mask_f).sum() / (gate_mask_f.sum() + 1e-6)

        if (
            self.frame_sup_enabled
            and self.frame_sup_weight > 0
            and isinstance(framewise_init_tracks, torch.Tensor)
            and framewise_init_tracks.shape == pred_tracks_for_loss.shape
        ):
            init_tracks = framewise_init_tracks.to(device=pred_tracks_for_loss.device, dtype=pred_tracks_for_loss.dtype)
            if self.frame_sup_detach_init:
                init_tracks = init_tracks.detach()

            frame_mask = effective_visibility
            if (
                self.frame_sup_base_error_threshold > 0
                and isinstance(base_tracks, torch.Tensor)
                and base_tracks.shape == gt_tracks.shape
            ):
                base_err = torch.norm(
                    base_tracks.to(device=gt_tracks.device, dtype=gt_tracks.dtype) - gt_tracks,
                    dim=-1,
                )
                frame_mask = frame_mask & (base_err > float(self.frame_sup_base_error_threshold))
            frame_mask_f = frame_mask.float()
            min_points = max(int(self.frame_sup_min_points), 0)
            if min_points <= 0 or frame_mask_f.sum() >= float(max(min_points, 1)):
                err = self._compute_framewise_error(pred_tracks_for_loss, init_tracks)
                loss_fw = (err * frame_mask_f.unsqueeze(-1)).sum() / (frame_mask_f.sum() + 1e-6)
                losses["framewise_init_supervision"] = loss_fw * float(self.frame_sup_weight)
        # 置信度加权位置损失（可选）
        if self.confidence_weighted_enabled:
            conf = confidence if confidence is not None else pred_visibility
            if conf.shape != pred_visibility.shape:
                if conf.dim() == 4 and conf.shape[-1] == 1:
                    conf = conf.squeeze(-1)
                if conf.dim() == 3 and conf.shape[1] == pred_visibility.shape[2] and conf.shape[2] == pred_visibility.shape[1]:
                    conf = conf.permute(0, 2, 1)
                if conf.dim() == 2 and pred_visibility.dim() == 3:
                    B, N, T = pred_visibility.shape
                    if conf.shape[0] == B * N and conf.shape[1] == T:
                        conf = conf.view(B, N, T)
                    elif conf.shape[0] == B and conf.shape[1] == N:
                        conf = conf.unsqueeze(-1).expand_as(pred_visibility)
            if conf.shape != pred_visibility.shape:
                logger.warning(
                    f"Confidence shape mismatch: {conf.shape} vs {pred_visibility.shape}, "
                    "falling back to pred_visibility."
                )
                conf = pred_visibility
            conf = conf.clamp(min=self.min_confidence)
            position_error = self._compute_position_error(pred_tracks_for_loss, gt_tracks).sum(dim=-1)
            valid_mask = effective_visibility.float()
            if valid_mask.sum() > 0:
                weighted_position_loss = (position_error * conf * valid_mask).sum() / valid_mask.sum()
            else:
                weighted_position_loss = torch.tensor(0.0, device=pred_tracks.device)
            position_loss = weighted_position_loss

            confidence_reg = -torch.log(conf + 1e-8) * valid_mask
            if valid_mask.sum() > 0:
                confidence_reg = confidence_reg.sum() / valid_mask.sum()
            else:
                confidence_reg = torch.tensor(0.0, device=pred_tracks.device)
            losses['confidence_reg'] = confidence_reg * self.confidence_weight
            losses['mean_confidence'] = conf.mean()

        losses['position'] = position_loss * self.position_weight
        
        # 2. 遮挡损失 (BCE)
        if self.occlusion_type == 'bce':
            occlusion_target = gt_visibility.float()
            occlusion_loss = _safe_bce_prob(
                pred_visibility,
                occlusion_target,
                reduction='none'
            )
            if mask is not None:
                occlusion_loss = (occlusion_loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
            else:
                occlusion_loss = occlusion_loss.mean()
        elif self.occlusion_type == 'focal':
            pred_prob = pred_visibility.clamp(1e-6, 1 - 1e-6)
            gt = gt_visibility.float()
            # 正确的focal loss: alpha用于正样本，1-alpha用于负样本
            pt = pred_prob * gt + (1 - pred_prob) * (1 - gt)
            alpha_factor = self.focal_alpha * gt + (1 - self.focal_alpha) * (1 - gt)
            focal = -alpha_factor * (1 - pt) ** self.focal_gamma * torch.log(pt)
            if mask is not None:
                focal = (focal * mask.float()).sum() / (mask.float().sum() + 1e-6)
            occlusion_loss = focal.mean() if mask is None else focal
        else:
            raise ValueError(f"Unknown occlusion loss type: {self.occlusion_type}")
        losses['occlusion'] = occlusion_loss * self.occlusion_weight

        # 2.1 Base visibility consistency loss (optional)
        base_vis_weight = self._get_base_visibility_weight()
        if base_visibility is not None and base_vis_weight > 0:
            base_vis_loss = self._base_visibility_consistency_loss(
                pred_visibility,
                base_visibility,
                gt_visibility=effective_visibility,
                mask=mask,
            )
            losses["base_visibility"] = base_vis_loss * base_vis_weight
        
        # 3. 频率正交性损失 (如果有)
        # 2.2 Base track consistency loss (optional)
        base_track_weight = self._get_base_track_weight()
        if base_tracks is not None and base_track_weight > 0:
            base_track_loss = self._base_track_consistency_loss(
                pred_tracks,
                base_tracks,
                gt_tracks=gt_tracks,
                gt_visibility=effective_visibility,
                base_visibility=base_visibility,
                mask=mask,
            )
            losses["base_tracks"] = base_track_loss * base_track_weight

        if freq_info is not None:
            if 'ortho_loss' in freq_info:
                losses['freq_ortho'] = freq_info['ortho_loss'] * self.freq_ortho_weight
            if self.freq_reconstruction_weight > 0 and 'reconstruction_loss' in freq_info:
                losses['freq_reconstruction'] = (
                    freq_info['reconstruction_loss'] * self.freq_reconstruction_weight
                )
            if self.freq_separation_weight > 0 and 'freq_separation_loss' in freq_info:
                losses['freq_separation'] = (
                    freq_info['freq_separation_loss'] * self.freq_separation_weight
                )

        # 4. 语义一致性损失
        if semantic_feat is not None and self.semantic_weight > 0:
            semantic_vis = effective_visibility
            if semantic_vis.dtype != torch.bool:
                semantic_vis = semantic_vis > 0.5
            semantic_loss = self._semantic_consistency_loss(
                semantic_feat, pred_visibility, semantic_vis
            )
            losses['semantic'] = semantic_loss * self.semantic_weight

        # 4.1 Correlation supervision loss (optional, Route A).
        if self.corr_sup_enabled and self.corr_sup_weight > 0:
            corr_sup_loss = self._correlation_supervision_loss(
                corr_logits=corr_logits,
                corr_center_tracks_bt=corr_center_tracks_bt,
                gt_tracks=gt_tracks,
                gt_visibility=effective_visibility,
                corr_feature_hw=corr_feature_hw,
                corr_window_size=corr_window_size,
            )
            losses["corr_supervision"] = corr_sup_loss * float(self.corr_sup_weight)
        
        # 5. 时序平滑损失（鼓励轨迹平滑）
        if self.temporal_smooth_weight > 0:
            # 计算轨迹的二阶差分（加速度）
            # pred_tracks: (B, N, T, 2)
            if pred_tracks.shape[2] < 3:
                smooth_loss = torch.tensor(0.0, device=pred_tracks.device)
            else:
                velocity = pred_tracks[:, :, 1:, :] - pred_tracks[:, :, :-1, :]  # (B, N, T-1, 2)
                acceleration = velocity[:, :, 1:, :] - velocity[:, :, :-1, :]  # (B, N, T-2, 2)
                # 只在全部3帧都可见的区域计算（加速度涉及 t, t+1, t+2）
                vis_mask = (
                    effective_visibility[:, :, :-2]
                    & effective_visibility[:, :, 1:-1]
                    & effective_visibility[:, :, 2:]
                ).unsqueeze(-1).float()
                if vis_mask.sum() > 0:
                    smooth_loss = (acceleration ** 2 * vis_mask).sum() / vis_mask.sum()
                else:
                    smooth_loss = torch.tensor(0.0, device=pred_tracks.device)
            losses['temporal_smooth'] = smooth_loss * self.temporal_smooth_weight

        # 6. 时序一致性正则化（顶会策略）
        if self.temporal_consistency_enabled:
            consistency_loss = self._temporal_consistency_loss(pred_tracks, effective_visibility)
            losses['temporal_consistency'] = consistency_loss

        # 7. 多迭代监督
        if self.multi_iter_enabled:
            if iter_tracks is None:
                if not self._warned_multi_iter:
                    logger.warning("multi_iteration enabled but no iter_tracks provided; skipping multi-iter loss.")
                    self._warned_multi_iter = True
            else:
                losses['multi_iteration'] = self._multi_iteration_loss(
                    iter_tracks, gt_tracks, effective_visibility, base_tracks=base_tracks
                )
        
        # 总损失
        total_loss = sum(
            value for key, value in losses.items()
            if key not in {
                'mean_confidence',
                'policy_gate_mean',
                'position_focus_mask_rate',
                'position_focus_num_points',
                'relocal_acceptor_mean',
                'relocal_acceptor_precision',
                'relocal_acceptor_recall',
                'relocal_acceptor_coverage',
                'relocal_acceptor_target_rate',
                'relocal_acceptor_accuracy',
                'relocal_acceptor_num_points',
            }
        )
        losses['total'] = total_loss
        
        return losses


def _compute_two_view_consistency_loss(
    model: nn.Module,
    video: torch.Tensor,
    query_points: torch.Tensor,
    gt_visibility: Optional[torch.Tensor],
    cfg: object,
    epoch: int,
    base_tracks: Optional[torch.Tensor] = None,
    base_visibility: Optional[torch.Tensor] = None,
    total_epochs: Optional[int] = None,
    ema=None,
    global_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    optimizer_step: Optional[int] = None,
    total_optimizer_steps: Optional[int] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Two-view spatial consistency regularization.

    Creates two randomly-warped views of {video, query_points} (and optionally cached base tracks),
    runs a teacher on view-1 and a student on view-2, inverse-maps predictions back to the original
    coordinate system, and penalizes track disagreement.

    This is config-gated and designed to be ablation-friendly (default-off).
    """

    def _cfg_get(local_cfg: Optional[object], key: str, default=None):
        if local_cfg is None:
            return default
        if isinstance(local_cfg, dict):
            return local_cfg.get(key, default)
        return getattr(local_cfg, key, default)

    def _scheduled_weight(base_weight: float, sched_cfg: Optional[object], key: str) -> float:
        if sched_cfg is None:
            return float(base_weight)
        enabled_sched = _cfg_get(sched_cfg, "enabled", True)
        if enabled_sched is not None and not bool(enabled_sched):
            return float(base_weight)
        sched_type = str(_cfg_get(sched_cfg, "type", "none")).lower().strip()
        if sched_type in ("none", "", "off", "disabled", "false"):
            return float(base_weight)

        start_weight = float(_cfg_get(sched_cfg, "start_weight", 0.0) or 0.0)
        end_weight = float(_cfg_get(sched_cfg, "end_weight", base_weight) or base_weight)

        unit = str(_cfg_get(sched_cfg, "unit", _cfg_get(sched_cfg, "by", "epoch"))).lower().strip()
        if unit in ("step", "steps", "iter", "iters", "iteration", "global_step", "global_steps", "batch", "batches"):
            current = int(global_step) if isinstance(global_step, int) else int(epoch)
            start = int(_cfg_get(sched_cfg, "start_step", 0) or 0)
            end = _cfg_get(sched_cfg, "end_step", None)
            if end is None:
                end = start
            try:
                end = int(end)
            except (TypeError, ValueError):
                end = start
            if end < 0 and isinstance(total_steps, int) and total_steps > 0:
                end = int(total_steps) + int(end)
        elif unit in ("optimizer_step", "optim_step", "opt_step", "update", "updates", "optimizer", "optim"):
            current = int(optimizer_step) if isinstance(optimizer_step, int) else (int(global_step) if isinstance(global_step, int) else int(epoch))
            start = int(_cfg_get(sched_cfg, "start_step", 0) or 0)
            end = _cfg_get(sched_cfg, "end_step", None)
            if end is None:
                end = start
            try:
                end = int(end)
            except (TypeError, ValueError):
                end = start
            if end < 0 and isinstance(total_optimizer_steps, int) and total_optimizer_steps > 0:
                end = int(total_optimizer_steps) + int(end)
        else:
            current = int(epoch)
            start = int(_cfg_get(sched_cfg, "start_epoch", 0) or 0)
            end = _cfg_get(sched_cfg, "end_epoch", None)
            if end is None:
                end = start
            try:
                end = int(end)
            except (TypeError, ValueError):
                end = start
            if end < 0 and isinstance(total_epochs, int) and total_epochs > 0:
                end = int(total_epochs) + int(end)

        if end < start:
            end = start

        if int(current) <= int(start):
            return start_weight
        if int(current) >= int(end):
            return end_weight

        denom = max(int(end) - int(start), 1)
        t = float(int(current) - int(start)) / float(denom)

        if sched_type in ("linear", "lerp", "ramp"):
            blend = t
        elif sched_type in ("cosine", "cos"):
            blend = 0.5 - 0.5 * math.cos(math.pi * t)
        elif sched_type in ("step", "constant"):
            blend = 1.0
        else:
            logger.warning(f"Unknown two_view_consistency schedule type {sched_type!r} for {key}; using linear.")
            blend = t
        return float(start_weight + (end_weight - start_weight) * float(blend))

    enabled = bool(_cfg_get(cfg, "enabled", False))
    base_track_weight = float(_cfg_get(cfg, "weight", 0.0) or 0.0)
    base_vis_weight = float(_cfg_get(cfg, "visibility_weight", 0.0) or 0.0)

    track_weight = _scheduled_weight(base_track_weight, _cfg_get(cfg, "schedule", None), key="tracks")
    vis_weight = _scheduled_weight(base_vis_weight, _cfg_get(cfg, "visibility_schedule", None), key="visibility")

    if (not enabled) or (track_weight <= 0 and vis_weight <= 0):
        return None

    warmup_epochs = int(_cfg_get(cfg, "warmup_epochs", 0) or 0)
    if int(epoch) < warmup_epochs:
        return None

    apply_prob = float(_cfg_get(cfg, "apply_prob", 1.0) or 0.0)
    apply_prob = 0.0 if apply_prob < 0 else (1.0 if apply_prob > 1 else apply_prob)
    if apply_prob < 1.0 and random.random() > apply_prob:
        return None

    require_base = bool(_cfg_get(cfg, "require_base_tracks", True))
    if require_base and (
        base_tracks is None
        or not isinstance(base_tracks, torch.Tensor)
        or base_visibility is None
        or not isinstance(base_visibility, torch.Tensor)
    ):
        return None

    # Teacher selection (EMA is recommended for stability).
    teacher_mode = str(_cfg_get(cfg, "teacher", "ema")).lower().strip()
    stop_grad = bool(_cfg_get(cfg, "stop_grad", True))
    teacher_disable_autocast = bool(_cfg_get(cfg, "teacher_disable_autocast", False))
    student_model = unwrap_model(model)
    teacher_model: nn.Module
    if teacher_mode in ("ema", "mean_teacher") and ema is not None and hasattr(ema, "ema_model"):
        teacher_model = ema.ema_model
    else:
        teacher_model = student_model

    # Keep teacher targets stable (disable dropout) when the teacher is a separate EMA model.
    teacher_requires_eval = teacher_model is not student_model
    teacher_was_training = bool(getattr(teacher_model, "training", False)) if teacher_requires_eval else False
    if teacher_requires_eval:
        teacher_model.eval()

    from utils.affine_augmentation import normalize_affine_cfg, sample_affine_matrices, transform_points_yx, warp_video_affine

    aug_cfg = normalize_affine_cfg(_cfg_get(cfg, "augmentation", None))
    align_corners = bool(aug_cfg.get("align_corners", False))
    padding_mode = str(aug_cfg.get("padding_mode", "zeros"))
    sample_mode = str(aug_cfg.get("mode", "bilinear"))

    B, T, _, _, _ = video.shape
    # Optional: subsample query points for the two-view regularizer to reduce compute.
    max_points = _cfg_get(cfg, "max_points", None)
    if max_points is not None:
        try:
            max_points = int(max_points)
        except (TypeError, ValueError):
            max_points = None
    if (
        isinstance(max_points, int)
        and max_points > 0
        and isinstance(query_points, torch.Tensor)
        and query_points.shape[1] > max_points
    ):
        n_total = int(query_points.shape[1])
        idx = torch.randperm(n_total, device=query_points.device)[:max_points]
        query_points = query_points[:, idx]
        if isinstance(gt_visibility, torch.Tensor) and gt_visibility.shape[1] == n_total:
            gt_visibility = gt_visibility[:, idx]
        if isinstance(base_tracks, torch.Tensor) and base_tracks.shape[1] == n_total:
            base_tracks = base_tracks[:, idx]
        if isinstance(base_visibility, torch.Tensor) and base_visibility.shape[1] == n_total:
            base_visibility = base_visibility[:, idx]

    max_resample = int(_cfg_get(cfg, "max_resample", 0) or 0)
    min_in_bounds_fraction = float(_cfg_get(cfg, "min_in_bounds_fraction", 0.0) or 0.0)
    min_in_bounds_fraction = (
        0.0 if min_in_bounds_fraction < 0.0 else (1.0 if min_in_bounds_fraction > 1.0 else min_in_bounds_fraction)
    )

    def _sample_theta_for_queries():
        tries = 1 if max_resample <= 0 else max_resample
        for _ in range(max(1, tries)):
            theta_o2a, theta_a2o = sample_affine_matrices(
                batch_size=B, device=video.device, dtype=torch.float32, cfg=aug_cfg
            )
            q_yx, q_in = transform_points_yx(query_points[..., 1:3], theta_o2a)
            if min_in_bounds_fraction <= 0.0:
                return theta_o2a, theta_a2o, q_yx, q_in
            frac = q_in.float().mean(dim=1)
            if bool((frac >= min_in_bounds_fraction).all().item()):
                return theta_o2a, theta_a2o, q_yx, q_in
        return None

    res1 = _sample_theta_for_queries()
    res2 = _sample_theta_for_queries()
    if res1 is None or res2 is None:
        return None
    theta1_o2a, theta1_a2o, q1_yx, q1_in = res1
    theta2_o2a, theta2_a2o, q2_yx, q2_in = res2

    video1 = warp_video_affine(
        video,
        theta1_a2o,
        mode=sample_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    video2 = warp_video_affine(
        video,
        theta2_a2o,
        mode=sample_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    q1 = query_points.clone()
    q2 = query_points.clone()
    q1[..., 1:3] = q1_yx
    q2[..., 1:3] = q2_yx

    temporal_cfg = _cfg_get(cfg, "temporal", None)
    temporal_enabled_cfg = bool(_cfg_get(temporal_cfg, "enabled", False))
    reverse_prob = float(_cfg_get(temporal_cfg, "reverse_prob", 0.5) or 0.0)
    roll_prob = float(_cfg_get(temporal_cfg, "roll_prob", 0.5) or 0.0)
    reverse_prob = 0.0 if reverse_prob < 0.0 else (1.0 if reverse_prob > 1.0 else reverse_prob)
    roll_prob = 0.0 if roll_prob < 0.0 else (1.0 if roll_prob > 1.0 else roll_prob)
    roll_max = _cfg_get(temporal_cfg, "roll_max", None)
    if roll_max is None:
        roll_max = T - 1
    try:
        roll_max = int(roll_max)
    except (TypeError, ValueError):
        roll_max = T - 1
    roll_max = max(0, min(int(roll_max), T - 1))

    def _sample_temporal_params() -> Dict[str, object]:
        if not temporal_enabled_cfg or T <= 1:
            return {"reverse": False, "offset": 0}
        do_reverse = random.random() < reverse_prob
        offset = 0
        if roll_max > 0 and random.random() < roll_prob:
            offset = int(random.randint(0, roll_max))
        return {"reverse": bool(do_reverse), "offset": int(offset)}

    meta1 = None
    meta2 = None
    temporal_params1: Dict[str, object] = {"reverse": False, "offset": 0}
    temporal_params2: Dict[str, object] = {"reverse": False, "offset": 0}

    temporal_enabled = temporal_enabled_cfg and isinstance(base_tracks, torch.Tensor) and isinstance(base_visibility, torch.Tensor)

    def _apply_temporal_view(
        vid: torch.Tensor,
        q: torch.Tensor,
        bt: torch.Tensor,
        bv: torch.Tensor,
        bt_in: torch.Tensor,
        params: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reverse = bool(params.get("reverse", False))
        offset = int(params.get("offset", 0) or 0) % T

        if reverse:
            vid = torch.flip(vid, dims=[1])
            bt = torch.flip(bt, dims=[2])
            bv = torch.flip(bv, dims=[2])
            bt_in = torch.flip(bt_in, dims=[2])
            q = q.clone()
            q[:, :, 0] = (T - 1) - q[:, :, 0]

        if offset != 0:
            vid = torch.roll(vid, shifts=-offset, dims=1)
            bt = torch.roll(bt, shifts=-offset, dims=2)
            bv = torch.roll(bv, shifts=-offset, dims=2)
            bt_in = torch.roll(bt_in, shifts=-offset, dims=2)
            q = q.clone()
            q[:, :, 0] = (q[:, :, 0] - float(offset)) % float(T)

        # Keep (y,x) at query frame consistent with (transformed) base tracks.
        q = q.clone()
        q_t = q[:, :, 0].round().long().clamp(0, T - 1)  # (B,N)
        b_idx = torch.arange(B, device=vid.device).view(B, 1).expand_as(q_t)
        n_idx = torch.arange(q_t.shape[1], device=vid.device).view(1, -1).expand_as(q_t)
        q_yx = bt[b_idx, n_idx, q_t]  # (B,N,2)
        q[:, :, 1:3] = q_yx
        q_in_new = bt_in[b_idx, n_idx, q_t]  # (B,N)
        return vid, q, bt, bv, bt_in, q_in_new

    if base_tracks is not None and base_visibility is not None:
        bt1, bt1_in = transform_points_yx(base_tracks, theta1_o2a)
        bt2, bt2_in = transform_points_yx(base_tracks, theta2_o2a)
        bv = base_visibility
        if bv.dtype == torch.bool:
            bv = bv.float()
        bv1 = bv.to(device=video.device, dtype=video.dtype) * bt1_in.to(device=video.device, dtype=video.dtype)
        bv2 = bv.to(device=video.device, dtype=video.dtype) * bt2_in.to(device=video.device, dtype=video.dtype)

        if temporal_enabled:
            temporal_params1 = _sample_temporal_params()
            temporal_params2 = _sample_temporal_params()
            video1, q1, bt1, bv1, bt1_in, q1_in = _apply_temporal_view(
                video1, q1, bt1, bv1, bt1_in, temporal_params1
            )
            video2, q2, bt2, bv2, bt2_in, q2_in = _apply_temporal_view(
                video2, q2, bt2, bv2, bt2_in, temporal_params2
            )

        meta1 = {"base_tracks": bt1, "base_visibility": bv1}
        meta2 = {"base_tracks": bt2, "base_visibility": bv2}

    def _forward_tracks_vis(m: nn.Module, vid: torch.Tensor, q: torch.Tensor, meta: Optional[Dict]):
        out = None
        try:
            out = m(vid, q, meta=meta, return_info=False)
        except TypeError:
            try:
                out = m(vid, q, return_info=False)
            except TypeError:
                out = m(vid, q)
        if isinstance(out, (list, tuple)):
            if len(out) >= 2:
                return out[0], out[1]
        raise RuntimeError("Model forward did not return (tracks, visibility).")

    # Teacher on view-1 (optionally stop-grad); student on view-2.
    try:
        teacher_amp_ctx = autocast(enabled=False) if teacher_disable_autocast else nullcontext()
        with teacher_amp_ctx:
            if stop_grad:
                with torch.no_grad():
                    t_tracks, t_vis = _forward_tracks_vis(teacher_model, video1, q1, meta1)
            else:
                t_tracks, t_vis = _forward_tracks_vis(teacher_model, video1, q1, meta1)
    finally:
        if teacher_requires_eval and teacher_was_training:
            teacher_model.train()

    s_tracks, s_vis = _forward_tracks_vis(model, video2, q2, meta2)

    # Map predictions back to original coordinates.
    t_tracks_o, t_in_o = transform_points_yx(t_tracks, theta1_a2o)
    s_tracks_o, s_in_o = transform_points_yx(s_tracks, theta2_a2o)

    def _invert_temporal_outputs(
        tracks: torch.Tensor,
        vis: torch.Tensor,
        in_bounds: torch.Tensor,
        params: Dict[str, object],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tracks.shape[:3] != vis.shape or tracks.shape[:3] != in_bounds.shape:
            return tracks, vis, in_bounds
        offset = int(params.get("offset", 0) or 0) % T
        reverse = bool(params.get("reverse", False))
        if offset != 0:
            tracks = torch.roll(tracks, shifts=offset, dims=2)
            vis = torch.roll(vis, shifts=offset, dims=2)
            in_bounds = torch.roll(in_bounds, shifts=offset, dims=2)
        if reverse:
            tracks = torch.flip(tracks, dims=[2])
            vis = torch.flip(vis, dims=[2])
            in_bounds = torch.flip(in_bounds, dims=[2])
        return tracks, vis, in_bounds

    if temporal_enabled:
        t_tracks_o, t_vis, t_in_o = _invert_temporal_outputs(t_tracks_o, t_vis, t_in_o, temporal_params1)
        s_tracks_o, s_vis, s_in_o = _invert_temporal_outputs(s_tracks_o, s_vis, s_in_o, temporal_params2)

    # Masking strategy.
    mask_mode = str(_cfg_get(cfg, "mask", "gt_visible")).lower().strip()
    pred_thr = float(_cfg_get(cfg, "pred_vis_threshold", 0.5) or 0.5)
    pred_mask = (t_vis.detach() > pred_thr) & (s_vis.detach() > pred_thr)
    if gt_visibility is not None:
        gt_mask = gt_visibility
        if gt_mask.dtype != torch.bool:
            gt_mask = gt_mask > 0.5
    else:
        gt_mask = None

    if mask_mode in ("all", "", "none"):
        mask = torch.ones_like(pred_mask, dtype=torch.bool)
    elif mask_mode in ("pred", "pred_visible", "pred_vis"):
        mask = pred_mask
    elif mask_mode in ("both", "gt_and_pred", "gt+pred", "gt_pred"):
        mask = pred_mask if gt_mask is None else (pred_mask & gt_mask)
    else:  # default: gt_visible
        mask = pred_mask if gt_mask is None else gt_mask

    use_in_bounds = bool(_cfg_get(cfg, "use_in_bounds", True))
    if use_in_bounds:
        mask = mask & t_in_o & s_in_o
        mask = mask & q1_in.unsqueeze(-1) & q2_in.unsqueeze(-1)

    exclude_query = bool(_cfg_get(cfg, "exclude_query_frame", False))
    if exclude_query and query_points.shape[2] >= 1:
        qt = query_points[:, :, 0].round().long().clamp(0, T - 1)
        b_idx = torch.arange(B, device=video.device).view(B, 1).expand_as(qt)
        n_idx = torch.arange(qt.shape[1], device=video.device).view(1, -1).expand_as(qt)
        mask = mask.clone()
        mask[b_idx, n_idx, qt] = False

    # Convert boolean mask to (optionally) soft weights for smoother training.
    mask_weighting = str(_cfg_get(cfg, "mask_weighting", "binary")).lower().strip()
    if mask_weighting in ("soft", "prob", "confidence", "soft_pred", "soft_pred_visible"):
        if mask_mode in ("all", "", "none"):
            weights = torch.ones_like(t_vis, dtype=torch.float32)
        elif mask_mode in ("pred", "pred_visible", "pred_vis"):
            weights = (t_vis.detach().float() * s_vis.detach().float())
        elif mask_mode in ("both", "gt_and_pred", "gt+pred", "gt_pred"):
            weights = (t_vis.detach().float() * s_vis.detach().float())
            if gt_mask is not None:
                weights = weights * gt_mask.float()
        else:  # default: gt_visible
            weights = (t_vis.detach().float() * s_vis.detach().float()) if gt_mask is None else gt_mask.float()
        if use_in_bounds:
            weights = weights * t_in_o.float() * s_in_o.float()
            weights = weights * q1_in.unsqueeze(-1).float() * q2_in.unsqueeze(-1).float()
        if exclude_query:
            weights = weights.clone()
            weights[b_idx, n_idx, qt] = 0.0
    else:
        weights = mask.float()

    denom = weights.sum().clamp(min=1.0)
    out: Dict[str, torch.Tensor] = {}

    total = torch.tensor(0.0, device=video.device, dtype=torch.float32)

    if track_weight > 0:
        loss_type = str(_cfg_get(cfg, "type", "l2")).lower().strip()
        if loss_type in ("l1", "abs"):
            per = (s_tracks_o - t_tracks_o).abs().sum(dim=-1)
        elif loss_type in ("smooth_l1", "huber"):
            beta = float(_cfg_get(cfg, "smooth_l1_beta", 1.0) or 1.0)
            per = F.smooth_l1_loss(s_tracks_o, t_tracks_o, reduction="none", beta=beta).sum(dim=-1)
        else:  # l2 / mse
            per = (s_tracks_o - t_tracks_o).pow(2).sum(dim=-1)
        tracks_loss = (per * weights).sum() / denom
        tracks_loss = tracks_loss * float(track_weight)
        out["tracks"] = tracks_loss
        total = total + tracks_loss

    if vis_weight > 0:
        vis_type = str(_cfg_get(cfg, "visibility_type", "mse")).lower().strip()
        t_target = t_vis.detach() if stop_grad else t_vis
        if vis_type in ("bce", "binary_cross_entropy"):
            eps = float(_cfg_get(cfg, "visibility_eps", 1e-4) or 1e-4)
            eps = 1e-6 if eps <= 0 else (0.1 if eps > 0.1 else eps)
            vis_loss_per = _safe_bce_prob(
                s_vis.clamp(eps, 1.0 - eps),
                t_target.clamp(eps, 1.0 - eps),
                reduction="none",
            )
        else:  # mse/l2
            vis_loss_per = (s_vis - t_target).pow(2)
        vis_loss = (vis_loss_per * weights).sum() / denom
        vis_loss = vis_loss * float(vis_weight)
        out["visibility"] = vis_loss
        total = total + vis_loss

    out["total"] = total
    return out


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    scaler,
    epoch,
    config,
    exp_logger,
    device,
    is_main_process: bool = True,
    use_prefetch: bool = True,
    ema=None,
    pseudo_trainer=None,
    pseudo_loader=None,
    pseudo_cfg=None,
):
    """训练一个epoch"""
    model.train()

    # Provide epoch context for any scheduled loss weights.
    if hasattr(criterion, "set_epoch"):
        try:
            total_epochs = int(getattr(getattr(config, "training", None), "epochs", 0) or 0)
        except (TypeError, ValueError):
            total_epochs = 0
        try:
            criterion.set_epoch(epoch, total_epochs=total_epochs if total_epochs > 0 else None)
        except TypeError:
            try:
                criterion.set_epoch(epoch)
            except Exception:
                pass
     
    total_loss = 0
    num_batches = len(train_loader)
    if num_batches == 0:
        if is_main_process:
            logger.warning("Empty training dataloader; skipping epoch.")
        return 0.0, {
            "position_focus_supervision_loss_mean": 0.0,
            "position_focus_mask_rate_mean": 0.0,
            "position_focus_active_batch_rate": 0.0,
            "position_focus_nonzero_batch_rate": 0.0,
            "position_focus_num_points_mean": 0.0,
        }
    accum_steps = max(1, int(config.training.gradient.accumulation_steps))
    log_every = int(getattr(getattr(config, 'logging', None), 'print_every', 100) or 100)
    if log_every <= 0:
        log_every = None
    else:
        log_every = max(1, log_every)
    wandb_cfg = getattr(getattr(config, 'logging', None), 'wandb', None)
    if wandb_cfg is not None and getattr(wandb_cfg, 'log_freq', None) is not None:
        try:
            log_freq = int(wandb_cfg.log_freq)
        except (TypeError, ValueError):
            log_freq = None
        if log_freq is not None:
            if log_freq > 0:
                log_every = max(1, log_freq)
            else:
                log_every = None
    
    # 使用数据预取器加速数据加载
    if use_prefetch and device.type == 'cuda':
        try:
            from utils.data_prefetcher import DataPrefetcher
            data_iter = DataPrefetcher(train_loader, device)
        except ImportError:
            data_iter = train_loader
            use_prefetch = False
    else:
        data_iter = train_loader
        use_prefetch = False
    
    pbar = tqdm(data_iter, desc=f'Epoch {epoch}', disable=not is_main_process, total=num_batches)
    optimizer.zero_grad(set_to_none=True)
    
    multi_iter_cfg = getattr(getattr(config, 'loss', None), 'multi_iteration', None)
    use_iter_tracks = bool(multi_iter_cfg is not None and getattr(multi_iter_cfg, 'enabled', False))

    two_view_cfg = getattr(getattr(config, 'loss', None), 'two_view_consistency', None)
    two_view_enabled = bool(
        two_view_cfg is not None
        and getattr(two_view_cfg, 'enabled', False)
    )
    warned_two_view_skip = False
    warned_two_view_oom = False

    focus_supervision_sum = 0.0
    focus_mask_rate_sum = 0.0
    focus_num_points_sum = 0.0
    focus_active_batches = 0.0
    focus_nonzero_loss_batches = 0.0

    total_training_epochs = int(getattr(getattr(config, "training", None), "epochs", 0) or 0)
    total_steps = (total_training_epochs * num_batches) if total_training_epochs > 0 else None
    optimizer_steps_per_epoch = int(math.ceil(float(num_batches) / float(accum_steps))) if accum_steps > 0 else num_batches
    total_optimizer_steps = (
        total_training_epochs * optimizer_steps_per_epoch if total_training_epochs > 0 else None
    )

    pseudo_enabled = pseudo_trainer is not None and pseudo_loader is not None
    pseudo_warmup = int(getattr(pseudo_cfg, 'warmup_epochs', 0) or 0) if pseudo_cfg is not None else 0
    pseudo_iter = None
    if pseudo_enabled and epoch >= pseudo_warmup:
        pseudo_iter = iter(pseudo_loader)
    elif pseudo_enabled and is_main_process:
        logger.info(f"Pseudo-labeling warmup: epoch {epoch} < {pseudo_warmup}, skipping pseudo steps.")
    sched_type = _get_scheduler_type(config)
    pseudo_onecycle_block = sched_type == 'onecyclelr'
    if pseudo_enabled and pseudo_onecycle_block and is_main_process:
        logger.warning("Pseudo-labeling skipped with OneCycleLR (step schedule mismatch).")
    if pseudo_enabled and accum_steps > 1 and is_main_process:
        logger.warning("Pseudo-labeling skipped with gradient accumulation (accumulation_steps > 1).")

    for batch_idx, batch in enumerate(pbar):
        global_step = epoch * num_batches + batch_idx
        optimizer_step = epoch * optimizer_steps_per_epoch + (batch_idx // accum_steps)
        # 如果使用预取，数据已经在GPU上；否则需要移动
        if use_prefetch:
            video = batch['video']
            query_points = batch['query_points']
            target_points = batch['target_points']
            occluded = batch['occluded']
        else:
            video = batch['video'].to(device)
            query_points = batch['query_points'].to(device)
            target_points = batch['target_points'].to(device)
            occluded = batch['occluded'].to(device)

        if occluded.dtype != torch.bool:
            occluded = occluded > 0.5
        
        # 混合精度训练
        amp_cfg = getattr(getattr(config, 'training', None), 'amp', None)
        amp_enabled = bool(amp_cfg is not None and getattr(amp_cfg, 'enabled', False) and device.type == 'cuda')
        amp_dtype = _get_amp_dtype(config)
        # 梯度更新
        should_step = (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches
        sync_context = nullcontext()
        if hasattr(model, 'no_sync') and not should_step and dist.is_available() and dist.is_initialized():
            sync_context = model.no_sync()

        with sync_context:
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                # 前向传播
                extra = None
                try:
                    meta = (
                        {
                            'video_name': batch.get('video_name', None),
                            'base_tracks': batch.get('base_tracks', None),
                            'base_visibility': batch.get('base_visibility', None),
                        }
                        if isinstance(batch, dict)
                        else None
                    )
                    outputs = model(
                        video,
                        query_points,
                        meta=meta,
                        return_info=True,
                        return_iter_tracks=use_iter_tracks,
                    )
                except TypeError:
                    try:
                        outputs = model(video, query_points, return_info=True, return_iter_tracks=use_iter_tracks)
                    except TypeError:
                        outputs = model(video, query_points)
                if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
                    pred_tracks, pred_visibility, extra = outputs
                    freq_info = extra.get('freq_info') if isinstance(extra, dict) else None
                else:
                    pred_tracks, pred_visibility = outputs
                    freq_info = None
                
                # 计算损失
                iter_tracks = extra.get('iter_tracks') if isinstance(extra, dict) else None
                confidence = extra.get('confidence') if isinstance(extra, dict) else None
                corr_logits = extra.get('corr_logits') if isinstance(extra, dict) else None
                corr_center_tracks_bt = extra.get('corr_center_tracks_bt') if isinstance(extra, dict) else None
                corr_feature_hw = extra.get('corr_feature_hw') if isinstance(extra, dict) else None
                corr_window_size = extra.get('corr_window_size') if isinstance(extra, dict) else None
                base_visibility_for_loss = None
                base_tracks_for_loss = None
                if isinstance(batch, dict):
                    base_visibility_for_loss = batch.get('base_visibility', None)
                    base_tracks_for_loss = batch.get('base_tracks', None)
                if base_visibility_for_loss is None and isinstance(extra, dict):
                    # Route A: base visibility comes from the base tracker inside the model.
                    base_visibility_for_loss = extra.get('base_visibility', None)
                if base_tracks_for_loss is None and isinstance(extra, dict):
                    # Route A: base tracks comes from the base tracker inside the model.
                    base_tracks_for_loss = extra.get('base_tracks', None)
                pre_gate_tracks = extra.get('pre_gate_tracks', None) if isinstance(extra, dict) else None
                pre_accept_tracks = extra.get('pre_accept_tracks', None) if isinstance(extra, dict) else None
                relocal_acceptor = (
                    extra.get('verifier_scores', extra.get('relocal_acceptor', None))
                    if isinstance(extra, dict) else None
                )
                relocal_accept_mask = (
                    extra.get('verifier_mask', extra.get('relocal_accept_mask', None))
                    if isinstance(extra, dict) else None
                )
                relocal_mask_nt = (
                    extra.get('relocal_mask', extra.get('relocalization_mask', None))
                    if isinstance(extra, dict) else None
                )
                relocal_conf_nt = (
                    extra.get('relocal_conf', extra.get('relocalization_conf', None))
                    if isinstance(extra, dict) else None
                )
                policy_gate = extra.get('policy_gate', None) if isinstance(extra, dict) else None
                framewise_init_tracks = extra.get('framewise_init_tracks', None) if isinstance(extra, dict) else None
                if (
                    (not use_prefetch)
                    and isinstance(base_visibility_for_loss, torch.Tensor)
                    and base_visibility_for_loss.device != device
                ):
                    base_visibility_for_loss = base_visibility_for_loss.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(base_tracks_for_loss, torch.Tensor)
                    and base_tracks_for_loss.device != device
                ):
                    base_tracks_for_loss = base_tracks_for_loss.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(pre_accept_tracks, torch.Tensor)
                    and pre_accept_tracks.device != device
                ):
                    pre_accept_tracks = pre_accept_tracks.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(relocal_acceptor, torch.Tensor)
                    and relocal_acceptor.device != device
                ):
                    relocal_acceptor = relocal_acceptor.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(relocal_accept_mask, torch.Tensor)
                    and relocal_accept_mask.device != device
                ):
                    relocal_accept_mask = relocal_accept_mask.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(pre_gate_tracks, torch.Tensor)
                    and pre_gate_tracks.device != device
                ):
                    pre_gate_tracks = pre_gate_tracks.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(policy_gate, torch.Tensor)
                    and policy_gate.device != device
                ):
                    policy_gate = policy_gate.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(framewise_init_tracks, torch.Tensor)
                    and framewise_init_tracks.device != device
                ):
                    framewise_init_tracks = framewise_init_tracks.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(relocal_mask_nt, torch.Tensor)
                    and relocal_mask_nt.device != device
                ):
                    relocal_mask_nt = relocal_mask_nt.to(device)
                if (
                    (not use_prefetch)
                    and isinstance(relocal_conf_nt, torch.Tensor)
                    and relocal_conf_nt.device != device
                ):
                    relocal_conf_nt = relocal_conf_nt.to(device)

                loss_visibility_mask = None
                exclude_query_frame_in_loss = bool(
                    getattr(getattr(config, 'loss', None), 'exclude_query_frame', False)
                )
                if (
                    exclude_query_frame_in_loss
                    and isinstance(query_points, torch.Tensor)
                    and isinstance(occluded, torch.Tensor)
                    and occluded.dim() == 3
                    and query_points.dim() >= 3
                    and query_points.shape[-1] >= 1
                ):
                    B_mask, N_mask, T_mask = occluded.shape
                    query_t = query_points[..., 0].round().long().clamp(0, T_mask - 1)
                    if query_t.device != occluded.device:
                        query_t = query_t.to(occluded.device)
                    if query_t.shape[0] == B_mask and query_t.shape[1] == N_mask:
                        loss_visibility_mask = torch.ones(
                            (B_mask, N_mask, T_mask),
                            device=occluded.device,
                            dtype=torch.bool,
                        )
                        batch_indices = torch.arange(B_mask, device=occluded.device).view(B_mask, 1).expand_as(query_t)
                        point_indices = torch.arange(N_mask, device=occluded.device).view(1, N_mask).expand_as(query_t)
                        loss_visibility_mask[batch_indices, point_indices, query_t] = False
                losses = criterion(
                    pred_tracks,
                    target_points,
                    pred_visibility,
                    ~occluded,  # 转换为可见性
                    freq_info=freq_info,
                    semantic_feat=extra.get('semantic_features') if isinstance(extra, dict) else None,
                    iter_tracks=iter_tracks,
                    confidence=confidence,
                    visibility_mask=loss_visibility_mask,
                    base_visibility=base_visibility_for_loss,
                    base_tracks=base_tracks_for_loss,
                    pre_accept_tracks=pre_accept_tracks,
                    relocal_acceptor=relocal_acceptor,
                    relocal_accept_mask=relocal_accept_mask,
                    pre_gate_tracks=pre_gate_tracks,
                    policy_gate=policy_gate,
                    framewise_init_tracks=framewise_init_tracks,
                    query_points=query_points,
                    relocal_mask=relocal_mask_nt,
                    corr_logits=corr_logits,
                    corr_center_tracks_bt=corr_center_tracks_bt,
                    corr_feature_hw=corr_feature_hw,
                    corr_window_size=corr_window_size,
                )

                # 训练统计必须覆盖所有 batch，而不是只覆盖被日志打印的 batch。
                # 否则当 print_every > num_batches 时，epoch 级 focus 指标会被严重低估。
                if 'position_focus_supervision' in losses:
                    focus_supervision_value = float(losses['position_focus_supervision'].detach().item())
                    focus_supervision_sum += focus_supervision_value
                    if focus_supervision_value > 0.0:
                        focus_nonzero_loss_batches += 1.0
                if 'position_focus_mask_rate' in losses:
                    focus_mask_rate = float(losses['position_focus_mask_rate'].detach().item())
                    focus_mask_rate_sum += focus_mask_rate
                    if focus_mask_rate > 0.0:
                        focus_active_batches += 1.0
                if 'position_focus_num_points' in losses:
                    focus_num_points_sum += float(losses['position_focus_num_points'].detach().item())

                if two_view_enabled:
                    tv_out = None
                    try:
                        tv_out = _compute_two_view_consistency_loss(
                            model=model,
                            video=video,
                            query_points=query_points,
                            gt_visibility=(~occluded),
                            cfg=two_view_cfg,
                            epoch=epoch,
                            base_tracks=base_tracks_for_loss,
                            base_visibility=base_visibility_for_loss,
                            total_epochs=total_training_epochs if total_training_epochs > 0 else None,
                            ema=ema,
                            global_step=int(global_step),
                            total_steps=int(total_steps) if isinstance(total_steps, int) else None,
                            optimizer_step=int(optimizer_step),
                            total_optimizer_steps=(
                                int(total_optimizer_steps) if isinstance(total_optimizer_steps, int) else None
                            ),
                        )
                    except RuntimeError as exc:
                        oom_safe = bool(getattr(two_view_cfg, "oom_safe", True))
                        msg = str(exc).lower()
                        if oom_safe and ("out of memory" in msg or "cuda" in msg and "alloc" in msg and "memory" in msg):
                            if device.type == "cuda":
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            if is_main_process and (not warned_two_view_oom):
                                logger.warning(
                                    "two_view_consistency hit OOM; skipping two-view loss for this batch "
                                    "(loss.two_view_consistency.oom_safe=true)."
                                )
                                warned_two_view_oom = True
                            tv_out = None
                        else:
                            raise
                    if tv_out is not None:
                        if isinstance(tv_out, dict):
                            if "tracks" in tv_out:
                                losses["two_view_tracks"] = tv_out["tracks"]
                            if "visibility" in tv_out:
                                losses["two_view_visibility"] = tv_out["visibility"]
                            tv_total = tv_out.get("total", None)
                            if isinstance(tv_total, torch.Tensor):
                                losses["two_view_consistency"] = tv_total
                                losses["total"] = losses["total"] + tv_total
                        else:
                            losses["two_view_consistency"] = tv_out
                            losses["total"] = losses["total"] + tv_out
                    elif (
                        is_main_process
                        and (not warned_two_view_skip)
                        and bool(getattr(two_view_cfg, "require_base_tracks", True))
                        and (base_tracks_for_loss is None or base_visibility_for_loss is None)
                    ):
                        logger.warning(
                            "two_view_consistency enabled but cached base_tracks/base_visibility are missing; skipping."
                        )
                        warned_two_view_skip = True
            effective_accum = accum_steps
            if accum_steps > 1:
                remaining = num_batches % accum_steps
                if remaining != 0 and batch_idx >= num_batches - remaining:
                    effective_accum = remaining
            loss = losses['total'] / effective_accum
            
            # 反向传播
            scaler.scale(loss).backward()
        if should_step:
            # 梯度裁剪
            if config.training.gradient.clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.training.gradient.clip_norm
                )
            
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(unwrap_model(model))
            optimizer.zero_grad(set_to_none=True)
            
            # OneCycleLR在每个optimizer step后更新
            if scheduler is not None and _get_scheduler_type(config) == 'onecyclelr':
                scheduler.step()

            # 伪标签训练（可选）
            if (
                pseudo_iter is not None
                and not pseudo_onecycle_block
                and accum_steps == 1
                and pseudo_enabled
            ):
                try:
                    pseudo_batch = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    pseudo_batch = next(pseudo_iter)
                missing_keys = ('query_points' not in pseudo_batch) or ('video' not in pseudo_batch)
                if dist.is_available() and dist.is_initialized():
                    skip_tensor = torch.tensor(1 if missing_keys else 0, device=device)
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX)
                    missing_keys = bool(skip_tensor.item())
                if missing_keys:
                    if is_main_process:
                        logger.warning("Pseudo batch missing video/query_points; skipping.")
                else:
                    video_u = pseudo_batch['video'].to(device)
                    query_u = pseudo_batch['query_points'].to(device)
                    pseudo_losses = pseudo_trainer.train_step(
                        video_u,
                        query_u,
                        criterion,
                        optimizer,
                        scaler if amp_enabled else None,
                        amp_dtype=amp_dtype if amp_enabled else None,
                    )
                    if ema is not None and pseudo_losses.get('did_step', True):
                        ema.update(unwrap_model(model))
                    if is_main_process and exp_logger is not None and log_every is not None and batch_idx % log_every == 0:
                        pseudo_step = epoch * num_batches + batch_idx
                        exp_logger.log_metrics(pseudo_losses, step=pseudo_step, prefix='pseudo/')
        
        # 记录
        total_loss += losses['total'].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'pos': losses['position'].item(),
            'occ': losses['occlusion'].item(),
        })
        
        # 记录日志
        if is_main_process and exp_logger is not None and log_every is not None and batch_idx % log_every == 0:
            log_metrics = {
                'loss': losses['total'].item(),
                'position_loss': losses['position'].item(),
                'occlusion_loss': losses['occlusion'].item(),
                'lr': optimizer.param_groups[0]['lr'],
            }
            if 'semantic' in losses:
                log_metrics['semantic_loss'] = losses['semantic'].item()
            if 'freq_ortho' in losses:
                log_metrics['freq_ortho_loss'] = losses['freq_ortho'].item()
            if 'freq_reconstruction' in losses:
                log_metrics['freq_reconstruction_loss'] = losses['freq_reconstruction'].item()
            if 'freq_separation' in losses:
                log_metrics['freq_separation_loss'] = losses['freq_separation'].item()
            if 'temporal_smooth' in losses:
                log_metrics['temporal_smooth_loss'] = losses['temporal_smooth'].item()
            if 'temporal_consistency' in losses:
                log_metrics['temporal_consistency_loss'] = losses['temporal_consistency'].item()
            if 'multi_iteration' in losses:
                log_metrics['multi_iteration_loss'] = losses['multi_iteration'].item()
            if 'confidence_reg' in losses:
                log_metrics['confidence_reg_loss'] = losses['confidence_reg'].item()
            if 'base_visibility' in losses:
                log_metrics['base_visibility_loss'] = losses['base_visibility'].item()
            if 'base_tracks' in losses:
                log_metrics['base_tracks_loss'] = losses['base_tracks'].item()
            if 'position_delta_supervision' in losses:
                log_metrics['position_delta_supervision_loss'] = losses['position_delta_supervision'].item()
            if 'position_no_harm_vs_base' in losses:
                log_metrics['position_no_harm_vs_base_loss'] = losses['position_no_harm_vs_base'].item()
            if 'position_focus_supervision' in losses:
                log_metrics['position_focus_supervision_loss'] = losses['position_focus_supervision'].item()
            if 'position_focus_mask_rate' in losses:
                log_metrics['position_focus_mask_rate'] = losses['position_focus_mask_rate'].item()
            if 'position_focus_num_points' in losses:
                log_metrics['position_focus_num_points'] = losses['position_focus_num_points'].item()
            if 'two_view_consistency' in losses:
                log_metrics['two_view_consistency_loss'] = losses['two_view_consistency'].item()
            if 'two_view_tracks' in losses:
                log_metrics['two_view_tracks_loss'] = losses['two_view_tracks'].item()
            if 'two_view_visibility' in losses:
                log_metrics['two_view_visibility_loss'] = losses['two_view_visibility'].item()
            if 'mean_confidence' in losses:
                log_metrics['mean_confidence'] = losses['mean_confidence'].item()
            if isinstance(relocal_mask_nt, torch.Tensor):
                relocal_mask_f = relocal_mask_nt.float()
                log_metrics['relocal_mask_rate'] = relocal_mask_f.mean().item()
                log_metrics['relocal_mask_num_points'] = relocal_mask_f.sum().item()
                if isinstance(relocal_conf_nt, torch.Tensor) and relocal_conf_nt.shape == relocal_mask_nt.shape:
                    active_mask = relocal_mask_nt.bool()
                    if active_mask.any():
                        relocal_conf_f = relocal_conf_nt.float()
                        active_conf = relocal_conf_f[active_mask]
                        log_metrics['relocal_conf_mean'] = active_conf.mean().item()
                        log_metrics['relocal_conf_std'] = (
                            active_conf.std(unbiased=False).item() if active_conf.numel() > 1 else 0.0
                        )
            if 'relocal_acceptor' in losses:
                log_metrics['relocal_acceptor_loss'] = losses['relocal_acceptor'].item()
            if 'relocal_acceptor_mean' in losses:
                log_metrics['relocal_acceptor_mean'] = losses['relocal_acceptor_mean'].item()
            if 'relocal_acceptor_precision' in losses:
                log_metrics['verifier_precision'] = losses['relocal_acceptor_precision'].item()
            if 'relocal_acceptor_recall' in losses:
                log_metrics['verifier_recall'] = losses['relocal_acceptor_recall'].item()
            if 'relocal_acceptor_coverage' in losses:
                log_metrics['verifier_coverage'] = losses['relocal_acceptor_coverage'].item()
            if 'relocal_acceptor_target_rate' in losses:
                log_metrics['verifier_target_rate'] = losses['relocal_acceptor_target_rate'].item()
            if 'relocal_acceptor_accuracy' in losses:
                log_metrics['verifier_accuracy'] = losses['relocal_acceptor_accuracy'].item()
            exp_logger.log_metrics(log_metrics, step=global_step, prefix='train/')
    
    avg_loss = total_loss / num_batches
    focus_stats = {
        "position_focus_supervision_loss_mean": focus_supervision_sum / float(num_batches) if num_batches > 0 else 0.0,
        "position_focus_mask_rate_mean": focus_mask_rate_sum / float(num_batches) if num_batches > 0 else 0.0,
        "position_focus_active_batch_rate": focus_active_batches / float(num_batches) if num_batches > 0 else 0.0,
        "position_focus_nonzero_batch_rate": focus_nonzero_loss_batches / float(num_batches) if num_batches > 0 else 0.0,
        "position_focus_num_points_mean": focus_num_points_sum / float(num_batches) if num_batches > 0 else 0.0,
    }
    if dist.is_available() and dist.is_initialized():
        stat_tensor = torch.tensor(
            [
                float(total_loss),
                float(num_batches),
                float(focus_supervision_sum),
                float(focus_mask_rate_sum),
                float(focus_num_points_sum),
                float(focus_active_batches),
                float(focus_nonzero_loss_batches),
            ],
            device=device,
            dtype=torch.float32,
        )
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
        if stat_tensor[1].item() > 0:
            avg_loss = (stat_tensor[0] / stat_tensor[1]).item()
        else:
            avg_loss = 0.0
        batch_count_total = float(stat_tensor[1].item())
        focus_stats = {
            "position_focus_supervision_loss_mean": (stat_tensor[2] / stat_tensor[1]).item() if batch_count_total > 0 else 0.0,
            "position_focus_mask_rate_mean": (stat_tensor[3] / stat_tensor[1]).item() if batch_count_total > 0 else 0.0,
            "position_focus_active_batch_rate": (stat_tensor[5] / stat_tensor[1]).item() if batch_count_total > 0 else 0.0,
            "position_focus_nonzero_batch_rate": (stat_tensor[6] / stat_tensor[1]).item() if batch_count_total > 0 else 0.0,
            "position_focus_num_points_mean": (stat_tensor[4] / stat_tensor[1]).item() if batch_count_total > 0 else 0.0,
        }
    return avg_loss, focus_stats


@torch.no_grad()
def evaluate(model, val_loader, epoch, config, exp_logger=None, is_main_process: bool = True):
    """评估模型"""
    from datasets.metrics import compute_tapvid_metrics
    
    model.eval()
    device = next(model.parameters()).device
    if val_loader is None:
        logger.warning("No validation samples available; skipping metrics.")
        return {}
    try:
        if len(val_loader) == 0:
            logger.warning("No validation samples available; skipping metrics.")
            return {}
    except TypeError:
        pass
    exclude_query_frame = True
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'exclude_query_frame'):
        exclude_query_frame = bool(config.evaluation.exclude_query_frame)
    metric_resolution_mode = 'original'
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'metric_resolution_mode'):
        raw_mode = getattr(config.evaluation, 'metric_resolution_mode')
        if raw_mode is not None and str(raw_mode).strip():
            metric_resolution_mode = str(raw_mode).strip().lower()
    if metric_resolution_mode not in ('original', 'input'):
        logger.warning(
            f"Unknown evaluation.metric_resolution_mode={metric_resolution_mode}, fallback to 'original'."
        )
        metric_resolution_mode = 'original'
    aux_metric_resolution_mode = None
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'aux_metric_resolution_mode'):
        raw_aux_mode = getattr(config.evaluation, 'aux_metric_resolution_mode')
        if raw_aux_mode is not None and str(raw_aux_mode).strip():
            aux_metric_resolution_mode = str(raw_aux_mode).strip().lower()
    if aux_metric_resolution_mode not in (None, 'original', 'input'):
        logger.warning(
            f"Unknown evaluation.aux_metric_resolution_mode={aux_metric_resolution_mode}, disable auxiliary metrics."
        )
        aux_metric_resolution_mode = None
    if aux_metric_resolution_mode == metric_resolution_mode:
        aux_metric_resolution_mode = None
    aux_metric_fixed_resolution = None
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'aux_metric_fixed_resolution'):
        raw_aux_resolution = getattr(config.evaluation, 'aux_metric_fixed_resolution')
        try:
            if isinstance(raw_aux_resolution, torch.Tensor):
                raw_aux_resolution = raw_aux_resolution.tolist()
            if isinstance(raw_aux_resolution, (list, tuple)) and len(raw_aux_resolution) >= 2:
                aux_metric_fixed_resolution = (int(raw_aux_resolution[0]), int(raw_aux_resolution[1]))
            elif raw_aux_resolution is not None and str(raw_aux_resolution).strip():
                res_int = int(raw_aux_resolution)
                aux_metric_fixed_resolution = (res_int, res_int)
        except Exception:
            logger.warning(
                "Invalid evaluation.aux_metric_fixed_resolution, disable fixed-resolution auxiliary metrics."
            )
            aux_metric_fixed_resolution = None
    query_mode = None
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'query_mode'):
        qm = getattr(config.evaluation, 'query_mode')
        if qm is not None and str(qm).strip():
            query_mode = str(qm).strip()
    compare_base = False
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'compare_base'):
        compare_base = bool(config.evaluation.compare_base)

    # Optional: long-occlusion re-localization subset evaluation.
    # This is meant for analysis/paper tables where we focus on points that are
    # occluded for a long time and then re-appear.
    long_occ_cfg = None
    long_occ_enabled = False
    long_occ_thresholds = []
    if hasattr(config, 'evaluation') and hasattr(config.evaluation, 'long_occlusion_subset'):
        long_occ_cfg = config.evaluation.long_occlusion_subset
        try:
            long_occ_enabled = bool(getattr(long_occ_cfg, 'enabled', False))
        except Exception:
            long_occ_enabled = False
        if long_occ_enabled:
            raw_thresholds = getattr(long_occ_cfg, 'thresholds', [10, 20, 30])
            try:
                if isinstance(raw_thresholds, torch.Tensor):
                    raw_thresholds = raw_thresholds.tolist()
                if raw_thresholds is None:
                    raw_thresholds = [10, 20, 30]
                if isinstance(raw_thresholds, (int, float, str)):
                    raw_thresholds = [raw_thresholds]
                thresholds = []
                for v in list(raw_thresholds):
                    try:
                        thresholds.append(int(v))
                    except Exception:
                        continue
                long_occ_thresholds = sorted({t for t in thresholds if t > 0})
            except Exception:
                long_occ_thresholds = [10, 20, 30]

    def _max_reappearance_occlusion_run(occluded_nt: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
        """
        For each query, compute the longest *continuous* occlusion run AFTER the
        query frame that is followed by at least one visible frame (reappearance).

        Args:
            occluded_nt: (N,T) bool (True=occluded)
            query_t: (N,) long indices in [0, T-1]
        Returns:
            max_run_len: (N,) long
        """
        if occluded_nt.ndim != 2:
            raise ValueError(f"Expected occluded shape (N,T), got {tuple(occluded_nt.shape)}")
        if query_t.ndim != 1:
            raise ValueError(f"Expected query_t shape (N,), got {tuple(query_t.shape)}")
        n_queries, num_frames = occluded_nt.shape
        out = torch.zeros((n_queries,), dtype=torch.long, device=occluded_nt.device)
        for q in range(n_queries):
            start = int(query_t[q].item()) + 1
            if start >= num_frames:
                continue
            mask = occluded_nt[q, start:]
            if mask.numel() == 0:
                continue
            best = 0
            run = 0
            for val in mask.tolist():
                if val:
                    run += 1
                    continue
                # Run ended and we are visible now => reappearance exists.
                if run > best:
                    best = run
                run = 0
            # If `run > 0` at the end, it means occluded until end-of-video (no reappearance).
            out[q] = best
        return out
    
    def _resolve_resolution_from_batch(batch, index, fallback):
        if not isinstance(batch, dict):
            return fallback
        original_size = batch.get('original_size', None)
        if original_size is None:
            return fallback
        try:
            size = None
            if isinstance(original_size, torch.Tensor):
                if original_size.ndim == 2 and original_size.shape[0] > index:
                    size = original_size[index].tolist()
                else:
                    size = original_size.tolist()
            elif isinstance(original_size, (list, tuple)):
                # DataLoader collate can turn per-sample `(H, W)` tuples into
                # `(tensor([H...]), tensor([W...]))`. Handle that explicitly.
                if (
                    len(original_size) == 2
                    and isinstance(original_size[0], torch.Tensor)
                    and isinstance(original_size[1], torch.Tensor)
                ):
                    h_tensor, w_tensor = original_size[0], original_size[1]
                    try:
                        h_val = (
                            h_tensor[index].item()
                            if h_tensor.numel() > index
                            else h_tensor.reshape(-1)[0].item()
                        )
                        w_val = (
                            w_tensor[index].item()
                            if w_tensor.numel() > index
                            else w_tensor.reshape(-1)[0].item()
                        )
                        return (int(h_val), int(w_val))
                    except Exception:
                        pass
                if len(original_size) > 0 and isinstance(original_size[0], (list, tuple, torch.Tensor)):
                    size = original_size[index] if index < len(original_size) else original_size[0]
                    if hasattr(size, 'tolist'):
                        size = size.tolist()
                else:
                    size = original_size
            else:
                size = original_size
            if isinstance(size, (list, tuple)) and len(size) >= 2:
                return (int(size[0]), int(size[1]))
        except Exception:
            return fallback
        return fallback

    def _resolve_metric_resolution(batch, index, fallback, mode, fixed_resolution=None):
        if fixed_resolution is not None:
            return fixed_resolution
        if mode == 'input':
            return fallback
        return _resolve_resolution_from_batch(batch, index, fallback)

    all_metrics = []
    all_aux_metrics = []
    all_aux_base_metrics = []
    all_base_metrics = []
    all_longocc_metrics = {thr: [] for thr in long_occ_thresholds}
    all_longocc_base_metrics = {thr: [] for thr in long_occ_thresholds}
    longocc_num_queries = {thr: 0 for thr in long_occ_thresholds}
    longocc_num_videos = {thr: 0 for thr in long_occ_thresholds}
    
    iterator = tqdm(val_loader, desc='Evaluating', disable=not is_main_process)
    for batch in iterator:
        try:
            if batch is None or not isinstance(batch, dict):
                logger.warning("Empty or invalid validation batch, skipping.")
                continue
            required_keys = ('video', 'query_points', 'target_points', 'occluded')
            missing_keys = [k for k in required_keys if k not in batch]
            if missing_keys:
                logger.warning(f"Validation batch missing keys {missing_keys}, skipping.")
                continue
            video = batch['video'].to(device)
            query_points = batch['query_points'].to(device)
            target_points = batch['target_points'].to(device)
            occluded = batch['occluded'].to(device)
            if occluded.dtype != torch.bool:
                occluded = occluded > 0.5
            if video.ndim == 4:
                video = video.unsqueeze(0)
                if query_points.ndim == 2:
                    query_points = query_points.unsqueeze(0)
                if target_points.ndim == 3:
                    target_points = target_points.unsqueeze(0)
                if occluded.ndim == 2:
                    occluded = occluded.unsqueeze(0)

            # 推理 - 统一处理2或3个返回值
            base_tracks = None
            base_visibility = None
            meta = (
                {
                    'video_name': batch.get('video_name', None),
                    'base_tracks': batch.get('base_tracks', None),
                    'base_visibility': batch.get('base_visibility', None),
                }
                if isinstance(batch, dict)
                else None
            )
            if compare_base:
                try:
                    outputs = model(video, query_points, meta=meta, return_info=True)
                except TypeError:
                    try:
                        outputs = model(video, query_points, return_info=True)
                    except TypeError:
                        outputs = model(video, query_points)
            else:
                try:
                    outputs = model(video, query_points, meta=meta)
                except TypeError:
                    outputs = model(video, query_points)
            if isinstance(outputs, (list, tuple)):
                pred_tracks, pred_visibility = outputs[0], outputs[1]
                if compare_base and len(outputs) >= 3 and isinstance(outputs[2], dict):
                    info = outputs[2]
                    base_tracks = info.get('base_tracks', None)
                    base_visibility = info.get('base_visibility', None)
            else:
                raise ValueError(f"Unexpected model output type: {type(outputs)}")

            # 计算指标
            batch_size = int(video.shape[0])
            for i in range(batch_size):
                resolution = _resolve_metric_resolution(
                    batch, i, tuple(video.shape[-2:]), metric_resolution_mode
                )
                vis_pred = pred_visibility[i]
                if vis_pred.dtype != torch.bool:
                    vis_pred = vis_pred > 0.5
                metrics = compute_tapvid_metrics(
                    pred_tracks[i],
                    target_points[i],
                    vis_pred,
                    ~occluded[i],
                    query_points[i],
                    resolution=resolution,
                    exclude_query_frame=exclude_query_frame,
                    query_mode=query_mode,
                )
                all_metrics.append(metrics)

                if long_occ_enabled and long_occ_thresholds:
                    try:
                        num_frames = int(pred_tracks[i].shape[1])
                        query_t = query_points[i, :, 0].round().long().clamp(0, num_frames - 1)
                        max_occ_run = _max_reappearance_occlusion_run(occluded[i], query_t)
                        for thr in long_occ_thresholds:
                            sel = max_occ_run >= int(thr)
                            num_sel = int(sel.long().sum().item())
                            if num_sel <= 0:
                                continue
                            longocc_num_queries[thr] += num_sel
                            longocc_num_videos[thr] += 1

                            subset_metrics = compute_tapvid_metrics(
                                pred_tracks[i][sel],
                                target_points[i][sel],
                                vis_pred[sel],
                                ~occluded[i][sel],
                                query_points[i][sel],
                                resolution=resolution,
                                exclude_query_frame=exclude_query_frame,
                                query_mode=query_mode,
                            )
                            all_longocc_metrics[thr].append(subset_metrics)

                            if base_tracks is not None and base_visibility is not None:
                                base_vis_subset = base_visibility[i]
                                if base_vis_subset.dtype != torch.bool:
                                    base_vis_subset = base_vis_subset > 0.5
                                base_subset_metrics = compute_tapvid_metrics(
                                    base_tracks[i][sel],
                                    target_points[i][sel],
                                    base_vis_subset[sel],
                                    ~occluded[i][sel],
                                    query_points[i][sel],
                                    resolution=resolution,
                                    exclude_query_frame=exclude_query_frame,
                                    query_mode=query_mode,
                                )
                                all_longocc_base_metrics[thr].append(base_subset_metrics)
                    except Exception as exc:
                        logger.warning(f"Long-occlusion subset evaluation failed: {exc}")

                if aux_metric_resolution_mode is not None:
                    aux_resolution = _resolve_metric_resolution(
                        batch,
                        i,
                        tuple(video.shape[-2:]),
                        aux_metric_resolution_mode,
                        fixed_resolution=aux_metric_fixed_resolution,
                    )
                    aux_metrics = compute_tapvid_metrics(
                        pred_tracks[i],
                        target_points[i],
                        vis_pred,
                        ~occluded[i],
                        query_points[i],
                        resolution=aux_resolution,
                        exclude_query_frame=exclude_query_frame,
                        query_mode=query_mode,
                    )
                    all_aux_metrics.append(aux_metrics)
                    if base_tracks is not None and base_visibility is not None:
                        try:
                            base_vis = base_visibility[i]
                            if base_vis.dtype != torch.bool:
                                base_vis = base_vis > 0.5
                            base_aux_metrics = compute_tapvid_metrics(
                                base_tracks[i],
                                target_points[i],
                                base_vis,
                                ~occluded[i],
                                query_points[i],
                                resolution=aux_resolution,
                                exclude_query_frame=exclude_query_frame,
                                query_mode=query_mode,
                            )
                            all_aux_base_metrics.append(base_aux_metrics)
                        except Exception as exc:
                            logger.warning(f"Failed to compute base aux metrics: {exc}")

                if base_tracks is not None and base_visibility is not None:
                    try:
                        base_vis = base_visibility[i]
                        if base_vis.dtype != torch.bool:
                            base_vis = base_vis > 0.5
                        base_metrics = compute_tapvid_metrics(
                            base_tracks[i],
                            target_points[i],
                            base_vis,
                            ~occluded[i],
                            query_points[i],
                            resolution=resolution,
                            exclude_query_frame=exclude_query_frame,
                            query_mode=query_mode,
                        )
                        all_base_metrics.append(base_metrics)
                    except Exception as exc:
                        logger.warning(f"Failed to compute base metrics: {exc}")
        except Exception as e:
            logger.warning(f"Skipping invalid validation sample: {e}")
            continue
    
    # 平均指标
    if len(all_metrics) == 0:
        logger.warning("No valid validation samples available; skipping metrics.")
        return {}
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = []
        for m in all_metrics:
            v = m.get(key)
            if v is None or not np.isfinite(v):
                continue
            values.append(v)
        avg_metrics[key] = float(np.mean(values)) if values else 0.0

    if compare_base and len(all_base_metrics) > 0:
        for key in all_base_metrics[0].keys():
            values = []
            for m in all_base_metrics:
                v = m.get(key)
                if v is None or not np.isfinite(v):
                    continue
                values.append(v)
            avg_metrics[f"{key}_base"] = float(np.mean(values)) if values else 0.0

        # Also report deltas (refined - base) for easier ablations.
        for key in all_base_metrics[0].keys():
            base_key = f"{key}_base"
            if key not in avg_metrics or base_key not in avg_metrics:
                continue
            try:
                avg_metrics[f"{key}_delta"] = float(avg_metrics[key]) - float(avg_metrics[base_key])
            except Exception:
                continue

    if aux_metric_resolution_mode is not None and len(all_aux_metrics) > 0:
        suffix = f"_{aux_metric_resolution_mode}"
        for key in all_aux_metrics[0].keys():
            values = []
            for m in all_aux_metrics:
                v = m.get(key)
                if v is None or not np.isfinite(v):
                    continue
                values.append(v)
            avg_metrics[f"{key}{suffix}"] = float(np.mean(values)) if values else 0.0

        if compare_base and len(all_aux_base_metrics) > 0:
            for key in all_aux_base_metrics[0].keys():
                values = []
                for m in all_aux_base_metrics:
                    v = m.get(key)
                    if v is None or not np.isfinite(v):
                        continue
                    values.append(v)
                avg_metrics[f"{key}{suffix}_base"] = float(np.mean(values)) if values else 0.0

            for key in all_aux_base_metrics[0].keys():
                refined_key = f"{key}{suffix}"
                base_key = f"{key}{suffix}_base"
                if refined_key not in avg_metrics or base_key not in avg_metrics:
                    continue
                try:
                    avg_metrics[f"{key}{suffix}_delta"] = float(avg_metrics[refined_key]) - float(avg_metrics[base_key])
                except Exception:
                    continue
    
    # 记录
    if long_occ_enabled and long_occ_thresholds:
        for thr in long_occ_thresholds:
            metrics_list = all_longocc_metrics.get(thr, [])
            if not metrics_list:
                continue
            occ_suffix = f"_longocc{int(thr)}"
            avg_metrics[f"num_queries{occ_suffix}"] = float(longocc_num_queries.get(thr, 0))
            avg_metrics[f"num_videos{occ_suffix}"] = float(longocc_num_videos.get(thr, 0))

            for key in metrics_list[0].keys():
                values = []
                for m in metrics_list:
                    v = m.get(key)
                    if v is None or not np.isfinite(v):
                        continue
                    values.append(v)
                avg_metrics[f"{key}{occ_suffix}"] = float(np.mean(values)) if values else 0.0

            base_list = all_longocc_base_metrics.get(thr, [])
            if compare_base and base_list:
                for key in base_list[0].keys():
                    values = []
                    for m in base_list:
                        v = m.get(key)
                        if v is None or not np.isfinite(v):
                            continue
                        values.append(v)
                    avg_metrics[f"{key}{occ_suffix}_base"] = float(np.mean(values)) if values else 0.0

                for key in base_list[0].keys():
                    refined_key = f"{key}{occ_suffix}"
                    base_key = f"{key}{occ_suffix}_base"
                    if refined_key not in avg_metrics or base_key not in avg_metrics:
                        continue
                    try:
                        avg_metrics[f"{key}{occ_suffix}_delta"] = float(avg_metrics[refined_key]) - float(avg_metrics[base_key])
                    except Exception:
                        continue

    if is_main_process:
        logger.info(f"Epoch {epoch} Evaluation:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        if exp_logger is not None:
            exp_logger.log_metrics(avg_metrics, step=epoch, prefix='val/')
    
    return avg_metrics


@torch.no_grad()
def maybe_visualize_predictions(model, val_loader, epoch, config, output_dir):
    """可选可视化预测结果"""
    vis_cfg = getattr(config.logging, 'visualize', None)
    if vis_cfg is None or not getattr(vis_cfg, 'enabled', False):
        return
    if val_loader is None:
        return
    try:
        if len(val_loader) == 0:
            return
    except TypeError:
        pass
    save_every = int(getattr(vis_cfg, 'save_every', 1) or 1)
    if save_every <= 0 or epoch % save_every != 0:
        return
    num_samples = int(getattr(vis_cfg, 'num_samples', 4) or 0)
    if num_samples <= 0:
        return

    try:
        from scripts.visualize import visualize_tracking_results
    except Exception as e:
        logger.warning(f"Visualization skipped: {e}")
        return

    device = next(model.parameters()).device
    vis_root = Path(output_dir) / "visualizations" / f"epoch_{epoch:04d}"
    vis_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    for i, batch in enumerate(val_loader):
        if i >= num_samples:
            break
        video = batch['video'].to(device)
        query_points = batch['query_points'].to(device)
        target_points = batch['target_points'].to(device)
        occluded = batch['occluded'].to(device)
        if occluded.dtype != torch.bool:
            occluded = occluded > 0.5
        video_name = batch.get('video_name', f'sample_{i}')
        if isinstance(video_name, (list, tuple)):
            video_name = video_name[0] if video_name else f'sample_{i}'
        video_name = str(video_name)

        meta = (
            {
                'video_name': batch.get('video_name', None),
                'base_tracks': batch.get('base_tracks', None),
                'base_visibility': batch.get('base_visibility', None),
            }
            if isinstance(batch, dict)
            else None
        )
        try:
            outputs = model(video, query_points, meta=meta)
        except TypeError:
            outputs = model(video, query_points)
        if isinstance(outputs, (list, tuple)):
            pred_tracks, pred_visibility = outputs[0], outputs[1]
        else:
            raise ValueError(f"Unexpected model output type: {type(outputs)}")

        video_np = video[0].cpu().permute(0, 2, 3, 1).numpy()
        pred_tracks_np = pred_tracks[0].cpu().numpy()
        pred_vis_np = (pred_visibility[0] > 0.5).cpu().numpy()
        gt_tracks_np = target_points[0].cpu().numpy()
        gt_vis_np = (~occluded[0]).cpu().numpy()

        visualize_tracking_results(
            video_np,
            pred_tracks_np,
            gt_tracks_np,
            pred_vis_np,
            gt_vis_np,
            output_dir=str(vis_root),
            video_name=str(video_name),
            max_points=20,
            fps=8,
        )


def _align_state_dict_keys(state_dict, model_state):
    if not state_dict or not model_state:
        return state_dict
    sd_keys = list(state_dict.keys())
    ms_keys = list(model_state.keys())
    if not sd_keys or not ms_keys:
        return state_dict
    has_module = sd_keys[0].startswith('module.')
    model_has_module = ms_keys[0].startswith('module.')
    if has_module and not model_has_module:
        return {k[7:]: v for k, v in state_dict.items()}
    if not has_module and model_has_module:
        return {f'module.{k}': v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    metrics,
    config,
    path,
    ema=None,
    early_stopping=None,
    is_main_process: bool = True,
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': OmegaConf.to_container(config),
    }
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    if early_stopping is not None:
        checkpoint['early_stopping_state'] = early_stopping.state_dict()
    torch.save(checkpoint, path)
    if is_main_process:
        logger.info(f"Saved checkpoint to {path}")


def _prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """保留最近的epoch检查点，清理过旧文件"""
    if keep_last is None or keep_last <= 0:
        return
    try:
        checkpoint_dir = Path(checkpoint_dir)
    except Exception:
        return
    if not checkpoint_dir.exists():
        return
    epoch_ckpts = []
    for path in checkpoint_dir.glob("epoch_*.pth"):
        stem = path.stem
        if not stem.startswith("epoch_"):
            continue
        num_str = stem[len("epoch_"):]
        if num_str.isdigit():
            epoch_ckpts.append((int(num_str), path))
    epoch_ckpts.sort(key=lambda x: x[0], reverse=True)
    for _, path in epoch_ckpts[keep_last:]:
        try:
            path.unlink()
        except Exception as exc:
            logger.warning(f"Failed to remove old checkpoint {path}: {exc}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, ema=None, early_stopping=None):
    """加载检查点"""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    if state_dict and model_state:
        state_dict = _align_state_dict_keys(state_dict, model_state)
    model.load_state_dict(state_dict)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 将优化器状态移动到模型所在设备，避免恢复训练时报设备不一致
        try:
            device = next(model.parameters()).device
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        except Exception as exc:
            logger.warning(f"Failed to move optimizer state to device: {exc}")
    
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if ema is not None and 'ema_state_dict' in checkpoint:
        ema_state = checkpoint['ema_state_dict']
        ema_model_state = ema.ema_model.state_dict()
        ema_model_dict = ema_state.get('ema_model', None)
        if ema_model_dict is not None and ema_model_state:
            ema_state['ema_model'] = _align_state_dict_keys(ema_model_dict, ema_model_state)
        ema.load_state_dict(ema_state)

    if early_stopping is not None and 'early_stopping_state' in checkpoint:
        early_stopping.load_state_dict(checkpoint['early_stopping_state'])
    
    return checkpoint['epoch'], checkpoint.get('metrics', {})


def load_pretrained(path, model):
    """加载预训练权重（仅模型参数，允许部分匹配）"""
    if not path:
        return
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        logger.warning(f"Pretrained checkpoint not found: {ckpt_path}")
        return
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = model.state_dict()
    if state_dict and model_state:
        state_dict = _align_state_dict_keys(state_dict, model_state)
        # Allow loading checkpoints across small architecture edits (e.g. different
        # local correlation window sizes). PyTorch will still raise on shape
        # mismatches even with strict=False, so we drop incompatible keys.
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            if key in model_state:
                target = model_state[key]
                if torch.is_tensor(value) and torch.is_tensor(target):
                    if tuple(value.shape) != tuple(target.shape):
                        skipped.append((key, tuple(value.shape), tuple(target.shape)))
                        continue
            filtered[key] = value
        if skipped:
            preview = ", ".join(
                [f"{k} {src}->{dst}" for k, src, dst in skipped[:5]]
            )
            extra = "" if len(skipped) <= 5 else f" (+{len(skipped) - 5} more)"
            logger.warning(
                f"Dropped {len(skipped)} pretrained keys due to shape mismatch: {preview}{extra}"
            )
        state_dict = filtered
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"Loaded pretrained from {ckpt_path} (missing={len(missing)}, unexpected={len(unexpected)})"
    )


def main():
    _configure_hf_download_endpoint()
    args, unknown = parse_args()
    project_root = Path(__file__).parent
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.config and not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return
    
    # 加载配置
    config = load_config(args.config)
    if unknown:
        overrides = []
        idx = 0
        while idx < len(unknown):
            item = unknown[idx]
            if not item.startswith('--'):
                idx += 1
                continue
            key = item.lstrip('-')
            if '=' in key:
                overrides.append(key)
            else:
                if idx + 1 >= len(unknown):
                    raise ValueError(f"Missing value for override {item}")
                value = unknown[idx + 1]
                overrides.append(f"{key}={value}")
                idx += 1
            idx += 1
        if overrides:
            config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    if args.resume:
        args.resume = _resolve_path(args.resume, project_root)

    if args.debug and hasattr(config, 'logging') and hasattr(config.logging, 'wandb'):
        config.logging.wandb.enabled = False

    # 解析相对路径
    if hasattr(config, 'paths'):
        config.paths.output_dir = _resolve_path(config.paths.output_dir, project_root)
        config.paths.checkpoint_dir = _resolve_path(config.paths.checkpoint_dir, project_root)
    if hasattr(config, 'logging') and hasattr(config.logging, 'log_dir'):
        config.logging.log_dir = _resolve_path(config.logging.log_dir, project_root)
    if hasattr(config, 'data'):
        config.data.train.root = _resolve_path(config.data.train.root, project_root)
        config.data.val.root = _resolve_path(config.data.val.root, project_root)
        if hasattr(config.data, 'pseudo') and hasattr(config.data.pseudo, 'root'):
            config.data.pseudo.root = _resolve_path(config.data.pseudo.root, project_root)

    # Progressive training (optional) - apply stage 0 before creating loaders
    progressive_cfg = getattr(getattr(config, 'training', None), 'progressive', None)
    progressive_scheduler = None
    progressive_stage_signature = None
    progressive_base_lr = None
    if progressive_cfg is not None and getattr(progressive_cfg, 'enabled', False):
        train_resolution = getattr(config.data.train, 'resolution', None)
        if train_resolution is None:
            logger.warning("Progressive training enabled but train resolution missing; disabling.")
        else:
            try:
                from utils.advanced_training import ProgressiveStage, ProgressiveTrainingScheduler, create_progressive_stages
                stages = []
                stages_cfg = getattr(progressive_cfg, 'stages', None)
                if stages_cfg:
                    for stage_cfg in stages_cfg:
                        if OmegaConf.is_config(stage_cfg):
                            stage_cfg = OmegaConf.to_container(stage_cfg, resolve=True)
                        stages.append(ProgressiveStage(
                            resolution=tuple(stage_cfg.get('resolution', train_resolution)),
                            num_frames=int(stage_cfg.get('num_frames', config.data.train.num_frames)),
                            num_points=int(stage_cfg.get('num_points', config.data.train.num_points)),
                            batch_size=int(stage_cfg.get('batch_size', config.training.batch_size)),
                            learning_rate=float(stage_cfg.get('learning_rate', config.training.optimizer.lr)),
                            epochs=int(stage_cfg.get('epochs', 1)),
                        ))
                else:
                    stages = create_progressive_stages(config)
                progressive_scheduler = ProgressiveTrainingScheduler(
                    stages=stages,
                    smooth_transition=bool(getattr(progressive_cfg, 'smooth_transition', False)),
                    transition_epochs=int(getattr(progressive_cfg, 'transition_epochs', 1) or 1),
                )
                progressive_base_lr = float(config.training.optimizer.lr)
                if getattr(progressive_cfg, 'smooth_transition', False):
                    stage0 = progressive_scheduler.get_interpolated_config(0)
                else:
                    stage0 = progressive_scheduler.get_current_stage(0)
                    stage0 = {
                        'resolution': stage0.resolution,
                        'num_frames': stage0.num_frames,
                        'num_points': stage0.num_points,
                        'batch_size': stage0.batch_size,
                        'learning_rate': stage0.learning_rate,
                    }
                _apply_progressive_stage(stage0, config, allow_lr_batch_update=True)
                progressive_stage_signature = (
                    tuple(stage0['resolution']),
                    int(stage0['num_frames']),
                    int(stage0['num_points']),
                    int(stage0['batch_size']),
                    float(stage0['learning_rate']),
                )
            except Exception as exc:
                logger.warning(f"Progressive training setup failed: {exc}. Disabling.")
                progressive_scheduler = None

    # 初始化分布式
    distributed, rank, world_size, local_rank = setup_distributed(config)
    is_main_process = (not distributed) or rank == 0
    if not is_main_process:
        logger.setLevel(logging.WARN)
    if distributed and world_size > 1 and is_main_process:
        logger.warning(
            "Distributed training in train.py does not auto-scale learning rate. "
            "Consider using scripts/train_distributed.py or adjust LR manually."
        )
    
    # 设置随机种子
    seed = int(config.experiment.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(config.experiment, 'deterministic') and config.experiment.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            logger.warning(f"Deterministic algorithms not fully enabled: {exc}")
    
    # 设备
    if distributed:
        if torch.cuda.is_available():
            device = torch.device('cuda', local_rank)
        else:
            device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    output_dir = Path(config.paths.output_dir) / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(config.paths.checkpoint_dir) / config.experiment.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process:
        OmegaConf.save(config, output_dir / 'config.yaml')
    
    # 初始化实验日志（仅主进程）
    log_root = output_dir
    if hasattr(config, 'logging') and hasattr(config.logging, 'log_dir'):
        log_root = Path(config.logging.log_dir) / config.experiment.name
        log_root.mkdir(parents=True, exist_ok=True)
    exp_logger = create_experiment_logger(
        config,
        log_root,
        is_main_process=is_main_process and not args.debug,
    )

    # Strict dependency check before model creation (fail-fast, avoid silent degraded runs).
    module_status = _collect_module_status(config, model=None)
    _validate_required_modules(config, module_status)
    
    # 创建模型
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # 加载预训练权重（不影响resume）
    pretrained_path = getattr(config, 'pretrained', None)
    if pretrained_path is None and hasattr(config, 'paths') and hasattr(config.paths, 'pretrained'):
        if isinstance(config.paths.pretrained, str):
            pretrained_path = config.paths.pretrained
        else:
            for key in ['fspt', 'model', 'checkpoint']:
                if hasattr(config.paths.pretrained, key):
                    value = getattr(config.paths.pretrained, key)
                    if value:
                        pretrained_path = value
                        break
    if pretrained_path and not args.resume:
        pretrained_path = _resolve_path(str(pretrained_path), project_root)
        load_pretrained(pretrained_path, model)

    # Optional generic phase schedule for staged freeze/unfreeze on any model.
    phase_schedule = _resolve_training_phase_schedule(config)
    active_phase_idx: Optional[int] = None
    if phase_schedule:
        stage_idx, stage_cfg = _find_active_phase(phase_schedule, 0)
        if stage_cfg is not None:
            stats = _apply_phase_trainability(model, stage_cfg)
            active_phase_idx = stage_idx
            if is_main_process:
                logger.info(
                    "Applied phase schedule at startup: "
                    f"phase={stage_cfg.get('name')} start_epoch={stage_cfg.get('start_epoch')} "
                    f"trainable={stats['trainable_params']:,}/{stats['total_params']:,}"
                )

    # Runtime module summary for reproducibility audits.
    module_status = _collect_module_status(config, model=model)
    _validate_required_modules(config, module_status)
    if is_main_process:
        status_path = output_dir / "module_status.json"
        try:
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(module_status, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved module status summary to {status_path}")
        except Exception as exc:
            logger.warning(f"Failed to write module status summary: {exc}")
        logger.info(
            "Module status summary:\n"
            + json.dumps(module_status, indent=2, ensure_ascii=False)
        )
    
    # 多GPU
    if distributed:
        dist_cfg = getattr(getattr(config, 'hardware', None), 'distributed', None)
        find_unused = bool(getattr(dist_cfg, 'find_unused_parameters', False)) if dist_cfg is not None else False
        if device.type == 'cuda':
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=find_unused,
            )
        else:
            model = nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=find_unused,
            )
    elif device.type == 'cuda':
        gpu_ids = []
        if hasattr(config, 'hardware') and hasattr(config.hardware, 'gpus'):
            gpu_ids = _parse_gpu_ids(config.hardware.gpus)
        available_gpus = torch.cuda.device_count()
        if gpu_ids:
            gpu_ids = [i for i in gpu_ids if 0 <= i < available_gpus]
        if not gpu_ids:
            gpu_ids = [device.index if device.index is not None else 0]
        if len(gpu_ids) > 1:
            if is_main_process:
                logger.warning(
                    "Using DataParallel with multiple GPUs. For better performance and stability, "
                    "prefer scripts/train_distributed.py (DDP)."
                )
            model = nn.DataParallel(model, device_ids=gpu_ids)

    if is_main_process and exp_logger is not None:
        wandb_cfg = getattr(getattr(config, 'logging', None), 'wandb', None)
        if wandb_cfg is not None and getattr(wandb_cfg, 'watch_model', False):
            log_freq = getattr(wandb_cfg, 'log_freq', 100)
            log_freq = int(log_freq) if log_freq is not None else 100
            if log_freq > 0:
                exp_logger.log_model_gradients(unwrap_model(model), log_freq=log_freq)
            else:
                logger.warning("wandb.log_freq <= 0; skip gradient watching.")
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据加载器
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        config,
        args.debug,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main_process,
    )
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        logger.info(f"Val samples: {len(val_loader.dataset)}")

    # 伪标签自训练数据（可选）
    pseudo_cfg = getattr(getattr(config, 'training', None), 'pseudo_labeling', None)
    pseudo_loader = None
    pseudo_trainer = None
    if pseudo_cfg is not None and getattr(pseudo_cfg, 'enabled', False):
        pseudo_loader = create_pseudo_dataloader(
            config,
            args.debug,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        if pseudo_loader is None:
            logger.warning("Pseudo-labeling enabled but pseudo dataloader not created; disabling.")
        else:
            from utils.advanced_training import PseudoLabelTrainer
            teacher_model = deepcopy(unwrap_model(model)).to(device)
            teacher_model.eval()
            pseudo_trainer = PseudoLabelTrainer(
                teacher_model=teacher_model,
                student_model=model,
                teacher_momentum=float(getattr(pseudo_cfg, 'teacher_momentum', 0.999) or 0.999),
                confidence_threshold=float(getattr(pseudo_cfg, 'confidence_threshold', 0.8) or 0.8),
                use_soft_labels=bool(getattr(pseudo_cfg, 'use_soft_labels', True)),
                confidence_source=str(getattr(pseudo_cfg, 'confidence_source', 'auto') or 'auto'),
            )
    
    # 创建损失函数
    criterion = PointTrackingLoss(config)

    # 学习率查找器（可选）
    amp_dtype = _get_amp_dtype(config)
    amp_cfg = getattr(getattr(config, 'training', None), 'amp', None)
    amp_enabled = bool(amp_cfg is not None and getattr(amp_cfg, 'enabled', False) and device.type == 'cuda')
    lr_finder_cfg = getattr(getattr(config, 'training', None), 'lr_finder', None)
    grad_cfg = getattr(getattr(config, 'training', None), 'gradient', None)
    accum_steps_for_lr = int(getattr(grad_cfg, 'accumulation_steps', 1) or 1)
    opt_cfg = getattr(getattr(config, 'training', None), 'optimizer', None)
    opt_type = str(getattr(opt_cfg, 'type', 'AdamW')).lower() if opt_cfg is not None else 'adamw'
    if opt_type == 'adamw':
        optimizer_cls = torch.optim.AdamW
    elif opt_type == 'adam':
        optimizer_cls = torch.optim.Adam
    else:
        raise ValueError(f"Unsupported optimizer for LR finder: {opt_type}")
    optimizer_kwargs = {}
    betas = getattr(opt_cfg, 'betas', None) if opt_cfg is not None else None
    if betas is not None:
        optimizer_kwargs['betas'] = tuple(betas)
    lr_finder_enabled = bool(lr_finder_cfg is not None and getattr(lr_finder_cfg, 'enabled', False))
    apply_lr = True
    suggested_lr = None
    if lr_finder_enabled and not args.resume and not args.eval_only:
        apply_lr = bool(getattr(lr_finder_cfg, 'apply', True))
        if is_main_process:
            from utils.lr_finder import find_optimal_lr
            plot_path = None
            if hasattr(lr_finder_cfg, 'plot_path') and lr_finder_cfg.plot_path:
                plot_path = _resolve_path(str(lr_finder_cfg.plot_path), project_root)
            suggested_lr = find_optimal_lr(
                model=unwrap_model(model),
                train_loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer_cls=optimizer_cls,
                start_lr=float(getattr(lr_finder_cfg, 'start_lr', 1e-7)),
                end_lr=float(getattr(lr_finder_cfg, 'end_lr', 1.0)),
                num_iter=int(getattr(lr_finder_cfg, 'num_iter', 100)),
                weight_decay=float(config.training.optimizer.weight_decay),
                plot_path=plot_path,
                accumulation_steps=accum_steps_for_lr,
                use_amp=amp_enabled,
                amp_dtype=amp_dtype if amp_enabled else None,
                optimizer_kwargs=optimizer_kwargs,
            )
            if suggested_lr is not None and math.isfinite(float(suggested_lr)):
                logger.info(f"LR finder suggested lr: {float(suggested_lr):.2e}")
                if exp_logger is not None:
                    exp_logger.log_metrics({'suggested_lr': float(suggested_lr)}, step=0, prefix='lr_finder/')
            else:
                logger.warning("LR finder did not return a valid lr; skipping.")
                suggested_lr = None
        if distributed and lr_finder_enabled:
            lr_value = float(suggested_lr) if suggested_lr is not None else float('nan')
            lr_tensor = torch.tensor([lr_value], device=device)
            dist.broadcast(lr_tensor, src=0)
            suggested_lr = float(lr_tensor.item())
            if not math.isfinite(suggested_lr):
                suggested_lr = None
        if apply_lr and suggested_lr is not None:
            config.training.optimizer.lr = float(suggested_lr)
            logger.info(f"Applied suggested lr: {config.training.optimizer.lr:.2e}")

        # 如果启用了渐进式训练，按比例缩放各阶段学习率
        if progressive_scheduler is not None and apply_lr and suggested_lr is not None and progressive_base_lr:
            scale = float(config.training.optimizer.lr) / float(progressive_base_lr)
            if scale != 1.0:
                for stage in progressive_scheduler.stages:
                    stage.learning_rate = float(stage.learning_rate) * scale
                if getattr(progressive_cfg, 'smooth_transition', False):
                    stage0 = progressive_scheduler.get_interpolated_config(0)
                else:
                    stage = progressive_scheduler.get_current_stage(0)
                    stage0 = {
                        'resolution': stage.resolution,
                        'num_frames': stage.num_frames,
                        'num_points': stage.num_points,
                        'batch_size': stage.batch_size,
                        'learning_rate': stage.learning_rate,
                    }
                _apply_progressive_stage(stage0, config, allow_lr_batch_update=True)
                progressive_stage_signature = (
                    tuple(stage0['resolution']),
                    int(stage0['num_frames']),
                    int(stage0['num_points']),
                    int(stage0['batch_size']),
                    float(stage0['learning_rate']),
                )

        if is_main_process:
            OmegaConf.save(config, output_dir / 'config.yaml')

    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    accum_steps = max(1, int(config.training.gradient.accumulation_steps))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    num_training_steps = steps_per_epoch * config.training.epochs
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    if phase_schedule and active_phase_idx is not None:
        phase_cfg = phase_schedule[active_phase_idx]
        if _apply_phase_lr_multipliers(optimizer, phase_cfg):
            _set_optimizer_lr(optimizer, float(config.training.optimizer.lr))
            _update_scheduler_base_lrs(scheduler, float(config.training.optimizer.lr))
            if is_main_process:
                logger.info(
                    "Applied phase-specific lr_scale_overrides at startup: "
                    f"phase={phase_cfg.get('name')}"
                )

    if is_main_process:
        global_batch = config.training.batch_size * (world_size if distributed else 1)
        effective_batch = global_batch * accum_steps
        logger.info(f"Global batch size: {global_batch} (accumulation {accum_steps} -> effective {effective_batch})")
    
    # 混合精度
    scaler_enabled = (
        amp_enabled and amp_dtype == torch.float16
    )
    scaler = GradScaler(enabled=scaler_enabled)

    # EMA与早停（仅主进程维护早停）
    ema = create_ema(unwrap_model(model), config, device)
    early_stopping = create_early_stopping(config) if is_main_process else None
    
    # 恢复训练
    start_epoch = 0
    monitor_cfg = getattr(config.training, 'early_stopping', None)
    monitor_key = getattr(monitor_cfg, 'monitor', 'AJ') if monitor_cfg else 'AJ'
    monitor_mode = getattr(monitor_cfg, 'mode', 'max') if monitor_cfg else 'max'
    best_score = float('-inf') if monitor_mode == 'max' else float('inf')
    best_metrics = {monitor_key: best_score}

    monitor_ckpt_cfg = getattr(getattr(config, 'training', None), 'monitor_checkpoints', None)
    save_best_aj = bool(getattr(monitor_ckpt_cfg, 'save_best_aj', True)) if monitor_ckpt_cfg is not None else True
    save_best_oa = bool(getattr(monitor_ckpt_cfg, 'save_best_oa', True)) if monitor_ckpt_cfg is not None else True
    best_aj_score = float('-inf')
    best_oa_score = float('-inf')

    plateau_cfg = getattr(getattr(config, 'training', None), 'plateau_guard', None)
    plateau_enabled = bool(plateau_cfg is not None and getattr(plateau_cfg, 'enabled', False))
    plateau_start_epoch = int(getattr(plateau_cfg, 'start_epoch', 0) or 0) if plateau_cfg is not None else 0
    plateau_patience = int(getattr(plateau_cfg, 'patience', 3) or 3) if plateau_cfg is not None else 3
    plateau_patience = max(1, plateau_patience)
    plateau_min_delta_aj = float(getattr(plateau_cfg, 'min_delta_aj', 0.0) or 0.0) if plateau_cfg is not None else 0.0
    plateau_min_delta_oa = float(getattr(plateau_cfg, 'min_delta_oa', 0.0) or 0.0) if plateau_cfg is not None else 0.0
    plateau_min_delta_4px = float(getattr(plateau_cfg, 'min_delta_4px', 0.0) or 0.0) if plateau_cfg is not None else 0.0
    plateau_min_delta_error = float(getattr(plateau_cfg, 'min_delta_error', 0.0) or 0.0) if plateau_cfg is not None else 0.0
    plateau_bad_aj = 0
    plateau_bad_oa = 0
    plateau_bad_4px = 0
    plateau_rising_error = 0
    plateau_best_aj = float('-inf')
    plateau_best_oa = float('-inf')
    plateau_best_4px = float('-inf')
    plateau_last_avg_error = None

    deadline_cfg = getattr(getattr(config, 'training', None), 'deadline_guard', None)
    deadline_enabled = bool(deadline_cfg is not None and getattr(deadline_cfg, 'enabled', False))
    deadline_epoch = int(getattr(deadline_cfg, 'deadline_epoch', 0) or 0) if deadline_cfg is not None else 0
    deadline_threshold = float(getattr(deadline_cfg, 'threshold', 0.0) or 0.0) if deadline_cfg is not None else 0.0
    deadline_monitor_key = str(getattr(deadline_cfg, 'monitor', monitor_key) or monitor_key) if deadline_cfg is not None else monitor_key
    deadline_mode = str(getattr(deadline_cfg, 'mode', monitor_mode) or monitor_mode) if deadline_cfg is not None else monitor_mode
    if deadline_mode not in ('max', 'min'):
        if is_main_process:
            logger.warning(
                "training.deadline_guard.mode must be 'max' or 'min'; "
                f"got {deadline_mode!r}. Falling back to {monitor_mode!r}."
            )
        deadline_mode = monitor_mode
    deadline_best_score = float('-inf') if deadline_mode == 'max' else float('inf')

    if is_main_process:
        logger.info(
            "Best-checkpoint monitors: "
            f"main={monitor_key}({monitor_mode}), save_best_aj={save_best_aj}, save_best_oa={save_best_oa}"
        )
        if plateau_enabled:
            logger.info(
                "Plateau guard enabled: "
                f"start_epoch={plateau_start_epoch}, patience={plateau_patience}, "
                f"min_delta_aj={plateau_min_delta_aj}, min_delta_oa={plateau_min_delta_oa}, "
                f"min_delta_4px={plateau_min_delta_4px}, "
                f"min_delta_error={plateau_min_delta_error}"
            )
        if deadline_enabled:
            logger.info(
                "Deadline guard enabled: "
                f"monitor={deadline_monitor_key}({deadline_mode}), "
                f"deadline_epoch={deadline_epoch}, threshold={deadline_threshold}"
            )
    
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        start_epoch, best_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema=ema, early_stopping=early_stopping
        )
        if start_epoch < 0:
            start_epoch = 0
        start_epoch += 1
        if best_metrics and monitor_key in best_metrics:
            best_score = best_metrics[monitor_key]
        if isinstance(best_metrics, dict):
            if 'AJ' in best_metrics:
                try:
                    best_aj_score = float(best_metrics['AJ'])
                    plateau_best_aj = max(plateau_best_aj, best_aj_score)
                except Exception:
                    pass
            if 'AJ_delta' in best_metrics:
                try:
                    plateau_best_aj = max(plateau_best_aj, float(best_metrics['AJ_delta']))
                except Exception:
                    pass
            if 'OA' in best_metrics:
                try:
                    best_oa_score = float(best_metrics['OA'])
                except Exception:
                    pass
            if '<4px' in best_metrics:
                try:
                    plateau_best_4px = max(plateau_best_4px, float(best_metrics['<4px']))
                except Exception:
                    pass
            if '<4px_delta' in best_metrics:
                try:
                    plateau_best_4px = max(plateau_best_4px, float(best_metrics['<4px_delta']))
                except Exception:
                    pass
            if 'avg_error_px' in best_metrics:
                try:
                    plateau_last_avg_error = float(best_metrics['avg_error_px'])
                except Exception:
                    pass
            if 'avg_error_px_delta' in best_metrics:
                try:
                    plateau_last_avg_error = float(best_metrics['avg_error_px_delta'])
                except Exception:
                    pass
            if deadline_enabled and deadline_monitor_key in best_metrics:
                try:
                    deadline_best_score = float(best_metrics[deadline_monitor_key])
                except Exception:
                    pass
        if deadline_enabled and deadline_monitor_key == monitor_key:
            deadline_best_score = best_score

        # When resuming from latest.pth / epoch_*.pth, the checkpoint metrics are
        # "last epoch" metrics, not necessarily the historical best. Restore the
        # tracked best scores from the corresponding best checkpoints to avoid
        # overwriting them with worse values after resume.
        if is_main_process:
            try:
                best_path = checkpoint_dir / "best.pth"
                if best_path.exists():
                    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
                    restored = ckpt.get("metrics", None)
                    if isinstance(restored, dict) and monitor_key in restored:
                        best_metrics = restored
                        best_score = float(restored[monitor_key])
                        if deadline_enabled and deadline_monitor_key == monitor_key:
                            deadline_best_score = best_score
                        logger.info(
                            f"Restored best monitor from {best_path} "
                            f"({monitor_key}={best_score:.6f})."
                        )
            except Exception as exc:
                logger.warning(f"Failed to restore best.pth metrics on resume: {exc}")

            if save_best_aj:
                try:
                    best_aj_path = checkpoint_dir / "best_aj.pth"
                    if best_aj_path.exists():
                        ckpt = torch.load(best_aj_path, map_location="cpu", weights_only=False)
                        restored = ckpt.get("metrics", None)
                        if isinstance(restored, dict) and "AJ" in restored:
                            best_aj_score = float(restored["AJ"])
                except Exception as exc:
                    logger.warning(f"Failed to restore best_aj.pth score on resume: {exc}")

            if save_best_oa:
                try:
                    best_oa_path = checkpoint_dir / "best_oa.pth"
                    if best_oa_path.exists():
                        ckpt = torch.load(best_oa_path, map_location="cpu", weights_only=False)
                        restored = ckpt.get("metrics", None)
                        if isinstance(restored, dict) and "OA" in restored:
                            best_oa_score = float(restored["OA"])
                except Exception as exc:
                    logger.warning(f"Failed to restore best_oa.pth score on resume: {exc}")

        # Ensure monitor-specific best checkpoints exist after resume.
        # This avoids missing best_aj/best_oa files when resumed runs never
        # surpass historical best scores.
        if is_main_process and isinstance(best_metrics, dict):
            if save_best_aj and np.isfinite(best_aj_score):
                best_aj_path = checkpoint_dir / 'best_aj.pth'
                if not best_aj_path.exists():
                    try:
                        save_checkpoint(
                            model, optimizer, scheduler, start_epoch - 1, best_metrics, config,
                            best_aj_path, ema=ema, early_stopping=early_stopping,
                            is_main_process=is_main_process,
                        )
                        logger.info(
                            f"Initialized best_aj.pth from resume checkpoint (AJ={best_aj_score:.4f})."
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to initialize best_aj.pth on resume: {exc}")
            if save_best_oa and np.isfinite(best_oa_score):
                best_oa_path = checkpoint_dir / 'best_oa.pth'
                if not best_oa_path.exists():
                    try:
                        save_checkpoint(
                            model, optimizer, scheduler, start_epoch - 1, best_metrics, config,
                            best_oa_path, ema=ema, early_stopping=early_stopping,
                            is_main_process=is_main_process,
                        )
                        logger.info(
                            f"Initialized best_oa.pth from resume checkpoint (OA={best_oa_score:.4f})."
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to initialize best_oa.pth on resume: {exc}")
        if pseudo_trainer is not None:
            try:
                pseudo_trainer.teacher.load_state_dict(unwrap_model(model).state_dict(), strict=False)
                pseudo_trainer.teacher.to(device)
                pseudo_trainer.teacher.eval()
                logger.info("Synced pseudo-label teacher with resumed student weights.")
            except Exception as exc:
                logger.warning(f"Failed to sync pseudo-label teacher after resume: {exc}")
        if phase_schedule:
            resume_phase_idx, resume_phase_cfg = _find_active_phase(phase_schedule, start_epoch)
            if resume_phase_cfg is not None:
                stats = _apply_phase_trainability(unwrap_model(model), resume_phase_cfg)
                active_phase_idx = resume_phase_idx
                if _apply_phase_lr_multipliers(optimizer, resume_phase_cfg):
                    _set_optimizer_lr(optimizer, float(config.training.optimizer.lr))
                    _update_scheduler_base_lrs(scheduler, float(config.training.optimizer.lr))
                if is_main_process:
                    logger.info(
                        "Re-applied phase schedule after resume: "
                        f"epoch={start_epoch} phase={resume_phase_cfg.get('name')} "
                        f"trainable={stats['trainable_params']:,}/{stats['total_params']:,}"
                    )
    
    ema_cfg = getattr(config.training, 'ema', None)
    use_ema_for_eval = bool(ema_cfg is not None and getattr(ema_cfg, 'use_for_eval', True))
    eval_cfg = getattr(config, 'evaluation', None)
    eval_every = int(getattr(eval_cfg, 'eval_every', 1) or 1)
    if eval_every <= 0:
        if is_main_process:
            logger.warning("evaluation.eval_every <= 0; only evaluating at final epoch.")
        eval_every = None
    checkpoint_cfg = getattr(getattr(config, 'training', None), 'checkpoint', None)
    save_every = int(getattr(checkpoint_cfg, 'save_every', 0) or 0)
    if save_every <= 0 and is_main_process:
        logger.warning("training.checkpoint.save_every <= 0; periodic checkpoints disabled.")
    keep_last = int(getattr(checkpoint_cfg, 'keep_last', 0) or 0)

    # ---------------------------------------------------------------------
    # Refiner stage logging (Route A / staged freeze-unfreeze schedules)
    # ---------------------------------------------------------------------
    stage_log_path = output_dir / "refiner_stage_log.jsonl"
    stage_transition_path = output_dir / "refiner_stage_transitions.jsonl"
    stage_schedule_path = output_dir / "refiner_stages.json"
    phase_log_path = output_dir / "training_phase_log.jsonl"
    phase_schedule_path = output_dir / "training_phases.json"
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    refiner_stage_logging_enabled = False
    resolved_stages = None
    try:
        refiner_cfg = getattr(getattr(config, "model", None), "refiner", None)
        stages = getattr(refiner_cfg, "stages", None) if refiner_cfg is not None else None
        if stages is not None:
            resolved_stages = []
            for s in list(stages):
                if s is None:
                    continue
                try:
                    stage_obj = OmegaConf.to_container(s, resolve=True)
                except Exception:
                    stage_obj = None
                if isinstance(stage_obj, dict):
                    resolved_stages.append(stage_obj)
                    continue
                try:
                    stage_obj = dict(s)
                except Exception:
                    stage_obj = None
                if isinstance(stage_obj, dict):
                    resolved_stages.append(stage_obj)
            refiner_stage_logging_enabled = len(resolved_stages) > 0
    except Exception:
        resolved_stages = None
        refiner_stage_logging_enabled = False

    # Fresh run: clear old epoch logs to avoid mixing experiments.
    if is_main_process and not args.resume and start_epoch == 0:
        try:
            if epoch_log_path.exists():
                epoch_log_path.unlink()
        except Exception:
            pass
        try:
            if phase_log_path.exists():
                phase_log_path.unlink()
        except Exception:
            pass

    if is_main_process and phase_schedule:
        if not args.resume and start_epoch == 0:
            try:
                if phase_log_path.exists():
                    phase_log_path.unlink()
            except Exception:
                pass
        try:
            with open(phase_schedule_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment": str(config.experiment.name),
                        "timestamp": datetime.now().isoformat(),
                        "phases": phase_schedule,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as exc:
            logger.warning(f"Failed to write training phase schedule: {exc}")

    if is_main_process and refiner_stage_logging_enabled:
        # Fresh run: clear old stage logs to avoid mixing experiments.
        if not args.resume and start_epoch == 0:
            for p in (stage_log_path, stage_transition_path):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        try:
            with open(stage_schedule_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "experiment": str(config.experiment.name),
                        "timestamp": datetime.now().isoformat(),
                        "stages": resolved_stages,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as exc:
            logger.warning(f"Failed to write refiner stage schedule: {exc}")

        # Optional: log a compact stage table to WandB (no-op if WandB disabled).
        if exp_logger is not None and resolved_stages is not None:
            try:
                table_rows = []
                for idx, stage in enumerate(resolved_stages):
                    start_epoch_i = stage.get("start_epoch", 0)
                    end_epoch_i = stage.get("end_epoch", stage.get("until_epoch", None))
                    trainable = stage.get("trainable_modules", None)
                    if isinstance(trainable, (list, tuple)):
                        trainable_s = ",".join([str(v) for v in trainable])
                    else:
                        trainable_s = str(trainable) if trainable is not None else ""
                    table_rows.append([int(idx), int(start_epoch_i), str(end_epoch_i), trainable_s])
                exp_logger.log_table(
                    tag="refiner/stages",
                    columns=["stage_idx", "start_epoch", "end_epoch", "trainable_modules"],
                    data=table_rows,
                    step=int(start_epoch),
                )
            except Exception:
                pass

    # 仅评估
    if args.eval_only:
        if is_main_process:
            eval_model = ema.ema_model if (ema is not None and use_ema_for_eval) else unwrap_model(model)
            metrics = evaluate(
                eval_model, val_loader, 0, config, exp_logger, is_main_process=is_main_process
            )
            if exp_logger is not None:
                exp_logger.finish()
        if distributed:
            dist.barrier()
        cleanup_distributed()
        return
    
    # 训练循环
    logger.info("Starting training...")
    
    metrics = dict(best_metrics) if args.resume and best_metrics else {}  # 初始化metrics变量

    def _metric_scalar(metric_dict: Dict[str, Any], key: str) -> Optional[float]:
        if not isinstance(metric_dict, dict):
            return None
        try:
            value = metric_dict.get(key, None)
            if value is None:
                return None
            value = float(value)
            if not np.isfinite(value):
                return None
            return value
        except Exception:
            return None
    
    for epoch in range(start_epoch, config.training.epochs):
        # Progressive training stage update
        if progressive_scheduler is not None:
            if getattr(progressive_cfg, 'smooth_transition', False):
                stage_cfg = progressive_scheduler.get_interpolated_config(epoch)
            else:
                stage = progressive_scheduler.get_current_stage(epoch)
                stage_cfg = {
                    'resolution': stage.resolution,
                    'num_frames': stage.num_frames,
                    'num_points': stage.num_points,
                    'batch_size': stage.batch_size,
                    'learning_rate': stage.learning_rate,
                }
            stage_signature = (
                tuple(stage_cfg['resolution']),
                int(stage_cfg['num_frames']),
                int(stage_cfg['num_points']),
                int(stage_cfg['batch_size']),
                float(stage_cfg['learning_rate']),
            )
            if stage_signature != progressive_stage_signature:
                allow_lr_update = _scheduler_allows_lr_update(config, scheduler)
                if not allow_lr_update and is_main_process:
                    logger.warning("Progressive stage updated with OneCycleLR; LR/batch_size changes skipped.")
                _apply_progressive_stage(stage_cfg, config, allow_lr_batch_update=allow_lr_update)
                if allow_lr_update:
                    _set_optimizer_lr(optimizer, float(stage_cfg['learning_rate']))
                    _update_scheduler_base_lrs(scheduler, float(stage_cfg['learning_rate']))
                train_loader, val_loader = create_dataloaders(
                    config,
                    args.debug,
                    distributed=distributed,
                    rank=rank,
                    world_size=world_size,
                    is_main_process=is_main_process,
                )
                if pseudo_loader is not None:
                    pseudo_loader = create_pseudo_dataloader(
                        config,
                        args.debug,
                        distributed=distributed,
                        rank=rank,
                        world_size=world_size,
                    )
                progressive_stage_signature = stage_signature
                if is_main_process:
                    logger.info(
                        "Progressive stage update: "
                        f"resolution={stage_cfg['resolution']}, "
                        f"frames={stage_cfg['num_frames']}, "
                        f"points={stage_cfg['num_points']}, "
                        f"batch={config.training.batch_size}, "
                        f"lr={config.training.optimizer.lr:.2e}"
                    )
        # 训练
        model_unwrapped = unwrap_model(model)
        if phase_schedule:
            phase_idx, phase_cfg = _find_active_phase(phase_schedule, epoch)
            if phase_cfg is not None and phase_idx != active_phase_idx:
                phase_stats = _apply_phase_trainability(model_unwrapped, phase_cfg)
                active_phase_idx = phase_idx
                if _apply_phase_lr_multipliers(optimizer, phase_cfg):
                    _set_optimizer_lr(optimizer, float(config.training.optimizer.lr))
                    _update_scheduler_base_lrs(scheduler, float(config.training.optimizer.lr))
                if is_main_process:
                    phase_record = {
                        "timestamp": datetime.now().isoformat(),
                        "event": "phase_change",
                        "epoch": int(epoch),
                        "phase_idx": int(phase_idx),
                        "phase_name": str(phase_cfg.get("name", f"phase_{phase_idx}")),
                        "start_epoch": int(phase_cfg.get("start_epoch", 0)),
                        "freeze_modules": _normalize_name_list(phase_cfg.get("freeze_modules")),
                        "unfreeze_modules": _normalize_name_list(phase_cfg.get("unfreeze_modules")),
                        "trainable_modules": _normalize_name_list(phase_cfg.get("trainable_modules")),
                        "trainable_params": int(phase_stats["trainable_params"]),
                        "total_params": int(phase_stats["total_params"]),
                    }
                    _append_jsonl(phase_log_path, phase_record)
                    logger.info(
                        "Applied phase schedule: "
                        f"epoch={epoch} phase={phase_record['phase_name']} "
                        f"trainable={phase_record['trainable_params']:,}/{phase_record['total_params']:,}"
                    )
                    if exp_logger is not None:
                        ratio = (
                            float(phase_stats["trainable_params"]) / float(phase_stats["total_params"])
                            if phase_stats["total_params"] > 0
                            else 0.0
                        )
                        exp_logger.log_metrics(
                            {
                                "phase_idx": int(phase_idx),
                                "phase_trainable_params": int(phase_stats["trainable_params"]),
                                "phase_trainable_ratio": float(ratio),
                            },
                            step=epoch,
                            prefix="phase/",
                        )

        # Optional: staged freeze/unfreeze schedule (Route A refiner).
        model_unwrapped = unwrap_model(model)
        if refiner_stage_logging_enabled and hasattr(model_unwrapped, "apply_refiner_stage"):
            stage_before = getattr(model_unwrapped, "_active_stage_idx", None)
            try:
                model_unwrapped.apply_refiner_stage(epoch)
            except Exception as exc:
                if is_main_process:
                    logger.warning(f"Failed to apply refiner stage at epoch {epoch}: {exc}")
            stage_after = getattr(model_unwrapped, "_active_stage_idx", None)

            if is_main_process:
                total_params, trainable_params = _count_parameters(model_unwrapped)
                trainable_ratio = float(trainable_params) / float(total_params) if total_params > 0 else 0.0
                stage_idx_scalar = int(stage_after) if stage_after is not None else -1

                stage_metrics = {
                    "refiner_stage_idx": stage_idx_scalar,
                    "trainable_params": int(trainable_params),
                    "trainable_ratio": float(trainable_ratio),
                }
                if exp_logger is not None:
                    exp_logger.log_metrics(stage_metrics, step=epoch, prefix="stage/")

                stage_cfg = _get_refiner_stage_cfg(config, stage_after)
                frozen = getattr(model_unwrapped, "_frozen_submodules", None)
                frozen_list = sorted(list(frozen)) if isinstance(frozen, set) else None
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "epoch": int(epoch),
                    "stage_idx": stage_idx_scalar,
                    "trainable_params": int(trainable_params),
                    "total_params": int(total_params),
                    "trainable_ratio": float(trainable_ratio),
                    "trainable_modules_cfg": (
                        [str(v) for v in list(stage_cfg.get("trainable_modules", []))]
                        if isinstance(stage_cfg, dict) and stage_cfg.get("trainable_modules", None) is not None
                        else None
                    ),
                    "frozen_submodules": frozen_list,
                }
                _append_jsonl(stage_log_path, record)

                if stage_before != stage_after:
                    record_transition = dict(record)
                    record_transition["event"] = "stage_change"
                    record_transition["prev_stage_idx"] = int(stage_before) if stage_before is not None else None
                    _append_jsonl(stage_transition_path, record_transition)

        if distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if distributed and pseudo_loader is not None and hasattr(pseudo_loader.sampler, 'set_epoch'):
            pseudo_loader.sampler.set_epoch(epoch)
        if distributed and val_loader is not None and hasattr(val_loader.sampler, 'set_epoch'):
            val_loader.sampler.set_epoch(epoch)
        data_loading_cfg = getattr(config.training, 'data_loading', None)
        use_prefetch = True
        if data_loading_cfg is not None and hasattr(data_loading_cfg, 'use_prefetch'):
            use_prefetch = bool(data_loading_cfg.use_prefetch)
        train_loss, train_focus_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, epoch, config, exp_logger, device,
            is_main_process=is_main_process,
            use_prefetch=use_prefetch,
            ema=ema,
            pseudo_trainer=pseudo_trainer,
            pseudo_loader=pseudo_loader,
            pseudo_cfg=pseudo_cfg,
        )
        
        if is_main_process:
            logger.info(
                "Epoch %s: train_loss=%.4f, focus_mask_rate=%.4f, focus_active_batch_rate=%.4f",
                epoch,
                train_loss,
                float(train_focus_stats.get("position_focus_mask_rate_mean", 0.0) or 0.0),
                float(train_focus_stats.get("position_focus_active_batch_rate", 0.0) or 0.0),
            )
            if exp_logger is not None:
                exp_logger.log_metrics(
                    {
                        'epoch_loss': train_loss,
                        **train_focus_stats,
                    },
                    step=epoch,
                    prefix='train/',
                )
            try:
                model_unwrapped = unwrap_model(model)
                stage_idx = getattr(model_unwrapped, "_active_stage_idx", None)
                stage_idx_scalar = int(stage_idx) if stage_idx is not None else -1
                stage_cfg = _get_refiner_stage_cfg(config, stage_idx)
                lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
                train_record = {
                    "timestamp": datetime.now().isoformat(),
                    "event": "train_epoch_end",
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "position_focus_supervision_loss_mean": float(train_focus_stats.get("position_focus_supervision_loss_mean", 0.0) or 0.0),
                    "position_focus_mask_rate_mean": float(train_focus_stats.get("position_focus_mask_rate_mean", 0.0) or 0.0),
                    "position_focus_active_batch_rate": float(train_focus_stats.get("position_focus_active_batch_rate", 0.0) or 0.0),
                    "position_focus_nonzero_batch_rate": float(train_focus_stats.get("position_focus_nonzero_batch_rate", 0.0) or 0.0),
                    "position_focus_num_points_mean": float(train_focus_stats.get("position_focus_num_points_mean", 0.0) or 0.0),
                    "lr": float(lrs[0]) if lrs else float(config.training.optimizer.lr),
                    "lrs": lrs,
                    "stage_idx": stage_idx_scalar,
                    "stage_trainable_modules": (
                        [str(v) for v in list(stage_cfg.get("trainable_modules", []))]
                        if isinstance(stage_cfg, dict) and stage_cfg.get("trainable_modules", None) is not None
                        else None
                    ),
                    "data": {
                        "resolution": (
                            list(config.data.train.resolution)
                            if hasattr(config, "data") and hasattr(config.data, "train") and hasattr(config.data.train, "resolution")
                            else None
                        ),
                        "num_frames": int(config.data.train.num_frames) if hasattr(config.data.train, "num_frames") else None,
                        "num_points": int(config.data.train.num_points) if hasattr(config.data.train, "num_points") and config.data.train.num_points is not None else None,
                        "batch_size": int(config.training.batch_size),
                        "accumulation_steps": int(config.training.gradient.accumulation_steps),
                    },
                }
                _append_jsonl(epoch_log_path, train_record)
            except Exception as exc:
                logger.warning(f"Failed to log epoch metrics: {exc}")
        
        # 非OneCycleLR的scheduler在epoch结束后更新
        if scheduler is not None and _get_scheduler_type(config) != 'onecyclelr':
            scheduler.step()
        
        # 评估
        should_eval = (epoch == config.training.epochs - 1)
        if eval_every is not None:
            should_eval = (epoch % eval_every == 0) or should_eval
        if should_eval:
            plateau_guard_stop = False
            plateau_guard_reason = ""
            deadline_guard_stop = False
            deadline_guard_reason = ""
            if distributed:
                dist.barrier()
            if is_main_process:
                eval_model = ema.ema_model if (ema is not None and use_ema_for_eval) else unwrap_model(model)
                metrics = evaluate(
                    eval_model, val_loader, epoch, config, exp_logger, is_main_process=is_main_process
                )
                maybe_visualize_predictions(eval_model, val_loader, epoch, config, output_dir)
                try:
                    model_unwrapped = unwrap_model(model)
                    stage_idx = getattr(model_unwrapped, "_active_stage_idx", None)
                    stage_idx_scalar = int(stage_idx) if stage_idx is not None else -1
                    eval_record = {
                        "timestamp": datetime.now().isoformat(),
                        "event": "val_epoch_end",
                        "epoch": int(epoch),
                        "stage_idx": stage_idx_scalar,
                        "metrics": dict(metrics) if isinstance(metrics, dict) else {},
                    }
                    _append_jsonl(epoch_log_path, eval_record)
                except Exception as exc:
                    logger.warning(f"Failed to log val metrics: {exc}")
                
                # 保存最佳模型
                if metrics and monitor_key in metrics:
                    current_score = metrics[monitor_key]
                    is_better = current_score > best_score if monitor_mode == 'max' else current_score < best_score
                    if is_better:
                        best_score = current_score
                        best_metrics = metrics
                        save_checkpoint(
                            model, optimizer, scheduler, epoch, metrics, config,
                            checkpoint_dir / 'best.pth', ema=ema, early_stopping=early_stopping,
                            is_main_process=is_main_process
                        )

                if metrics:
                    aj_score = _metric_scalar(metrics, 'AJ')
                    oa_score = _metric_scalar(metrics, 'OA')

                    if save_best_aj and aj_score is not None and aj_score > best_aj_score:
                        best_aj_score = aj_score
                        save_checkpoint(
                            model, optimizer, scheduler, epoch, metrics, config,
                            checkpoint_dir / 'best_aj.pth', ema=ema, early_stopping=early_stopping,
                            is_main_process=is_main_process
                        )

                    if save_best_oa and oa_score is not None and oa_score > best_oa_score:
                        best_oa_score = oa_score
                        save_checkpoint(
                            model, optimizer, scheduler, epoch, metrics, config,
                            checkpoint_dir / 'best_oa.pth', ema=ema, early_stopping=early_stopping,
                            is_main_process=is_main_process
                        )

                    if plateau_enabled and epoch >= plateau_start_epoch:
                        # Route A: if we have base/refined deltas, the most useful signal
                        # for "stop wasting GPU time" is whether we *beat the base*.
                        # Prefer *_delta metrics when present, otherwise fall back to
                        # absolute metrics.
                        aj_plateau_key = 'AJ_delta' if 'AJ_delta' in metrics else 'AJ'
                        four_px_plateau_key = '<4px_delta' if '<4px_delta' in metrics else '<4px'
                        avg_error_plateau_key = (
                            'avg_error_px_delta' if 'avg_error_px_delta' in metrics else 'avg_error_px'
                        )

                        aj_score_for_plateau = _metric_scalar(metrics, aj_plateau_key)
                        oa_plateau_key = 'OA_delta' if 'OA_delta' in metrics else 'OA'
                        oa_score_for_plateau = _metric_scalar(metrics, oa_plateau_key)
                        four_px_score = _metric_scalar(metrics, four_px_plateau_key)
                        if four_px_score is None:
                            alt_key = (
                                'pts_within_4_delta'
                                if 'pts_within_4_delta' in metrics
                                else 'pts_within_4'
                            )
                            four_px_score = _metric_scalar(metrics, alt_key)
                        avg_error_px = _metric_scalar(metrics, avg_error_plateau_key)

                        if aj_score_for_plateau is not None:
                            if aj_score_for_plateau > plateau_best_aj + plateau_min_delta_aj:
                                plateau_best_aj = aj_score_for_plateau
                                plateau_bad_aj = 0
                            else:
                                plateau_bad_aj += 1

                        if oa_score_for_plateau is not None:
                            if oa_score_for_plateau > plateau_best_oa + plateau_min_delta_oa:
                                plateau_best_oa = oa_score_for_plateau
                                plateau_bad_oa = 0
                            else:
                                plateau_bad_oa += 1

                        if four_px_score is not None:
                            if four_px_score > plateau_best_4px + plateau_min_delta_4px:
                                plateau_best_4px = four_px_score
                                plateau_bad_4px = 0
                            else:
                                plateau_bad_4px += 1

                        if avg_error_px is not None:
                            if plateau_last_avg_error is not None and avg_error_px > plateau_last_avg_error + plateau_min_delta_error:
                                plateau_rising_error += 1
                            else:
                                plateau_rising_error = 0
                            plateau_last_avg_error = avg_error_px

                        if (
                            aj_score_for_plateau is not None
                            and oa_score_for_plateau is not None
                            and four_px_score is not None
                            and avg_error_px is not None
                            and plateau_bad_aj >= plateau_patience
                            and plateau_bad_oa >= plateau_patience
                            and plateau_bad_4px >= plateau_patience
                            and plateau_rising_error >= plateau_patience
                        ):
                            plateau_guard_stop = True
                            plateau_guard_reason = (
                                "Plateau guard triggered: "
                                f"{aj_plateau_key} no-improve={plateau_bad_aj}, {four_px_plateau_key} no-improve={plateau_bad_4px}, "
                                f"avg_error rising={plateau_rising_error}"
                            )
                            logger.warning(plateau_guard_reason)

                    if deadline_enabled and metrics and deadline_monitor_key in metrics:
                        score = _metric_scalar(metrics, deadline_monitor_key)
                        if score is not None:
                            is_better = (
                                score > deadline_best_score
                                if deadline_mode == 'max'
                                else score < deadline_best_score
                            )
                            if is_better:
                                deadline_best_score = score

                        if epoch >= deadline_epoch:
                            unmet = (
                                deadline_best_score < deadline_threshold
                                if deadline_mode == 'max'
                                else deadline_best_score > deadline_threshold
                            )
                            if unmet:
                                deadline_guard_stop = True
                                deadline_guard_reason = (
                                    "Deadline guard triggered: "
                                    f"best {deadline_monitor_key}={deadline_best_score:.6f} "
                                    f"did not reach {deadline_threshold:.6f} by epoch {deadline_epoch}"
                                )
                                logger.warning(deadline_guard_reason)
            if distributed:
                dist.barrier()
            
            # 早停检查
            stop_training = False
            if is_main_process and early_stopping is not None and metrics and monitor_key in metrics:
                eval_ref_model = ema.ema_model if (ema is not None and use_ema_for_eval) else unwrap_model(model)
                stop_training = early_stopping(metrics[monitor_key], eval_ref_model, epoch)
            if is_main_process and plateau_guard_stop:
                stop_training = True
            if is_main_process and deadline_guard_stop:
                stop_training = True
            
            if distributed:
                stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            
            if stop_training:
                if is_main_process and deadline_guard_stop and deadline_guard_reason:
                    logger.info(f"Stopping training due to deadline guard. {deadline_guard_reason}")
                elif is_main_process and plateau_guard_stop and plateau_guard_reason:
                    logger.info(f"Stopping training due to plateau guard. {plateau_guard_reason}")
                else:
                    logger.info("Early stopping triggered, exiting training loop.")
                break
            if distributed:
                dist.barrier()
        # 定期保存
        if is_main_process and save_every > 0 and (epoch + 1) % save_every == 0:
            metrics_to_save = dict(metrics) if isinstance(metrics, dict) else {}
            metrics_to_save['train_loss'] = float(train_loss)
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics_to_save, config,
                checkpoint_dir / f'epoch_{epoch}.pth', ema=ema, early_stopping=early_stopping,
                is_main_process=is_main_process
            )
            _prune_checkpoints(checkpoint_dir, keep_last)
        
        # 保存最新
        if is_main_process:
            metrics_to_save = dict(metrics) if isinstance(metrics, dict) else {}
            metrics_to_save['train_loss'] = float(train_loss)
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics_to_save, config,
                checkpoint_dir / 'latest.pth', ema=ema, early_stopping=early_stopping,
                is_main_process=is_main_process
            )
    
    # 清理
    if exp_logger is not None:
        exp_logger.finish()
    
    if is_main_process:
        logger.info("Training complete!")
        if best_metrics and monitor_key in best_metrics:
            logger.info(f"Best {monitor_key}: {best_metrics[monitor_key]:.4f}")
    cleanup_distributed()


if __name__ == '__main__':
    main()
