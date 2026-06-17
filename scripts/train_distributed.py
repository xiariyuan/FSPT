#!/usr/bin/env python3
"""
分布式训练启动脚本

使用方法:
    # 单机多卡
    torchrun --nproc_per_node=4 scripts/train_distributed.py --config configs/fspt_base.yaml
    
    # 多机多卡
    torchrun --nnodes=2 --node_rank=0 --master_addr=<master_ip> --master_port=29500 \
             --nproc_per_node=4 scripts/train_distributed.py --config configs/fspt_base.yaml
"""

import os
import sys
import argparse
import logging
import random
import math
import json
from datetime import datetime
from pathlib import Path
from copy import deepcopy

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
import numpy as np
from omegaconf import OmegaConf

from train import (
    create_model,
    create_dataloaders,
    create_pseudo_dataloader,
    create_optimizer,
    create_scheduler,
    PointTrackingLoss,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    load_pretrained,
    load_config,
    _resolve_path,
    unwrap_model,
    _prune_checkpoints,
    _set_optimizer_lr,
    _update_scheduler_base_lrs,
    _scheduler_allows_lr_update,
    _apply_progressive_stage,
    _get_amp_dtype,
    _count_parameters,
    _append_jsonl,
    _get_refiner_stage_cfg,
)
from utils.wandb_logger import create_experiment_logger
from utils.ema import create_ema
from utils.early_stopping import create_early_stopping
from utils.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_device,
    wrap_model_ddp,
    is_main_process,
    barrier,
    get_world_size,
    get_rank,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='FSPT Distributed Training')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (overrides config)')
    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    args, unknown = parse_args()
    project_root = Path(__file__).parent.parent
    if args.config and not Path(args.config).is_absolute():
        args.config = str(project_root / args.config)
    if args.config and not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        return
    if args.resume and not Path(args.resume).is_absolute():
        args.resume = str(project_root / args.resume)
    if args.output_dir and not Path(args.output_dir).is_absolute():
        args.output_dir = str(project_root / args.output_dir)
    
    # 加载配置（支持defaults继承）
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

    # 初始化分布式（使用配置中的backend/init_method）
    dist_cfg = getattr(getattr(config, 'hardware', None), 'distributed', None)
    backend = getattr(dist_cfg, 'backend', 'nccl') if dist_cfg is not None else 'nccl'
    init_method = getattr(dist_cfg, 'init_method', 'env://') if dist_cfg is not None else 'env://'
    rank, world_size, is_main = setup_distributed(backend=backend, init_method=init_method)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = get_device(local_rank)
    if not dist.is_initialized():
        logger.error("Distributed process group not initialized. Please launch with torchrun.")
        return

    # 只在主进程输出日志
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)

    logger.info(f"Process {rank}/{world_size} on device {device}")
    
    # 调整学习率（线性缩放规则）
    if world_size > 1:
        base_lr = config.training.optimizer.lr
        config.training.optimizer.lr = base_lr * world_size
        logger.info(f"Scaled learning rate: {base_lr} -> {config.training.optimizer.lr}")
    
    project_root = Path(__file__).parent.parent
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

    # 设置输出目录
    if args.output_dir:
        config.paths.output_dir = _resolve_path(args.output_dir, project_root)
        config.paths.checkpoint_dir = _resolve_path(str(Path(args.output_dir) / "checkpoints"), project_root)
        if hasattr(config, 'logging') and hasattr(config.logging, 'log_dir'):
            config.logging.log_dir = _resolve_path(str(Path(args.output_dir) / "logs"), project_root)

    output_dir = Path(config.paths.output_dir) / config.experiment.name
    checkpoint_dir = Path(config.paths.checkpoint_dir) / config.experiment.name
    
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
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # 保存配置
        OmegaConf.save(config, output_dir / 'config.yaml')

    # 实验日志（仅主进程）
    log_root = output_dir
    if hasattr(config, 'logging') and hasattr(config.logging, 'log_dir'):
        log_root = Path(config.logging.log_dir) / config.experiment.name
        log_root.mkdir(parents=True, exist_ok=True)
    exp_logger = create_experiment_logger(config, log_root, is_main_process=is_main)
    
    # 同步
    barrier()
    
    # 设置随机种子（DDP需要一致初始化）
    seed = config.experiment.seed
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
    
    # 创建模型
    model = create_model(config).to(device)

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
    
    # 包装为DDP
    find_unused = bool(getattr(dist_cfg, 'find_unused_parameters', False)) if dist_cfg is not None else False
    model = wrap_model_ddp(
        model,
        device,
        find_unused_parameters=find_unused,
    )

    if is_main and exp_logger is not None:
        wandb_cfg = getattr(getattr(config, 'logging', None), 'wandb', None)
        if wandb_cfg is not None and getattr(wandb_cfg, 'watch_model', False):
            log_freq = getattr(wandb_cfg, 'log_freq', 100)
            log_freq = int(log_freq) if log_freq is not None else 100
            if log_freq > 0:
                exp_logger.log_model_gradients(unwrap_model(model), log_freq=log_freq)
            else:
                logger.warning("wandb.log_freq <= 0; skip gradient watching.")
    
    # 创建数据加载器（训练使用分布式采样器）
    train_loader, val_loader = create_dataloaders(
        config,
        distributed=True,
        rank=rank,
        world_size=world_size,
        is_main_process=is_main,
    )
    
    # 伪标签自训练数据（可选）
    pseudo_cfg = getattr(getattr(config, 'training', None), 'pseudo_labeling', None)
    pseudo_loader = None
    pseudo_trainer = None
    if pseudo_cfg is not None and getattr(pseudo_cfg, 'enabled', False):
        pseudo_loader = create_pseudo_dataloader(
            config,
            distributed=True,
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
            )
    
    # 创建损失函数
    criterion = PointTrackingLoss(config)

    # 学习率查找器（可选）
    amp_dtype = _get_amp_dtype(config)
    amp_cfg = getattr(getattr(config, 'training', None), 'amp', None)
    use_amp = bool(amp_cfg is not None and getattr(amp_cfg, 'enabled', False) and device.type == 'cuda')
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
    if lr_finder_enabled and not args.resume:
        apply_lr = bool(getattr(lr_finder_cfg, 'apply', True))
        if is_main:
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
                use_amp=use_amp,
                amp_dtype=amp_dtype if use_amp else None,
                optimizer_kwargs=optimizer_kwargs,
            )
            if suggested_lr is not None and math.isfinite(float(suggested_lr)):
                logger.info(f"LR finder suggested lr: {float(suggested_lr):.2e}")
                if exp_logger is not None:
                    exp_logger.log_metrics({'suggested_lr': float(suggested_lr)}, step=0, prefix='lr_finder/')
            else:
                logger.warning("LR finder did not return a valid lr; skipping.")
                suggested_lr = None
        lr_value = float(suggested_lr) if suggested_lr is not None else float('nan')
        lr_tensor = torch.tensor([lr_value], device=device)
        dist.broadcast(lr_tensor, src=0)
        suggested_lr = float(lr_tensor.item())
        if not math.isfinite(suggested_lr):
            suggested_lr = None
        if apply_lr and suggested_lr is not None:
            config.training.optimizer.lr = float(suggested_lr)
            logger.info(f"Applied suggested lr: {config.training.optimizer.lr:.2e}")
            # Note: LR finder already ran with the world_size-scaled LR,
            # so the suggested LR already accounts for multi-GPU scaling.
            # Do NOT scale again.

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

        if is_main:
            OmegaConf.save(config, output_dir / 'config.yaml')
    
    # 创建优化器和调度器
    optimizer = create_optimizer(model, config)
    accum_steps = max(1, int(config.training.gradient.accumulation_steps))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum_steps))
    num_training_steps = steps_per_epoch * int(config.training.epochs)
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # 混合精度
    scaler_enabled = use_amp and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    # EMA与早停（仅主进程维护早停）
    ema = create_ema(unwrap_model(model), config, device)
    early_stopping = create_early_stopping(config) if is_main else None

    # 恢复训练
    start_epoch = 0
    monitor_cfg = getattr(config.training, 'early_stopping', None)
    monitor_key = getattr(monitor_cfg, 'monitor', 'AJ') if monitor_cfg else 'AJ'
    monitor_mode = getattr(monitor_cfg, 'mode', 'max') if monitor_cfg else 'max'
    best_score = float('-inf') if monitor_mode == 'max' else float('inf')
    best_metrics = {monitor_key: best_score}
    metrics = {}

    deadline_cfg = getattr(getattr(config, 'training', None), 'deadline_guard', None)
    deadline_enabled = bool(deadline_cfg is not None and getattr(deadline_cfg, 'enabled', False))
    deadline_epoch = int(getattr(deadline_cfg, 'deadline_epoch', 0) or 0) if deadline_cfg is not None else 0
    deadline_threshold = float(getattr(deadline_cfg, 'threshold', 0.0) or 0.0) if deadline_cfg is not None else 0.0
    deadline_monitor_key = str(getattr(deadline_cfg, 'monitor', monitor_key) or monitor_key) if deadline_cfg is not None else monitor_key
    deadline_mode = str(getattr(deadline_cfg, 'mode', monitor_mode) or monitor_mode) if deadline_cfg is not None else monitor_mode
    if deadline_mode not in ('max', 'min'):
        if is_main:
            logger.warning(
                "training.deadline_guard.mode must be 'max' or 'min'; "
                f"got {deadline_mode!r}. Falling back to {monitor_mode!r}."
            )
        deadline_mode = monitor_mode
    deadline_best_score = float('-inf') if deadline_mode == 'max' else float('inf')

    if is_main and deadline_enabled:
        logger.info(
            "Deadline guard enabled: "
            f"monitor={deadline_monitor_key}({deadline_mode}), "
            f"deadline_epoch={deadline_epoch}, threshold={deadline_threshold}"
        )
    if args.resume:
        start_epoch, loaded_metrics = load_checkpoint(
            args.resume, model, optimizer, scheduler, ema=ema, early_stopping=early_stopping
        )
        if start_epoch < 0:
            start_epoch = 0
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}")
        if loaded_metrics and monitor_key in loaded_metrics:
            best_score = loaded_metrics[monitor_key]
            best_metrics = loaded_metrics
            metrics = dict(loaded_metrics)
        if deadline_enabled:
            if isinstance(loaded_metrics, dict) and deadline_monitor_key in loaded_metrics:
                try:
                    deadline_best_score = float(loaded_metrics[deadline_monitor_key])
                except Exception:
                    pass
            if deadline_monitor_key == monitor_key:
                deadline_best_score = best_score
        if pseudo_trainer is not None:
            try:
                pseudo_trainer.teacher.load_state_dict(unwrap_model(model).state_dict(), strict=False)
                pseudo_trainer.teacher.to(device)
                pseudo_trainer.teacher.eval()
                logger.info("Synced pseudo-label teacher with resumed student weights.")
            except Exception as exc:
                logger.warning(f"Failed to sync pseudo-label teacher after resume: {exc}")

    ema_cfg = getattr(config.training, 'ema', None)
    use_ema_for_eval = bool(ema_cfg is not None and getattr(ema_cfg, 'use_for_eval', True))
    eval_cfg = getattr(config, 'evaluation', None)
    eval_every = int(getattr(eval_cfg, 'eval_every', 1) or 1)
    if eval_every <= 0:
        if is_main:
            logger.warning("evaluation.eval_every <= 0; only evaluating at final epoch.")
        eval_every = None
    checkpoint_cfg = getattr(getattr(config, 'training', None), 'checkpoint', None)
    save_every = int(getattr(checkpoint_cfg, 'save_every', 0) or 0)
    if save_every <= 0 and is_main:
        logger.warning("training.checkpoint.save_every <= 0; periodic checkpoints disabled.")
    keep_last = int(getattr(checkpoint_cfg, 'keep_last', 0) or 0)

    # ---------------------------------------------------------------------
    # Refiner stage logging (Route A / staged freeze-unfreeze schedules)
    # ---------------------------------------------------------------------
    stage_log_path = output_dir / "refiner_stage_log.jsonl"
    stage_transition_path = output_dir / "refiner_stage_transitions.jsonl"
    stage_schedule_path = output_dir / "refiner_stages.json"
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
                if isinstance(s, dict):
                    resolved_stages.append(dict(s))
                else:
                    try:
                        resolved_stages.append(dict(s))
                    except Exception:
                        continue
            refiner_stage_logging_enabled = len(resolved_stages) > 0
    except Exception:
        resolved_stages = None
        refiner_stage_logging_enabled = False

    # Fresh run: clear old epoch logs to avoid mixing experiments.
    if is_main and not args.resume and start_epoch == 0:
        try:
            if epoch_log_path.exists():
                epoch_log_path.unlink()
        except Exception:
            pass

    if is_main and refiner_stage_logging_enabled:
        if not args.resume and start_epoch == 0:
            for p in (stage_log_path, stage_transition_path):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        try:
            stage_schedule_path.parent.mkdir(parents=True, exist_ok=True)
            stage_schedule_path.write_text(
                json.dumps(
                    {
                        "experiment": str(config.experiment.name),
                        "timestamp": datetime.now().isoformat(),
                        "stages": resolved_stages,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"Failed to write refiner stage schedule: {exc}")
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
    
    # 训练循环
    num_epochs = config.training.epochs
    last_train_loss = None
    
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, num_epochs):
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
                if not allow_lr_update and is_main:
                    logger.warning("Progressive stage updated with OneCycleLR; LR/batch_size changes skipped.")
                _apply_progressive_stage(stage_cfg, config, allow_lr_batch_update=allow_lr_update)
                if allow_lr_update:
                    _set_optimizer_lr(optimizer, float(stage_cfg['learning_rate']))
                    _update_scheduler_base_lrs(scheduler, float(stage_cfg['learning_rate']))
                train_loader, val_loader = create_dataloaders(
                    config,
                    distributed=True,
                    rank=rank,
                    world_size=world_size,
                    is_main_process=is_main,
                )
                if pseudo_loader is not None:
                    pseudo_loader = create_pseudo_dataloader(
                        config,
                        distributed=True,
                        rank=rank,
                        world_size=world_size,
                    )
                progressive_stage_signature = stage_signature
                if is_main:
                    logger.info(
                        "Progressive stage update: "
                        f"resolution={stage_cfg['resolution']}, "
                        f"frames={stage_cfg['num_frames']}, "
                        f"points={stage_cfg['num_points']}, "
                        f"batch={config.training.batch_size}, "
                        f"lr={config.training.optimizer.lr:.2e}"
                    )

        # Optional: staged freeze/unfreeze schedule (Route A refiner).
        model_unwrapped = unwrap_model(model)
        if refiner_stage_logging_enabled and hasattr(model_unwrapped, "apply_refiner_stage"):
            stage_before = getattr(model_unwrapped, "_active_stage_idx", None)
            try:
                model_unwrapped.apply_refiner_stage(epoch)
            except Exception as exc:
                if is_main:
                    logger.warning(f"Failed to apply refiner stage at epoch {epoch}: {exc}")
            stage_after = getattr(model_unwrapped, "_active_stage_idx", None)
            if is_main:
                total_params, trainable_params = _count_parameters(model_unwrapped)
                trainable_ratio = float(trainable_params) / float(total_params) if total_params > 0 else 0.0
                stage_idx_scalar = int(stage_after) if stage_after is not None else -1
                if exp_logger is not None:
                    exp_logger.log_metrics(
                        {
                            "refiner_stage_idx": stage_idx_scalar,
                            "trainable_params": int(trainable_params),
                            "trainable_ratio": float(trainable_ratio),
                        },
                        step=epoch,
                        prefix="stage/",
                    )

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
        # 设置epoch用于采样器
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        if pseudo_loader is not None and hasattr(pseudo_loader.sampler, 'set_epoch'):
            pseudo_loader.sampler.set_epoch(epoch)
        data_loading_cfg = getattr(config.training, 'data_loading', None)
        use_prefetch = True
        if data_loading_cfg is not None and hasattr(data_loading_cfg, 'use_prefetch'):
            use_prefetch = bool(data_loading_cfg.use_prefetch)
        
        # 训练
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            config=config,
            exp_logger=exp_logger,
            device=device,
            is_main_process=is_main,
            use_prefetch=use_prefetch,
            ema=ema,
            pseudo_trainer=pseudo_trainer,
            pseudo_loader=pseudo_loader,
            pseudo_cfg=pseudo_cfg,
        )
        last_train_loss = train_loss
        last_epoch = epoch
        if is_main and exp_logger is not None:
            exp_logger.log_metrics({'epoch_loss': train_loss}, step=epoch, prefix='train/')
        if is_main:
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
        if scheduler is not None and _scheduler_allows_lr_update(config, scheduler):
            scheduler.step()
        
        # 验证（仅主进程评估，但所有进程同步参与）
        should_eval = (epoch == num_epochs - 1)
        if eval_every is not None:
            should_eval = (epoch % eval_every == 0) or should_eval
        if should_eval:
            deadline_guard_stop = False
            deadline_guard_reason = ""
            eval_metrics = None
            if is_main and val_loader is not None:
                eval_model = ema.ema_model if (ema is not None and use_ema_for_eval) else unwrap_model(model)
                eval_metrics = evaluate(
                    eval_model, val_loader, epoch, config, exp_logger, is_main_process=is_main
                )
                if eval_metrics:
                    metrics = eval_metrics
                try:
                    model_unwrapped = unwrap_model(model)
                    stage_idx = getattr(model_unwrapped, "_active_stage_idx", None)
                    stage_idx_scalar = int(stage_idx) if stage_idx is not None else -1
                    eval_record = {
                        "timestamp": datetime.now().isoformat(),
                        "event": "val_epoch_end",
                        "epoch": int(epoch),
                        "stage_idx": stage_idx_scalar,
                        "metrics": dict(eval_metrics) if isinstance(eval_metrics, dict) else {},
                    }
                    _append_jsonl(epoch_log_path, eval_record)
                except Exception as exc:
                    logger.warning(f"Failed to log val metrics: {exc}")
                
                if eval_metrics and monitor_key in eval_metrics:
                    logger.info(f"Epoch {epoch} validation: {monitor_key}={eval_metrics.get(monitor_key, 0):.4f}")
                    # 保存最佳模型
                    current_score = eval_metrics.get(monitor_key, 0)
                    is_better = current_score > best_score if monitor_mode == 'max' else current_score < best_score
                    if is_better:
                        best_score = current_score
                        best_metrics = eval_metrics
                        save_checkpoint(
                            model, optimizer, scheduler, epoch, eval_metrics, config,
                            checkpoint_dir / 'best.pth', ema=ema, early_stopping=early_stopping,
                            is_main_process=is_main
                        )
                if deadline_enabled and eval_metrics and deadline_monitor_key in eval_metrics:
                    score = float(eval_metrics[deadline_monitor_key])
                    is_better = score > deadline_best_score if deadline_mode == 'max' else score < deadline_best_score
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
            
            # 早停检查（广播到所有进程）
            stop_training = False
            if is_main and early_stopping is not None and eval_metrics and monitor_key in eval_metrics:
                eval_ref_model = ema.ema_model if (ema is not None and use_ema_for_eval) else unwrap_model(model)
                stop_training = early_stopping(eval_metrics[monitor_key], eval_ref_model, epoch)
            if is_main and deadline_guard_stop:
                stop_training = True
            stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            stop_training = bool(stop_tensor.item())
            if stop_training:
                if is_main and deadline_guard_stop and deadline_guard_reason:
                    logger.info(f"Stopping training due to deadline guard. {deadline_guard_reason}")
                else:
                    logger.info("Early stopping triggered, exiting training loop.")
                break
        
        # 定期保存检查点
        if is_main and save_every > 0 and (epoch + 1) % save_every == 0:
            metrics_to_save = dict(metrics) if isinstance(metrics, dict) else {}
            metrics_to_save['train_loss'] = float(train_loss)
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics_to_save, config,
                checkpoint_dir / f'epoch_{epoch}.pth', ema=ema, early_stopping=early_stopping,
                is_main_process=is_main
            )
            _prune_checkpoints(checkpoint_dir, keep_last)
        
        # 保存最新
        if is_main:
            metrics_to_save = dict(metrics) if isinstance(metrics, dict) else {}
            metrics_to_save['train_loss'] = float(train_loss)
            save_checkpoint(
                model, optimizer, scheduler, epoch, metrics_to_save, config,
                checkpoint_dir / 'latest.pth', ema=ema, early_stopping=early_stopping,
                is_main_process=is_main
            )
        
        # 同步
        barrier()
    
    # 保存最终模型
    if is_main:
        metrics_to_save = dict(metrics) if isinstance(metrics, dict) else {}
        metrics_to_save['train_loss'] = float(last_train_loss) if last_train_loss is not None else 0.0
        final_epoch = last_epoch if last_epoch >= 0 else (num_epochs - 1)
        save_checkpoint(
            model, optimizer, scheduler, final_epoch, metrics_to_save, config,
            checkpoint_dir / 'final_model.pth', ema=ema, early_stopping=early_stopping,
            is_main_process=is_main
        )
    
    # 清理
    if exp_logger is not None:
        exp_logger.finish()
    cleanup_distributed()
    
    logger.info("Training completed!")
    if is_main and best_metrics and monitor_key in best_metrics:
        logger.info(f"Best {monitor_key}: {best_metrics[monitor_key]:.4f}")


if __name__ == '__main__':
    main()
