#!/usr/bin/env python3
"""
消融实验批量运行脚本

自动运行所有消融实验并生成对比报告

使用方法:
    python scripts/run_ablations.py --output-dir outputs/ablations
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# 消融实验配置列表
ABLATION_CONFIGS = [
    # (配置文件, 实验名称, 描述)
    ("configs/fspt_base.yaml", "full_model", "完整FSPT模型"),
    ("configs/ablations/no_frequency.yaml", "no_frequency", "无频率分解"),
    ("configs/ablations/no_semantic.yaml", "no_semantic", "无语义编码"),
    ("configs/ablations/no_multiscale.yaml", "no_multiscale", "无多尺度融合"),
    ("configs/ablations/no_refinement.yaml", "no_refinement", "无迭代精化"),
    ("configs/ablations/refinement_5iters.yaml", "refinement_5iters", "5次迭代精化"),
    ("configs/ablations/no_occlusion.yaml", "no_occlusion", "无遮挡预测"),
    ("configs/ablations/freq_2bands.yaml", "freq_2bands", "2个频带"),
    ("configs/ablations/temporal_consistency.yaml", "temporal_consistency", "时序一致性正则"),
    ("configs/ablations/multi_iteration.yaml", "multi_iteration", "多迭代监督"),
    ("configs/ablations/confidence_weighted.yaml", "confidence_weighted", "置信度加权损失"),
]


def parse_args():
    parser = argparse.ArgumentParser(description='Run FSPT Ablation Studies')
    parser.add_argument('--output-dir', type=str, default='outputs/ablations',
                        help='Output directory for all experiments')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs per experiment')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip experiments with existing results')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Specific experiments to run (by name)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print commands without running')
    return parser.parse_args()


def run_experiment(config_path: str, experiment_name: str, output_dir: Path,
                   epochs: int, gpu: int, dry_run: bool = False) -> dict:
    """
    运行单个实验
    
    Returns:
        metrics: 实验结果指标
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return {"status": "error", "error": f"Config not found: {config_path}"}
    cmd = [
        sys.executable, str(PROJECT_ROOT / "train.py"),
        "--config", str(config_path),
        "--experiment.name", experiment_name,
        "--paths.output_dir", str(output_dir),
        "--paths.checkpoint_dir", str(output_dir / "checkpoints"),
        "--logging.log_dir", str(output_dir / "logs"),
        "--training.epochs", str(epochs),
    ]
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        return {"status": "dry_run"}

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_name}.log"
    
    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            result = subprocess.run(
                cmd,
                env=env,
                cwd=Path(__file__).parent.parent,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600 * 24,  # 24小时超时
            )
        
        if result.returncode != 0:
            logger.error(f"Experiment {experiment_name} failed!")
            logger.error(f"Logs saved to: {log_path}")
            return {"status": "failed", "error": f"See log: {log_path}"}
        
        # 读取结果（从最佳checkpoint中提取指标）
        metrics = {"status": "completed"}
        ckpt_path = output_dir / "checkpoints" / experiment_name / "best.pth"
        if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
            try:
                import torch
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                metrics.update(checkpoint.get("metrics", {}))
            except Exception as exc:
                logger.warning(f"Failed to load checkpoint {ckpt_path}: {exc}")
                metrics["note"] = f"Checkpoint load failed: {exc}"
        else:
            metrics["note"] = "No best checkpoint found"
        
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment {experiment_name} timed out!")
        return {"status": "timeout"}
    except KeyboardInterrupt:
        logger.warning(f"Experiment {experiment_name} interrupted by user")
        return {"status": "interrupted"}
    except Exception as e:
        logger.exception(f"Experiment {experiment_name} error")
        return {"status": "error", "error": str(e)}


def generate_report(results: dict, output_dir: Path):
    """生成消融实验对比报告"""
    if not _is_main_process():
        return None
    report_path = output_dir / "ablation_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# FSPT 消融实验报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验结果对比\n\n")
        f.write("| 实验 | 描述 | AJ | <4px | OA | 状态 |\n")
        f.write("|------|------|-----|---------|-----|------|\n")

        def _fmt_metric(metrics: dict, key: str) -> str:
            ref = metrics.get(key, None)
            base = metrics.get(f"{key}_base", None)
            delta = metrics.get(f"{key}_delta", None)

            def _fmt(value, signed: bool = False) -> str:
                if value is None:
                    return "N/A"
                if isinstance(value, float):
                    return f"{value:+.4f}" if signed else f"{value:.4f}"
                if isinstance(value, int):
                    return f"{value:+d}" if signed else str(value)
                return str(value)

            ref_s = _fmt(ref)
            if base is None and delta is None:
                return ref_s
            base_s = _fmt(base)
            delta_s = _fmt(delta, signed=True)
            return f"{ref_s} / {base_s} ({delta_s})"

        for config_path, name, desc in ABLATION_CONFIGS:
            if name in results:
                r = results[name]
                status = r.get('status', 'completed')
                if status == 'completed' or 'AJ' in r:
                    aj = _fmt_metric(r, 'AJ')
                    dx = _fmt_metric(r, '<4px')
                    oa = _fmt_metric(r, 'OA')
                    
                    f.write(f"| {name} | {desc} | {aj} | {dx} | {oa} | ✓ |\n")
                else:
                    f.write(f"| {name} | {desc} | - | - | - | {status} |\n")
            else:
                f.write(f"| {name} | {desc} | - | - | - | pending |\n")
        
        f.write("\n## 关键发现\n\n")
        f.write("（根据实验结果填写）\n\n")
        
        f.write("### 1. 频率分解的影响\n")
        f.write("对比 `full_model` 与 `no_frequency`\n\n")
        
        f.write("### 2. 语义编码的影响\n")
        f.write("对比 `full_model` 与 `no_semantic`\n\n")
        
        f.write("### 3. 多尺度融合的影响\n")
        f.write("对比 `full_model` 与 `no_multiscale`\n\n")
        
        f.write("### 4. 迭代精化的影响\n")
        f.write("对比 `full_model`, `no_refinement`, `refinement_5iters`\n\n")
        
        f.write("### 5. 遮挡预测的影响\n")
        f.write("对比 `full_model` 与 `no_occlusion`\n\n")
    
    logger.info(f"Report saved to {report_path}")
    return report_path


def _is_main_process() -> bool:
    rank = os.environ.get("RANK")
    if rank is None:
        return True
    try:
        return int(rank) == 0
    except ValueError:
        return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()
    args.output_dir = str(output_dir)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 过滤要运行的实验
    experiments_to_run = ABLATION_CONFIGS
    if args.experiments:
        experiments_to_run = [
            (c, n, d) for c, n, d in ABLATION_CONFIGS
            if n in args.experiments
        ]
    
    # 跳过已存在的实验
    if args.skip_existing:
        filtered = []
        for config, name, desc in experiments_to_run:
            ckpt_path = output_dir / "checkpoints" / name / "best.pth"
            if ckpt_path.exists() and ckpt_path.stat().st_size > 0:
                logger.info(f"Skipping {name} (best checkpoint exists)")
            else:
                filtered.append((config, name, desc))
        experiments_to_run = filtered
    
    # 运行实验
    results = {}
    
    # 尝试加载已有结果
    results_file = output_dir / "all_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load results file: {exc}")
            results = {}
    
    for config_path, name, desc in experiments_to_run:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting: {name} - {desc}")
        logger.info(f"{'='*60}\n")
        
        metrics = run_experiment(
            config_path, name, output_dir,
            args.epochs, args.gpu, args.dry_run
        )
        
        results[name] = metrics
        
        # 保存中间结果
        if _is_main_process():
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    # 生成报告
    if not args.dry_run:
        generate_report(results, output_dir)
    
    logger.info("\nAll ablation experiments completed!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()
