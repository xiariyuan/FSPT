#!/usr/bin/env python3
"""
FSPT 完整实验运行脚本

用于批量运行所有实验，确保顶会标准的完整性。

使用方法:
    # 运行所有实验
    python scripts/run_all_experiments.py --all
    
    # 运行特定实验组
    python scripts/run_all_experiments.py --baselines
    python scripts/run_all_experiments.py --ablations
    python scripts/run_all_experiments.py --main
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 实验配置
# ============================================================================

LIST_ARGS_AS_NARGS = {'datasets'}
PATH_ARGS = {
    'checkpoint',
    'config',
    'output_dir',
    'checkpoint_dir',
    'log_dir',
    'paths.output_dir',
    'paths.checkpoint_dir',
    'logging.log_dir',
    'paths.pretrained.fspt',
    'paths.pretrained.clip',
}

EXPERIMENTS = {
    # ========== 基线复现 ==========
    'baseline_tapir': {
        'type': 'baseline',
        'script': 'baselines/tapir/evaluation.py',
        'optional': True,
        'args': {
            'checkpoint': 'baselines/tapir/checkpoints/tapir_checkpoint.npy',
            'datasets': ['davis', 'kinetics'],
        },
        'gpu': 1,
        'time': '2h',
    },
    'baseline_cotracker3': {
        'type': 'baseline',
        'script': 'baselines/cotracker/scripts/evaluate_tap.py',
        'optional': True,
        'args': {
            'checkpoint': 'baselines/cotracker/checkpoints/scaled_offline.pth',
            'datasets': ['davis', 'kinetics'],
        },
        'gpu': 1,
        'time': '2h',
    },
    'baseline_locotrack': {
        'type': 'baseline',
        'script': 'baselines/locotrack/evaluate.py',
        'optional': True,
        'args': {
            'checkpoint': 'baselines/locotrack/checkpoints/locotrack.pth',
            'datasets': ['davis', 'kinetics'],
        },
        'gpu': 1,
        'time': '1h',
    },
    
    # ========== FSPT主实验 ==========
    'megadepth_pretrain': {
        'type': 'main',
        'script': 'train.py',
        'config': 'configs/fspt_megadepth_pairs_pretrain.yaml',
        'args': {
            'experiment.name': 'fspt_megadepth_pairs_pretrain',
            'paths.output_dir': 'outputs',
            'paths.checkpoint_dir': 'checkpoints',
            'logging.log_dir': 'logs',
        },
        'gpu': 0,
        'time': '24h',
    },
    'megadepth_ft_eval256': {
        'type': 'main',
        'script': 'train.py',
        'config': 'configs/fspt_megadepth_to_kubric_ft_eval256.yaml',
        'args': {
            'experiment.name': 'fspt_megadepth_to_kubric_ft_eval256',
            'paths.output_dir': 'outputs',
            'paths.checkpoint_dir': 'checkpoints',
            'logging.log_dir': 'logs',
        },
        'gpu': 0,
        'time': '12h',
        'depends_on': ['megadepth_pretrain'],
    },
    'megadepth_ft_eval256_baseline': {
        'type': 'main',
        'script': 'train.py',
        'config': 'configs/fspt_megadepth_to_kubric_ft_eval256_baseline.yaml',
        'args': {
            'experiment.name': 'fspt_megadepth_to_kubric_ft_eval256_baseline',
            'paths.output_dir': 'outputs',
            'paths.checkpoint_dir': 'checkpoints',
            'logging.log_dir': 'logs',
        },
        'gpu': 0,
        'time': '12h',
    },
    
    # ========== 核心消融实验 ==========
    'ablation_no_freq': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_no_freq',
            'model.frequency.enabled': False,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_no_semantic': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_no_semantic',
            'model.semantic.enabled': False,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_no_occlusion': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_no_occlusion',
            'model.occlusion.enabled': False,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_freq_only': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_freq_only',
            'model.frequency.enabled': True,
            'model.semantic.enabled': False,
            'model.occlusion.enabled': False,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_semantic_only': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_semantic_only',
            'model.frequency.enabled': False,
            'model.semantic.enabled': True,
            'model.occlusion.enabled': False,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_temporal_consistency': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_temporal_consistency',
            'loss.temporal_consistency.enabled': True,
            'loss.temporal_smooth_weight': 0.0,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_multi_iteration': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_multi_iteration',
            'loss.multi_iteration.enabled': True,
        },
        'gpu': 4,
        'time': '20h',
    },
    'ablation_confidence_weighted': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_confidence_weighted',
            'loss.confidence_weighted.enabled': True,
        },
        'gpu': 4,
        'time': '20h',
    },
    
    # ========== 频带数量消融 ==========
    'ablation_bands_2': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_bands_2',
            'model.frequency.num_bands': 2,
        },
        'gpu': 4,
        'time': '18h',
    },
    'ablation_bands_3': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_bands_3',
            'model.frequency.num_bands': 3,
        },
        'gpu': 4,
        'time': '19h',
    },
    'ablation_bands_6': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_bands_6',
            'model.frequency.num_bands': 6,
        },
        'gpu': 4,
        'time': '22h',
    },
    'ablation_bands_8': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_bands_8',
            'model.frequency.num_bands': 8,
        },
        'gpu': 4,
        'time': '24h',
    },
    
    # ========== CLIP模型消融 ==========
    'ablation_clip_vit_b32': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_clip_vit_b32',
            'model.clip.model': 'ViT-B/32',
        },
        'gpu': 4,
        'time': '18h',
    },
    'ablation_clip_vit_l14': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_clip_vit_l14',
            'model.clip.model': 'ViT-L/14',
        },
        'gpu': 4,
        'time': '28h',
    },
    
    # ========== 骨干网络消融 ==========
    'ablation_backbone_resnet18': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_backbone_resnet18',
            'model.backbone.type': 'resnet18',
        },
        'gpu': 4,
        'time': '16h',
    },
    'ablation_backbone_resnet101': {
        'type': 'ablation',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {
            'experiment.name': 'ablation_backbone_resnet101',
            'model.backbone.type': 'resnet101',
        },
        'gpu': 4,
        'time': '26h',
    },
    
    # ========== 多次运行（统计显著性） ==========
    'fspt_seed_42': {
        'type': 'multirun',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {'experiment.seed': 42, 'experiment.name': 'fspt_seed_42'},
        'gpu': 4,
        'time': '24h',
    },
    'fspt_seed_123': {
        'type': 'multirun',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {'experiment.seed': 123, 'experiment.name': 'fspt_seed_123'},
        'gpu': 4,
        'time': '24h',
    },
    'fspt_seed_456': {
        'type': 'multirun',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {'experiment.seed': 456, 'experiment.name': 'fspt_seed_456'},
        'gpu': 4,
        'time': '24h',
    },
    'fspt_seed_789': {
        'type': 'multirun',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {'experiment.seed': 789, 'experiment.name': 'fspt_seed_789'},
        'gpu': 4,
        'time': '24h',
    },
    'fspt_seed_1024': {
        'type': 'multirun',
        'script': 'train.py',
        'config': 'configs/fspt_base.yaml',
        'args': {'experiment.seed': 1024, 'experiment.name': 'fspt_seed_1024'},
        'gpu': 4,
        'time': '24h',
    },
}


def build_command(experiment):
    """构建实验命令"""
    script_path = Path(experiment['script'])
    if not script_path.is_absolute():
        script_path = PROJECT_ROOT / script_path
    script_path = script_path.resolve()
    cmd = [sys.executable, str(script_path)]
    use_ddp = False
    ddp_world_size = 1
    if experiment.get('script') == 'train.py':
        gpu = experiment.get('gpu')
        if isinstance(gpu, int):
            ddp_world_size = int(gpu)
        elif isinstance(gpu, (list, tuple)):
            ddp_world_size = len(gpu)
        use_ddp = ddp_world_size > 1 and bool(experiment.get('use_ddp', True))
        if use_ddp:
            ddp_script = (PROJECT_ROOT / "scripts" / "train_distributed.py").resolve()
            cmd = ["torchrun", f"--nproc_per_node={ddp_world_size}", str(ddp_script)]
    
    if 'config' in experiment:
        config_path = Path(experiment['config'])
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        cmd.extend(['--config', str(config_path)])
    
    args = dict(experiment.get('args', {}))
    # 如果是训练脚本且指定了多GPU（非DDP），显式覆盖 hardware.gpus
    if experiment.get('script') == 'train.py' and not use_ddp and 'hardware.gpus' not in args:
        gpu = experiment.get('gpu')
        if isinstance(gpu, int) and gpu > 1:
            args['hardware.gpus'] = f"[{','.join(str(i) for i in range(gpu))}]"
        elif isinstance(gpu, (list, tuple)) and len(gpu) > 1:
            args['hardware.gpus'] = f"[{','.join(str(i) for i in gpu)}]"

    for key, value in args.items():
        if isinstance(value, bool):
            value = str(value).lower()
        elif isinstance(value, list):
            if key in LIST_ARGS_AS_NARGS:
                cmd.append(f'--{key}')
                cmd.extend([str(v) for v in value])
            else:
                for v in value:
                    cmd.extend([f'--{key}', str(v)])
            continue
        if isinstance(value, str) and key in PATH_ARGS:
            value_path = Path(value)
            if not value_path.is_absolute():
                value = str(PROJECT_ROOT / value_path)
        cmd.extend([f'--{key}', str(value)])
    
    return cmd


def run_experiment(name, experiment, dry_run=False):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Type: {experiment['type']}")
    print(f"Estimated time: {experiment.get('time', 'unknown')}")
    print(f"{'='*60}")

    script_path = Path(experiment['script'])
    if not script_path.is_absolute():
        script_path = PROJECT_ROOT / script_path
    script_path = script_path.resolve()
    if not script_path.exists():
        print(f"Warning: script not found: {script_path}")
        if experiment.get('optional', False):
            print("Skipping optional experiment.")
            return None
        return False
    if 'config' in experiment:
        config_path = Path(experiment['config'])
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        config_path = config_path.resolve()
        if not config_path.exists():
            print(f"Warning: config not found: {config_path}")
            if experiment.get('optional', False):
                print("Skipping optional experiment.")
                return None
            return False
    
    cmd = build_command(experiment)
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("(dry run, not executing)")
        return True
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 运行
    try:
        env = os.environ.copy()
        if 'gpu' in experiment:
            gpu = experiment['gpu']
            if isinstance(gpu, (list, tuple)):
                env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu)
            elif isinstance(gpu, int):
                if gpu > 0:
                    if gpu == 1:
                        env['CUDA_VISIBLE_DEVICES'] = '0'
                    else:
                        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu))
            elif isinstance(gpu, str):
                env['CUDA_VISIBLE_DEVICES'] = gpu
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
        )
        success = True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        success = False
    
    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nCompleted: {name}")
    print(f"Duration: {duration}")
    print(f"Status: {'SUCCESS' if success else 'FAILED'}")
    
    return success


def run_experiments(experiment_names, dry_run=False):
    """运行多个实验"""
    results = {}
    
    for name in experiment_names:
        if name not in EXPERIMENTS:
            print(f"Warning: Unknown experiment '{name}'")
            continue
        
        experiment = EXPERIMENTS[name]
        
        # 检查依赖
        depends_on = experiment.get('depends_on', [])
        skip = False
        for dep in depends_on:
            if results.get(dep) is not True:
                print(f"Skipping {name}: dependency {dep} not completed")
                skip = True
                break
        if skip:
            results[name] = None
            continue
        
        results[name] = run_experiment(name, experiment, dry_run)
    
    return results


def get_experiments_by_type(exp_type):
    """获取特定类型的实验"""
    return [name for name, exp in EXPERIMENTS.items() if exp['type'] == exp_type]


def print_experiment_summary():
    """打印实验摘要"""
    print("\n" + "="*60)
    print("FSPT Experiment Summary")
    print("="*60)
    
    by_type = {}
    for name, exp in EXPERIMENTS.items():
        exp_type = exp['type']
        if exp_type not in by_type:
            by_type[exp_type] = []
        by_type[exp_type].append((name, exp))
    
    for exp_type, exps in by_type.items():
        print(f"\n{exp_type.upper()} ({len(exps)} experiments)")
        print("-" * 40)
        for name, exp in exps:
            print(f"  {name}: {exp.get('time', 'N/A')}")
    
    total_time = sum(
        int(exp.get('time', '0h').rstrip('h'))
        for exp in EXPERIMENTS.values()
    )
    print(f"\nTotal estimated GPU time: ~{total_time}h")


def main():
    parser = argparse.ArgumentParser(description='Run FSPT experiments')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--baselines', action='store_true', help='Run baseline experiments')
    parser.add_argument('--main', action='store_true', help='Run main experiments')
    parser.add_argument('--ablations', action='store_true', help='Run ablation experiments')
    parser.add_argument('--multirun', action='store_true', help='Run multi-seed experiments')
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to run')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--list', action='store_true', help='List all experiments')
    args = parser.parse_args()
    
    if args.list:
        print_experiment_summary()
        return
    
    experiments_to_run = []
    
    if args.all:
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        if args.baselines:
            experiments_to_run.extend(get_experiments_by_type('baseline'))
        if args.main:
            experiments_to_run.extend([
                'megadepth_pretrain',
                'megadepth_ft_eval256',
                'megadepth_ft_eval256_baseline',
            ])
        if args.ablations:
            experiments_to_run.extend(get_experiments_by_type('ablation'))
        if args.multirun:
            experiments_to_run.extend(get_experiments_by_type('multirun'))
        if args.experiments:
            experiments_to_run.extend(args.experiments)
    
    if not experiments_to_run:
        parser.print_help()
        return
    
    # 去重
    experiments_to_run = list(dict.fromkeys(experiments_to_run))
    
    print(f"Experiments to run: {experiments_to_run}")
    
    results = run_experiments(experiments_to_run, args.dry_run)
    
    # 汇总
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    successful = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for name, success in results.items():
            if success is False:
                print(f"  - {name}")
    if skipped > 0:
        print("\nSkipped experiments:")
        for name, success in results.items():
            if success is None:
                print(f"  - {name}")


if __name__ == '__main__':
    main()
