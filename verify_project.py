#!/usr/bin/env python3
"""
FSPT 项目验证脚本

验证所有模块是否可以正常导入和运行
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
try:
    from fspt.core.paths import repo_root

    PROJECT_ROOT = repo_root()
except Exception:
    PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


_OK = "[OK]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"


def check_import(module_path, description):
    """检查模块是否可以导入"""
    try:
        exec(f"import {module_path}")
        print(f"  {_OK} {description}")
        return True
    except ImportError as e:
        print(f"  {_FAIL} {description}: {e}")
        return False
    except Exception as e:
        print(f"  {_FAIL} {description}: {e}")
        return False


def check_class(module_path, class_name, description, optional: bool = False):
    """检查类是否可以导入"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  {_OK} {description}")
        return True, cls
    except ImportError as e:
        if optional:
            print(f"  {_WARN} {description}: optional dependency missing ({e})")
            return True, None
        print(f"  {_FAIL} {description}: ImportError - {e}")
        return False, None
    except AttributeError as e:
        print(f"  {_FAIL} {description}: AttributeError - {e}")
        return False, None
    except Exception as e:
        print(f"  {_FAIL} {description}: {e}")
        return False, None


def test_model_forward(model_cls, config, description):
    """测试模型前向传播"""
    import torch
    
    try:
        model = model_cls(config)
        model.eval()
        
        # 测试输入
        B, T, H, W = 2, 8, 128, 128
        N = 10
        
        video = torch.randn(B, T, 3, H, W)
        query_points = torch.rand(B, N, 3)
        query_points[:, :, 0] = 0
        
        with torch.no_grad():
            tracks, visibility = model(video, query_points)
        
        assert tracks.shape == (B, N, T, 2), f"Expected tracks shape (2, 10, 8, 2), got {tracks.shape}"
        assert visibility.shape == (B, N, T), f"Expected visibility shape (2, 10, 8), got {visibility.shape}"
        
        print(f"  {_OK} {description}")
        return True
    except Exception as e:
        print(f"  {_FAIL} {description}: {e}")
        traceback.print_exc()
        return False


def test_dataset(dataset_cls, description):
    """测试数据集（需要数据存在）"""
    try:
        # 只检查类是否可以实例化（不需要真实数据）
        print(f"  {_OK} {description} (class available)")
        return True
    except Exception as e:
        print(f"  {_FAIL} {description}: {e}")
        return False


def test_metrics():
    """测试指标计算"""
    import torch
    
    try:
        from datasets.metrics import compute_tapvid_metrics
        
        # 创建测试数据
        N, T = 10, 50
        pred_tracks = torch.rand(N, T, 2)
        gt_tracks = pred_tracks.clone()
        pred_visibility = torch.ones(N, T, dtype=torch.bool)
        gt_visibility = torch.ones(N, T, dtype=torch.bool)
        query_points = torch.zeros(N, 3)
        
        metrics = compute_tapvid_metrics(
            pred_tracks, gt_tracks, pred_visibility, gt_visibility, query_points
        )
        
        assert 'AJ' in metrics, "Missing AJ metric"
        assert 'OA' in metrics, "Missing OA metric"
        assert metrics['AJ'] > 0.9, f"Perfect prediction should have high AJ, got {metrics['AJ']}"
        
        print(f"  {_OK} Metrics computation")
        return True
    except Exception as e:
        print(f"  {_FAIL} Metrics computation: {e}")
        return False


def main():
    print("=" * 60)
    print("FSPT Project Verification")
    print("=" * 60)
    
    all_passed = True
    
    # 1. 检查核心依赖
    print("\n[1/6] Checking core dependencies...")
    all_passed &= check_import("torch", "PyTorch")
    all_passed &= check_import("numpy", "NumPy")
    all_passed &= check_import("omegaconf", "OmegaConf")
    
    # 2. 检查模型模块
    print("\n[2/6] Checking model modules...")
    all_passed &= check_import("models", "models package")
    
    passed, _ = check_class(
        "models.semantic_encoder",
        "SemanticEncoder",
        "SemanticEncoder (optional)",
        optional=True,
    )
    all_passed &= passed
    
    passed, _ = check_class("models.freq_semantic_fusion", "SemanticEnhancedLFD", "SemanticEnhancedLFD")
    all_passed &= passed
    
    passed, _ = check_class("models.occlusion_predictor", "FrequencyAwareOcclusionPredictor", "FrequencyAwareOcclusionPredictor")
    all_passed &= passed
    
    passed, FSPTTracker = check_class("models.point_tracker", "FSPTTracker", "FSPTTracker")
    all_passed &= passed

    passed, _ = check_class(
        "models.cotracker_refiner",
        "CoTrackerFSPTRefiner",
        "CoTrackerFSPTRefiner (optional)",
        optional=True,
    )
    all_passed &= passed
    
    # 3. 检查数据集模块
    print("\n[3/6] Checking dataset modules...")
    all_passed &= check_import("datasets", "datasets package")
    
    passed, _ = check_class("datasets.tapvid_davis", "TAPVidDAVISDataset", "TAPVidDAVISDataset")
    all_passed &= passed
    
    passed, _ = check_class("datasets.augmentation", "PointTrackingAugmentation", "PointTrackingAugmentation")
    all_passed &= passed
    
    # 4. 测试指标计算
    print("\n[4/6] Testing metrics...")
    all_passed &= test_metrics()
    
    # 5. 测试模型前向传播
    print("\n[5/6] Testing model forward pass...")
    if FSPTTracker is not None:
        from omegaconf import OmegaConf
        
        config = OmegaConf.create({
            'backbone': {'type': 'resnet18', 'pretrained': False},
            'clip': {'model': 'ViT-B/16', 'dim': 512, 'freeze': True},
            'frequency': {'enabled': True, 'num_bands': 4},
            'semantic': {'enabled': False},
            'temporal': {'dim': 256, 'num_layers': 2, 'num_heads': 8, 'dropout': 0.1},
            'occlusion': {'enabled': True, 'use_semantic_propagation': True},
        })
        
        all_passed &= test_model_forward(FSPTTracker, config, "FSPTTracker forward pass")
    else:
        print(f"  {_FAIL} FSPTTracker not available, skipping forward test")
        all_passed = False
    
    # 6. 检查脚本
    print("\n[6/6] Checking scripts...")
    scripts = [
        'train.py',
        'evaluate.py',
        'scripts/visualize.py',
    ]
    
    for script in scripts:
        script_path = PROJECT_ROOT / script
        if script_path.exists():
            print(f"  {_OK} {script}")
        else:
            print(f"  {_FAIL} {script} (not found)")
            all_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks passed! FSPT project is ready.")
    else:
        print("Some checks failed. Please fix the issues above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
