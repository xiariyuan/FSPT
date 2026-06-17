#!/usr/bin/env python3
"""
FSPT 模型单元测试

运行方式:
    pytest tests/test_models.py -v
    python -m pytest tests/test_models.py -v
"""

import pytest

torch = pytest.importorskip("torch")
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestGeometricBackbone:
    """测试几何骨干网络"""
    
    @pytest.fixture
    def backbone(self):
        from models.point_tracker import GeometricBackbone
        return GeometricBackbone(
            backbone_type='resnet18',
            pretrained=False,
            output_dim=256,
            use_multiscale=False,
        )
    
    @pytest.fixture
    def backbone_multiscale(self):
        from models.point_tracker import GeometricBackbone
        return GeometricBackbone(
            backbone_type='resnet18',
            pretrained=False,
            output_dim=256,
            use_multiscale=True,
        )
    
    def test_single_image_forward(self, backbone):
        """测试单张图像前向传播"""
        x = torch.randn(2, 3, 256, 256)
        out = backbone(x)
        assert out.dim() == 4  # (B, H', W', C)
        assert out.shape[-1] == 256  # output_dim
        
    def test_video_forward(self, backbone):
        """测试视频前向传播"""
        x = torch.randn(2, 8, 3, 256, 256)
        out = backbone(x)
        assert out.dim() == 5  # (B, T, H', W', C)
        assert out.shape[0] == 2
        assert out.shape[1] == 8
        assert out.shape[-1] == 256
    
    def test_multiscale_forward(self, backbone_multiscale):
        """测试多尺度特征融合"""
        x = torch.randn(2, 3, 256, 256)
        out = backbone_multiscale(x)
        assert out.dim() == 4
        assert out.shape[-1] == 256


class TestTemporalTransformer:
    """测试时序Transformer"""
    
    @pytest.fixture
    def transformer(self):
        from models.point_tracker import TemporalTransformer
        return TemporalTransformer(
            dim=256,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
        )
    
    def test_forward(self, transformer):
        """测试前向传播"""
        x = torch.randn(2, 24, 100, 256)  # (B, T, N, C)
        out = transformer(x)
        assert out.shape == x.shape
    
    def test_window_attention(self):
        """测试窗口注意力"""
        from models.point_tracker import TemporalTransformer
        transformer = TemporalTransformer(
            dim=256,
            num_layers=2,
            num_heads=8,
            max_window_size=10,
        )
        x = torch.randn(2, 30, 50, 256)  # 超过窗口大小
        out = transformer(x)
        assert out.shape == x.shape


class TestPositionDecoder:
    """测试位置解码器"""
    
    @pytest.fixture
    def decoder(self):
        from models.point_tracker import PositionDecoder
        return PositionDecoder(dim=256)
    
    def test_forward(self, decoder):
        """测试前向传播"""
        features = torch.randn(2, 24, 100, 256)  # (B, T, N, C)
        init_positions = torch.rand(2, 100, 2)  # (B, N, 2)
        
        positions, visibility = decoder(features, init_positions)
        
        assert positions.shape == (2, 100, 24, 2)  # (B, N, T, 2)
        assert visibility.shape == (2, 100, 24)  # (B, N, T)
        assert (positions >= 0).all() and (positions <= 1).all()
        assert (visibility >= 0).all() and (visibility <= 1).all()


class TestSemanticEnhancedLFD:
    """测试语义增强频率分解"""
    
    @pytest.fixture
    def lfd(self):
        from models.freq_semantic_fusion import SemanticEnhancedLFD
        return SemanticEnhancedLFD(
            geo_dim=256,
            semantic_dim=256,
            num_bands=4,
            fusion_type='cross_attention',
        )

    def test_forward(self, lfd):
        """测试前向传播"""
        geo_feat = torch.randn(2, 24, 100, 256)
        semantic_feat = torch.randn(2, 24, 100, 256)
        
        out, info = lfd(geo_feat, semantic_feat)
        
        assert out.shape == geo_feat.shape
        assert 'band_features' in info
        assert 'gate' in info


class TestOcclusionPredictor:
    """测试遮挡预测器"""
    
    @pytest.fixture
    def predictor(self):
        from models.occlusion_predictor import FrequencyAwareOcclusionPredictor
        return FrequencyAwareOcclusionPredictor(
            dim=256,
            semantic_dim=256,
            num_bands=4,
        )

    def test_forward(self, predictor):
        """测试前向传播"""
        band_features = {
            f'band_{i}': torch.randn(2, 24, 100, 256)
            for i in range(4)
        }
        semantic_feat = torch.randn(2, 24, 100, 256)
        positions = torch.rand(2, 24, 100, 2)
        
        occ_prob, pred_pos, confidence = predictor(
            band_features, semantic_feat, positions
        )
        
        assert occ_prob.shape == (2, 24, 100)
        assert pred_pos.shape == (2, 24, 100, 2)
        assert confidence.shape == (2, 24, 100)


class TestFSPTTracker:
    """测试完整模型"""
    
    @pytest.fixture
    def config(self):
        from omegaconf import OmegaConf
        return OmegaConf.create({
            'backbone': {
                'type': 'resnet18',
                'pretrained': False,
                'use_multiscale': False,
            },
            'clip': {
                'model': 'ViT-B/16',
                'dim': 512,
                'freeze': True,
            },
            'frequency': {
                'enabled': True,
                'num_bands': 4,
            },
            'semantic': {
                'enabled': False,  # 测试时禁用CLIP
            },
            'temporal': {
                'dim': 256,
                'num_layers': 2,
                'num_heads': 8,
                'dropout': 0.1,
                'num_refinement_iters': 1,
            },
            'occlusion': {
                'enabled': True,
                'use_semantic_propagation': True,
            },
        })
    
    @pytest.fixture
    def model(self, config):
        from models.point_tracker import FSPTTracker
        return FSPTTracker(config)
    
    def test_forward(self, model):
        """测试前向传播"""
        video = torch.randn(2, 8, 3, 128, 128)
        query_points = torch.rand(2, 50, 3)
        query_points[:, :, 0] = 0  # 查询帧为0
        
        tracks, visibility = model(video, query_points)
        
        assert tracks.shape == (2, 50, 8, 2)
        assert visibility.shape == (2, 50, 8)
    
    def test_forward_with_info(self, model):
        """测试带info返回的前向传播"""
        video = torch.randn(2, 8, 3, 128, 128)
        query_points = torch.rand(2, 50, 3)
        query_points[:, :, 0] = 0
        
        tracks, visibility, info = model(video, query_points, return_info=True)
        
        assert 'freq_info' in info
    
    def test_iterative_refinement(self, config):
        """测试迭代精化"""
        config.temporal.num_refinement_iters = 3
        from models.point_tracker import FSPTTracker
        model = FSPTTracker(config)
        
        video = torch.randn(2, 8, 3, 128, 128)
        query_points = torch.rand(2, 50, 3)
        query_points[:, :, 0] = 0
        
        tracks, visibility = model(video, query_points)
        
        assert tracks.shape == (2, 50, 8, 2)


class TestMetrics:
    """测试评估指标"""
    
    def test_perfect_prediction(self):
        """测试完美预测"""
        from datasets.metrics import compute_tapvid_metrics
        
        N, T = 10, 20
        gt_tracks = torch.rand(N, T, 2)
        pred_tracks = gt_tracks.clone()
        gt_visibility = torch.ones(N, T, dtype=torch.bool)
        pred_visibility = gt_visibility.clone()
        query_points = torch.zeros(N, 3)
        query_points[:, 0] = 0
        query_points[:, 1:3] = gt_tracks[:, 0]
        
        metrics = compute_tapvid_metrics(
            pred_tracks, gt_tracks, pred_visibility, gt_visibility, query_points
        )
        
        assert metrics['AJ'] > 0.99
        assert metrics['OA'] > 0.99
    
    def test_noisy_prediction(self):
        """测试带噪声预测"""
        from datasets.metrics import compute_tapvid_metrics
        
        N, T = 10, 20
        gt_tracks = torch.rand(N, T, 2)
        pred_tracks = gt_tracks + torch.randn_like(gt_tracks) * 0.05
        gt_visibility = torch.ones(N, T, dtype=torch.bool)
        pred_visibility = gt_visibility.clone()
        query_points = torch.zeros(N, 3)
        query_points[:, 0] = 0
        query_points[:, 1:3] = gt_tracks[:, 0]
        
        metrics = compute_tapvid_metrics(
            pred_tracks, gt_tracks, pred_visibility, gt_visibility, query_points
        )
        
        assert 0 < metrics['AJ'] < 1
        assert metrics['OA'] == 1.0  # 可见性完美


class TestDataAugmentation:
    """测试数据增强"""
    
    @pytest.fixture
    def augmentation(self):
        from datasets.augmentation import PointTrackingAugmentation
        return PointTrackingAugmentation(
            random_crop=True,
            random_flip=True,
            color_jitter=0.4,
            random_scale=(0.9, 1.1),
            crop_size=(128, 128),
        )
    
    def test_augmentation(self, augmentation):
        """测试增强"""
        video = torch.rand(24, 3, 256, 256)
        points = torch.rand(100, 24, 2)
        occluded = torch.zeros(100, 24, dtype=torch.bool)
        
        video_aug, points_aug, occluded_aug = augmentation(video, points, occluded)
        
        assert video_aug.shape[0] == 24
        assert points_aug.shape == points.shape
        assert (points_aug >= 0).all() and (points_aug <= 1).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
