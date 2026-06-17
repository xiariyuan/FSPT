"""
点追踪数据增强

需要同时变换视频和点坐标
"""

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Tuple, Optional


class PointTrackingAugmentation:
    """
    点追踪数据增强
    
    需要同时变换视频和点坐标，保持一致性
    
    Args:
        random_crop: 是否随机裁剪
        random_flip: 是否随机水平翻转
        color_jitter: 颜色抖动强度
        random_scale: 随机缩放范围
        random_rotation: 随机旋转角度范围（度）
    """
    
    def __init__(
        self,
        random_crop: bool = True,
        random_flip: bool = True,
        color_jitter: float = 0.4,
        random_scale: Tuple[float, float] = (0.8, 1.2),
        random_rotation: float = 15.0,
        crop_size: Tuple[int, int] = (256, 256),
    ):
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        if isinstance(random_scale, (int, float)):
            random_scale = (float(random_scale), float(random_scale))
        elif isinstance(random_scale, (list, tuple)) and len(random_scale) == 2:
            random_scale = (float(random_scale[0]), float(random_scale[1]))
        else:
            random_scale = None
        self.random_scale = random_scale
        self.random_rotation = random_rotation
        self.crop_size = crop_size
        
        # 颜色增强
        if color_jitter > 0:
            self.color_transform = T.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter * 0.1,
            )
        else:
            self.color_transform = None
    
    def __call__(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        occluded: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用数据增强
        
        Args:
            video: (T, C, H, W) 视频张量
            points: (N, T, 2) 点坐标，归一化[0,1]，格式[y, x]
            occluded: (N, T) 遮挡标签
            query_points: (N, 3) 查询点 [t, y, x]
            
        Returns:
            augmented video, points, occluded
        """
        T_frames, C, H, W = video.shape
        
        # 1. 随机水平翻转
        if self.random_flip and np.random.rand() > 0.5:
            video = torch.flip(video, dims=[-1])
            points = points.clone()
            points[:, :, 1] = 1 - points[:, :, 1]  # x坐标翻转
            if query_points is not None:
                query_points = query_points.clone()
                query_points[:, 2] = 1 - query_points[:, 2]
        
        # 2. 随机缩放
        if self.random_scale is not None:
            outputs = self._random_scale(
                video, points, occluded, query_points
            )
            if query_points is not None:
                video, points, occluded, query_points = outputs
            else:
                video, points, occluded = outputs
        
        # 3. 随机裁剪
        if self.random_crop:
            outputs = self._random_crop(
                video, points, occluded, query_points
            )
            if query_points is not None:
                video, points, occluded, query_points = outputs
            else:
                video, points, occluded = outputs

        # 4. 随机旋转
        if self.random_rotation and np.random.rand() > 0.5:
            outputs = self._random_rotate(
                video, points, occluded, query_points
            )
            if query_points is not None:
                video, points, occluded, query_points = outputs
            else:
                video, points, occluded = outputs
        
        # 5. 颜色增强 (每帧独立)
        if self.color_transform is not None:
            video = self._apply_color_jitter(video)
        
        if query_points is not None:
            return video, points, occluded, query_points
        return video, points, occluded
    
    def _random_scale(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        occluded: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机缩放"""
        T_frames, C, H, W = video.shape
        
        scale = np.random.uniform(*self.random_scale)
        new_H, new_W = max(1, int(H * scale)), max(1, int(W * scale))
        
        # 缩放视频
        video = F.interpolate(
            video,  # (T, C, H, W) 将T视作batch
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )
        
        # 裁剪或填充回原始大小
        points = points.clone()
        
        if scale > 1:
            # 随机裁剪
            start_y = np.random.randint(0, max(1, new_H - H + 1))
            start_x = np.random.randint(0, max(1, new_W - W + 1))
            video = video[:, :, start_y:start_y+H, start_x:start_x+W]
            
            # 调整点坐标
            # 新坐标 = (旧坐标 * scale - offset) / scale_back
            points[:, :, 0] = points[:, :, 0] * scale - start_y / H
            points[:, :, 1] = points[:, :, 1] * scale - start_x / W
            if query_points is not None:
                query_points = query_points.clone()
                query_points[:, 1] = query_points[:, 1] * scale - start_y / H
                query_points[:, 2] = query_points[:, 2] * scale - start_x / W
            
        else:
            # 中心填充
            pad_y = (H - new_H) // 2
            pad_x = (W - new_W) // 2
            padded = torch.zeros(T_frames, C, H, W, dtype=video.dtype, device=video.device)
            padded[:, :, pad_y:pad_y+new_H, pad_x:pad_x+new_W] = video
            video = padded
            
            # 调整点坐标
            points[:, :, 0] = points[:, :, 0] * scale + pad_y / H
            points[:, :, 1] = points[:, :, 1] * scale + pad_x / W
            if query_points is not None:
                query_points = query_points.clone()
                query_points[:, 1] = query_points[:, 1] * scale + pad_y / H
                query_points[:, 2] = query_points[:, 2] * scale + pad_x / W
        
        # 标记越界点为遮挡
        out_of_bounds = (
            (points[:, :, 0] < 0) | (points[:, :, 0] > 1) |
            (points[:, :, 1] < 0) | (points[:, :, 1] > 1)
        )
        occluded = occluded | out_of_bounds
        points = torch.clamp(points, 0, 1)
        if query_points is not None:
            query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0, 1)
        
        if query_points is not None:
            return video, points, occluded, query_points
        return video, points, occluded
    
    def _random_crop(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        occluded: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机裁剪到目标大小"""
        T_frames, C, H, W = video.shape
        crop_H, crop_W = self.crop_size
        
        if H <= crop_H or W <= crop_W:
            if query_points is not None:
                return video, points, occluded, query_points
            return video, points, occluded
        
        # 随机裁剪位置
        start_y = np.random.randint(0, max(1, H - crop_H + 1))
        start_x = np.random.randint(0, max(1, W - crop_W + 1))
        
        # 裁剪视频
        video = video[:, :, start_y:start_y+crop_H, start_x:start_x+crop_W]
        
        # 调整点坐标
        points = points.clone()
        
        # 从像素坐标转换
        points[:, :, 0] = (points[:, :, 0] * H - start_y) / crop_H
        points[:, :, 1] = (points[:, :, 1] * W - start_x) / crop_W
        if query_points is not None:
            query_points = query_points.clone()
            query_points[:, 1] = (query_points[:, 1] * H - start_y) / crop_H
            query_points[:, 2] = (query_points[:, 2] * W - start_x) / crop_W
        
        # 标记越界点为遮挡
        out_of_bounds = (
            (points[:, :, 0] < 0) | (points[:, :, 0] > 1) |
            (points[:, :, 1] < 0) | (points[:, :, 1] > 1)
        )
        occluded = occluded | out_of_bounds
        points = torch.clamp(points, 0, 1)
        if query_points is not None:
            query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0, 1)
        
        if query_points is not None:
            return video, points, occluded, query_points
        return video, points, occluded
    
    def _apply_color_jitter(self, video: torch.Tensor) -> torch.Tensor:
        """应用颜色抖动"""
        T_frames, C, H, W = video.shape
        
        # 对所有帧应用相同的变换
        transform_params = self.color_transform.get_params(
            self.color_transform.brightness,
            self.color_transform.contrast,
            self.color_transform.saturation,
            self.color_transform.hue,
        )
        
        for t in range(T_frames):
            frame = video[t]
            
            # 应用变换
            for fn_id in transform_params[0]:
                if fn_id == 0 and transform_params[1] is not None:
                    frame = T.functional.adjust_brightness(frame, transform_params[1])
                elif fn_id == 1 and transform_params[2] is not None:
                    frame = T.functional.adjust_contrast(frame, transform_params[2])
                elif fn_id == 2 and transform_params[3] is not None:
                    frame = T.functional.adjust_saturation(frame, transform_params[3])
                elif fn_id == 3 and transform_params[4] is not None:
                    frame = T.functional.adjust_hue(frame, transform_params[4])
            
            if frame.dtype.is_floating_point:
                frame = frame.clamp(0.0, 1.0)
            video[t] = frame
        
        return video

    def _random_rotate(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        occluded: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """随机旋转"""
        T_frames, C, H, W = video.shape
        if H <= 1 or W <= 1:
            if query_points is not None:
                return video, points, occluded, query_points
            return video, points, occluded

        angle = np.random.uniform(-self.random_rotation, self.random_rotation)

        # 旋转视频
        rotated_frames = []
        for t in range(T_frames):
            rotated_frames.append(
                T.functional.rotate(
                    video[t],
                    angle=angle,
                    interpolation=T.InterpolationMode.BILINEAR,
                    expand=False,
                    fill=0,
                )
            )
        video = torch.stack(rotated_frames, dim=0)

        # 旋转点坐标 (归一化)
        # torchvision.rotate rotates image CCW in image coords (y-down),
        # so points must be rotated with negative angle in standard math coords.
        points = points.clone()
        angle_rad = np.deg2rad(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        cx = W / 2.0
        cy = H / 2.0

        x = points[:, :, 1] * W
        y = points[:, :, 0] * H
        x_shift = x - cx
        y_shift = y - cy

        x_rot = x_shift * cos_a - y_shift * sin_a + cx
        y_rot = x_shift * sin_a + y_shift * cos_a + cy

        points[:, :, 0] = y_rot / H
        points[:, :, 1] = x_rot / W
        if query_points is not None:
            query_points = query_points.clone()
            qx = query_points[:, 2] * W
            qy = query_points[:, 1] * H
            qx_shift = qx - cx
            qy_shift = qy - cy
            qx_rot = qx_shift * cos_a - qy_shift * sin_a + cx
            qy_rot = qx_shift * sin_a + qy_shift * cos_a + cy
            query_points[:, 1] = qy_rot / H
            query_points[:, 2] = qx_rot / W

        out_of_bounds = (
            (points[:, :, 0] < 0) | (points[:, :, 0] > 1) |
            (points[:, :, 1] < 0) | (points[:, :, 1] > 1)
        )
        occluded = occluded | out_of_bounds
        points = torch.clamp(points, 0, 1)
        if query_points is not None:
            query_points[:, 1:3] = torch.clamp(query_points[:, 1:3], 0, 1)

        if query_points is not None:
            return video, points, occluded, query_points
        return video, points, occluded


class TemporalAugmentation:
    """
    时序数据增强
    """
    
    def __init__(
        self,
        random_reverse: bool = True,
        random_speed: Tuple[float, float] = (0.5, 2.0),
        random_start: bool = True,
    ):
        self.random_reverse = random_reverse
        if isinstance(random_speed, (int, float)):
            self.random_speed = (float(random_speed), float(random_speed))
        elif isinstance(random_speed, (list, tuple)) and len(random_speed) == 2:
            self.random_speed = (float(random_speed[0]), float(random_speed[1]))
        else:
            self.random_speed = None
        self.random_start = random_start
    
    def __call__(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        occluded: torch.Tensor,
        query_points: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video: (T, C, H, W)
            points: (N, T, 2)
            occluded: (N, T)
            query_points: (N, 3) [t, y, x]
        """
        T = video.shape[0]
        
        # 随机时间反转
        if self.random_reverse and np.random.rand() > 0.5:
            video = torch.flip(video, dims=[0])
            points = torch.flip(points, dims=[1])
            occluded = torch.flip(occluded, dims=[1])
            query_points = query_points.clone()
            query_points[:, 0] = T - 1 - query_points[:, 0]

        # 随机速度变化（时间重采样）
        if self.random_speed is not None:
            speed = np.random.uniform(*self.random_speed)
            if abs(speed - 1.0) > 1e-3:
                video, points, occluded, query_points = self._temporal_resample(
                    video, points, occluded, query_points, speed
                )

        # 随机起始帧（循环移位）
        if self.random_start and T > 1:
            offset = np.random.randint(0, T)
            if offset > 0:
                video = torch.roll(video, shifts=-offset, dims=0)
                points = torch.roll(points, shifts=-offset, dims=1)
                occluded = torch.roll(occluded, shifts=-offset, dims=1)
                query_points = query_points.clone()
                query_points[:, 0] = (query_points[:, 0] - offset) % T
        
        return video, points, occluded, query_points

    def _temporal_resample(
        self,
        video: torch.Tensor,
        points: torch.Tensor,
        occluded: torch.Tensor,
        query_points: torch.Tensor,
        speed: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """时间维度重采样，保持输出长度不变"""
        T = video.shape[0]
        device = video.device

        time = torch.linspace(0, T - 1, T, device=device)
        center = (T - 1) / 2.0
        src_times = (time - center) * speed + center
        src_times = src_times.clamp(0, T - 1)

        t0 = torch.floor(src_times).long()
        t1 = torch.clamp(t0 + 1, max=T - 1)
        alpha = (src_times - t0.float()).view(T, 1, 1, 1)

        # 视频重采样
        video = (1 - alpha) * video[t0] + alpha * video[t1]

        # 点坐标重采样
        alpha_pts = alpha.view(1, T, 1)
        points = (1 - alpha_pts) * points[:, t0, :] + alpha_pts * points[:, t1, :]

        # 遮挡标签使用最近邻
        choose_t1 = (src_times - t0.float()) > 0.5
        occluded = torch.where(choose_t1.view(1, T), occluded[:, t1], occluded[:, t0])

        # 调整查询帧索引（寻找最接近的源时间）
        query_points = query_points.clone()
        q_t = query_points[:, 0].view(-1, 1)
        new_t = torch.abs(src_times.view(1, T) - q_t).argmin(dim=1).float()
        query_points[:, 0] = new_t
        # 同步查询点位置到重采样后的轨迹
        new_t_idx = new_t.long().clamp(0, T - 1)
        point_idx = torch.arange(points.shape[0], device=points.device)
        query_points[:, 1] = points[point_idx, new_t_idx, 0]
        query_points[:, 2] = points[point_idx, new_t_idx, 1]

        return video, points, occluded, query_points


if __name__ == '__main__':
    # 测试增强
    print("Testing PointTrackingAugmentation...")
    
    aug = PointTrackingAugmentation(
        random_crop=True,
        random_flip=True,
        color_jitter=0.4,
        random_scale=(0.8, 1.2),
        crop_size=(224, 224),
    )
    
    # 创建测试数据
    T, C, H, W = 24, 3, 256, 256
    N = 100
    
    video = torch.rand(T, C, H, W)
    points = torch.rand(N, T, 2)
    occluded = torch.zeros(N, T, dtype=torch.bool)
    
    # 应用增强
    video_aug, points_aug, occluded_aug = aug(video, points, occluded)
    
    print(f"Original video shape: {video.shape}")
    print(f"Augmented video shape: {video_aug.shape}")
    print(f"Points range: [{points_aug.min():.3f}, {points_aug.max():.3f}]")
    print(f"New occluded ratio: {occluded_aug.float().mean():.3f}")
    
    print("Test passed!")
