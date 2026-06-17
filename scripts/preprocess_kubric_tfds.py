#!/usr/bin/env python3
"""
TFDS Kubric数据预处理脚本
将TensorFlow Datasets格式转换为Pickle格式，大幅降低训练时的内存消耗

用法:
    python scripts/preprocess_kubric_tfds.py \
        --tfds_root /path/to/tapvid_kubric \
        --output_dir /path/to/output \
        --split train \
        --num_samples 9749
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='预处理TFDS Kubric数据集')
    parser.add_argument('--tfds_root', type=str, required=True,
                        help='TFDS数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test'],
                        help='数据集分割')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='处理的样本数量（None表示全部）')
    parser.add_argument('--num_frames', type=int, default=24,
                        help='每个样本的帧数')
    parser.add_argument('--num_points', type=int, default=256,
                        help='每个样本的点数')
    parser.add_argument('--resolution', type=int, nargs=2, default=[256, 256],
                        help='图像分辨率 [H, W]')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批处理大小（用于加载）')
    return parser.parse_args()

def setup_tfds():
    """设置TensorFlow Datasets"""
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # 禁用GPU（仅用于数据加载）
        tf.config.set_visible_devices([], 'GPU')

        return tf, tfds
    except ImportError as e:
        print(f"错误: 无法导入TensorFlow或TFDS: {e}")
        print("请安装: pip install tensorflow tensorflow-datasets")
        sys.exit(1)

def load_tfds_dataset(tfds_root, split, tf, tfds):
    """加载TFDS数据集"""
    print(f"\n加载TFDS数据集: {tfds_root}")
    print(f"分割: {split}")

    try:
        # 尝试方法1: 标准TFDS加载
        try:
            ds = tfds.load(
                'movi_e/256x256',
                data_dir=tfds_root,
                split=split,
                shuffle_files=False,
            )
            return ds
        except Exception as e1:
            print(f"标准加载失败: {e1}")
            print("尝试直接从文件加载...")

        # 方法2: 直接从tfrecord文件加载
        # 检查数据集路径
        data_path = Path(tfds_root) / "256x256" / "1.0.0"
        if not data_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {data_path}")

        # 查找tfrecord文件
        pattern = f"movi_e-{split}.tfrecord-*"
        tfrecord_files = sorted(data_path.glob(pattern))

        if not tfrecord_files:
            raise FileNotFoundError(f"未找到tfrecord文件: {data_path / pattern}")

        print(f"找到 {len(tfrecord_files)} 个tfrecord文件")

        # 直接从tfrecord文件创建数据集
        file_paths = [str(f) for f in tfrecord_files]
        ds = tf.data.TFRecordDataset(file_paths)

        # 解析tfrecord
        # 注意: 这需要知道数据的schema，这里简化处理
        # 实际上Kubric的数据格式比较复杂，可能需要更详细的解析
        print("警告: 直接从tfrecord加载可能需要手动解析数据格式")
        print("建议: 在服务器上使用标准TFDS路径结构")

        return ds

    except Exception as e:
        print(f"错误: 无法加载TFDS数据集: {e}")
        print("\n建议:")
        print("1. 确保数据集路径正确")
        print("2. 或者在服务器上直接预处理（服务器上的TFDS结构正确）")
        sys.exit(1)

def sample_points_from_tracks(tracks, occluded, num_points=256, seed=None):
    """
    从完整轨迹中采样点

    Args:
        tracks: (T, H, W, 2) 光流轨迹
        occluded: (T, H, W) 遮挡mask
        num_points: 采样点数
        seed: 随机种子

    Returns:
        query_points: (N, 3) [t, y, x]
        target_points: (N, T, 2) [x, y]
        occluded: (N, T)
    """
    if seed is not None:
        np.random.seed(seed)

    T, H, W, _ = tracks.shape

    # 随机选择查询帧（第一帧）
    query_frame = 0

    # 在第一帧中随机采样点
    valid_y = np.arange(H)
    valid_x = np.arange(W)

    # 随机采样
    sampled_indices = np.random.choice(H * W, size=min(num_points, H * W), replace=False)
    sampled_y = sampled_indices // W
    sampled_x = sampled_indices % W

    # 构建query_points (N, 3) [t, y, x]
    query_points = np.stack([
        np.full(len(sampled_y), query_frame),
        sampled_y,
        sampled_x
    ], axis=1).astype(np.float32)

    # 提取对应的轨迹 (N, T, 2)
    target_points = tracks[:, sampled_y, sampled_x, :]  # (T, N, 2)
    target_points = np.transpose(target_points, (1, 0, 2))  # (N, T, 2)

    # 提取遮挡信息 (N, T)
    occluded_points = occluded[:, sampled_y, sampled_x]  # (T, N)
    occluded_points = np.transpose(occluded_points, (1, 0))  # (N, T)

    return query_points, target_points, occluded_points

def process_sample(sample, num_frames, num_points, resolution):
    """
    处理单个样本

    Args:
        sample: TFDS样本
        num_frames: 帧数
        num_points: 点数
        resolution: [H, W]

    Returns:
        processed_sample: 处理后的样本字典
    """
    # 提取视频帧
    video = sample['video'].numpy()  # (T, H, W, 3)
    T, H, W, C = video.shape

    # 调整帧数
    if T > num_frames:
        # 均匀采样
        indices = np.linspace(0, T - 1, num_frames, dtype=int)
        video = video[indices]
        T = num_frames
    elif T < num_frames:
        # 填充
        padding = np.tile(video[-1:], (num_frames - T, 1, 1, 1))
        video = np.concatenate([video, padding], axis=0)
        T = num_frames

    # 调整分辨率
    if [H, W] != resolution:
        import cv2
        resized_video = []
        for frame in video:
            resized_frame = cv2.resize(frame, (resolution[1], resolution[0]))
            resized_video.append(resized_frame)
        video = np.stack(resized_video, axis=0)
        H, W = resolution

    # 提取光流轨迹（如果有）
    if 'tracks' in sample:
        tracks = sample['tracks'].numpy()  # (T, H, W, 2)
        occluded = sample['occluded'].numpy()  # (T, H, W)

        # 调整轨迹的帧数和分辨率
        if tracks.shape[0] != T or tracks.shape[1:3] != tuple(resolution):
            # 简化处理：重新生成轨迹（实际应该插值）
            print("警告: 轨迹尺寸不匹配，将使用简化处理")
            tracks = np.zeros((T, H, W, 2), dtype=np.float32)
            occluded = np.zeros((T, H, W), dtype=bool)
    else:
        # 如果没有轨迹，创建空轨迹
        tracks = np.zeros((T, H, W, 2), dtype=np.float32)
        occluded = np.zeros((T, H, W), dtype=bool)

    # 采样点
    query_points, target_points, occluded_points = sample_points_from_tracks(
        tracks, occluded, num_points=num_points
    )

    # 构建输出样本
    processed_sample = {
        'video': video.astype(np.uint8),  # (T, H, W, 3)
        'query_points': query_points.astype(np.float32),  # (N, 3) [t, y, x]
        'target_points': target_points.astype(np.float32),  # (N, T, 2) [x, y]
        'occluded': occluded_points.astype(bool),  # (N, T)
    }

    return processed_sample

def main():
    args = parse_args()

    print("=" * 70)
    print("TFDS Kubric数据预处理")
    print("=" * 70)
    print(f"TFDS根目录: {args.tfds_root}")
    print(f"输出目录: {args.output_dir}")
    print(f"分割: {args.split}")
    print(f"样本数: {args.num_samples if args.num_samples else '全部'}")
    print(f"帧数: {args.num_frames}")
    print(f"点数: {args.num_points}")
    print(f"分辨率: {args.resolution}")
    print("=" * 70)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置TensorFlow
    tf, tfds = setup_tfds()

    # 加载数据集
    ds = load_tfds_dataset(args.tfds_root, args.split, tf, tfds)

    # 处理样本
    processed_samples = []

    print(f"\n开始处理样本...")

    # 限制样本数
    if args.num_samples:
        ds = ds.take(args.num_samples)

    # 处理每个样本
    for idx, sample in enumerate(tqdm(ds, desc="处理样本", total=args.num_samples)):
        try:
            processed_sample = process_sample(
                sample,
                num_frames=args.num_frames,
                num_points=args.num_points,
                resolution=args.resolution
            )
            processed_samples.append(processed_sample)

            # 定期保存（避免内存溢出）
            if (idx + 1) % 1000 == 0:
                print(f"\n已处理 {idx + 1} 个样本，定期保存...")
                temp_file = output_dir / f"{args.split}_temp_{idx + 1}.pkl"
                with open(temp_file, 'wb') as f:
                    pickle.dump(processed_samples, f, protocol=4)
                print(f"临时文件已保存: {temp_file}")

        except Exception as e:
            print(f"\n警告: 处理样本 {idx} 时出错: {e}")
            continue

    # 保存最终结果
    output_file = output_dir / f"{args.split}.pkl"
    print(f"\n保存最终结果到: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(processed_samples, f, protocol=4)

    print(f"\n完成！")
    print(f"处理样本数: {len(processed_samples)}")
    print(f"输出文件: {output_file}")
    print(f"文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    # 保存元数据
    metadata = {
        'num_samples': len(processed_samples),
        'num_frames': args.num_frames,
        'num_points': args.num_points,
        'resolution': args.resolution,
        'split': args.split,
    }

    metadata_file = output_dir / f"{args.split}_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"元数据已保存: {metadata_file}")

    # 清理临时文件
    for temp_file in output_dir.glob(f"{args.split}_temp_*.pkl"):
        temp_file.unlink()
        print(f"已删除临时文件: {temp_file}")

if __name__ == '__main__':
    main()
