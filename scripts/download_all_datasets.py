#!/usr/bin/env python3
"""下载所有训练和测试数据集"""
import os
import sys
import argparse
import shutil
from pathlib import Path


def download_kubric_movi_e(data_dir: Path):
    """下载 Kubric MOVi-E 数据集"""
    print("=" * 60)
    print("Downloading Kubric MOVi-E (256x256)...")
    print("Estimated size: ~150 GB")
    print("=" * 60)

    try:
        import tensorflow_datasets as tfds
    except Exception as exc:
        raise ImportError("tensorflow_datasets is required for Kubric download.") from exc

    data_dir.mkdir(parents=True, exist_ok=True)

    # 下载训练集
    print("Downloading training set...")
    train_ds = tfds.load(
        "movi_e/256x256",
        data_dir=str(data_dir),
        split="train",
        download=True,
    )
    print(f"Training set loaded")

    # 下载验证集
    print("Downloading validation set...")
    val_ds = tfds.load(
        "movi_e/256x256",
        data_dir=str(data_dir),
        split="validation",
        download=True,
    )
    print(f"Validation set loaded")

    print("Kubric MOVi-E download complete!")
    print("Note: TFDS does not include point tracks. For debugging, enable use_tfds + allow_synthetic_tracks.")
    print("      For real training, provide official Kubric pickle annotations.")


def download_tapvid_kinetics(kinetics_dir: Path):
    """下载 TAP-Vid-Kinetics 标注文件"""
    print("=" * 60)
    print("Downloading TAP-Vid-Kinetics annotations...")
    print("=" * 60)

    import urllib.request
    import zipfile

    kinetics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = kinetics_dir / "tapvid_kinetics.csv"
    nested_dir = kinetics_dir / "tapvid_kinetics"
    if nested_dir.is_dir():
        for child in nested_dir.iterdir():
            target = kinetics_dir / child.name
            if not target.exists():
                shutil.move(str(child), str(target))
        nested_dir.rmdir()

    if csv_path.exists():
        if csv_path.stat().st_size == 0:
            print(f"Found empty {csv_path}, re-downloading.")
            csv_path.unlink()
        else:
            print(f"Found existing {csv_path}, skipping download.")
            return

    zip_url = "https://storage.googleapis.com/dm-tapnet/tapvid_kinetics.zip"
    zip_path = kinetics_dir / "tapvid_kinetics.zip"

    print(f"Downloading {zip_url}...")
    urllib.request.urlretrieve(zip_url, str(zip_path))
    print(f"Saved to {zip_path}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(kinetics_dir)
        print("Extracted tapvid_kinetics.zip")
        if nested_dir.is_dir():
            for child in nested_dir.iterdir():
                target = kinetics_dir / child.name
                if not target.exists():
                    shutil.move(str(child), str(target))
            nested_dir.rmdir()
    finally:
        if zip_path.exists():
            zip_path.unlink()

    print("TAP-Vid-Kinetics annotations download complete!")
    print("Note: The official package provides csv/txt split files, not tapvid_kinetics.pkl.")
    print("      Kinetics clips can be populated from the official Kinetics-700 tar archives")
    print("      under /gemini/code/datasets/kinetics/raw_*_targz and linked with scripts/link_tapvid_kinetics_videos.py.")


def download_tapvid_davis(davis_dir: Path):
    """下载 TAP-Vid-DAVIS 标注文件"""
    print("=" * 60)
    print("Downloading TAP-Vid-DAVIS annotations...")
    print("=" * 60)

    import urllib.request

    davis_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = davis_dir / "tapvid_davis.pkl"
    if pkl_path.exists():
        if pkl_path.stat().st_size == 0:
            print(f"Found empty {pkl_path}, re-downloading.")
            pkl_path.unlink()
        else:
            print(f"Found existing {pkl_path}, skipping download.")
            return

    pkl_url = "https://storage.googleapis.com/dm-tapnet/tapvid_davis.pkl"
    print(f"Downloading {pkl_url}...")
    urllib.request.urlretrieve(pkl_url, str(pkl_path))
    print(f"Saved to {pkl_path}")
    print("TAP-Vid-DAVIS annotations download complete!")
    print("Note: If videos are not embedded in the pkl, download DAVIS frames separately.")


def parse_args():
    parser = argparse.ArgumentParser(description="Download TAP-Vid datasets")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Datasets root directory (default: /gemini/code/datasets)",
    )
    parser.add_argument(
        "--kubric-tfds",
        action="store_true",
        help="Download Kubric MOVi-E via TFDS (no point tracks; debug only)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workspace_root = Path("/gemini/code")
    if args.root:
        datasets_root = Path(args.root)
        if not datasets_root.is_absolute():
            datasets_root = workspace_root / datasets_root
        datasets_root = datasets_root.resolve()
    else:
        datasets_root = Path("/gemini/code/datasets").resolve()

    print("Starting dataset downloads...")
    print(f"Datasets root: {datasets_root}")
    print()

    exit_code = 0

    # 先下载DAVIS标注（较小）
    try:
        download_tapvid_davis(datasets_root / "tapvid_davis")
    except Exception as e:
        print(f"Error downloading DAVIS: {e}")
        exit_code = 1

    # 下载Kinetics标注
    try:
        download_tapvid_kinetics(datasets_root / "tapvid_kinetics")
    except Exception as e:
        print(f"Error downloading Kinetics: {e}")
        exit_code = 1

    # 下载Kubric MOVi-E（可选：仅调试）
    if args.kubric_tfds:
        try:
            download_kubric_movi_e(datasets_root / "tapvid_kubric")
        except Exception as e:
            print(f"Error downloading Kubric: {e}")
            exit_code = 1
    else:
        print("Skipping Kubric TFDS download (debug only).")
        print("Note: For training, download official TAP-Vid Kubric pickle annotations.")

    print()
    print("=" * 60)
    print("All downloads complete!")
    print("=" * 60)
    sys.exit(exit_code)
