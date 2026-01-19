# src/depth_3d.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Depth3DResult:
    original_bgr: np.ndarray
    gray: np.ndarray
    depth_colormap_bgr: np.ndarray
    points_3d: np.ndarray  # (H, W, 3) float32 [X,Y,Z]


def load_image_bgr(path: str) -> np.ndarray:
    if not path or not isinstance(path, str):
        raise ValueError("이미지 경로가 올바르지 않습니다.")
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"이미지를 불러오지 못했습니다: {path}")
    return image


def generate_depth_map(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image_bgr is None:
        raise ValueError("입력된 이미지가 없습니다.")

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("입력은 numpy.ndarray 여야 합니다.")

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("입력 이미지는 (H, W, 3) 형태의 BGR 이미지여야 합니다.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    depth_colormap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)  # 문서 예시 방식 :contentReference[oaicite:4]{index=4}
    return gray, depth_colormap


def depth_to_pointcloud(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        raise ValueError("gray(depth) 입력이 없습니다.")
    if not isinstance(gray, np.ndarray):
        raise TypeError("gray는 numpy.ndarray 여야 합니다.")
    if gray.ndim != 2:
        raise ValueError("gray는 (H, W) 형태의 2D 배열이어야 합니다.")

    h, w = gray.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)

    points_3d = np.dstack((X.astype(np.float32), Y.astype(np.float32), Z))
    return points_3d


def run_2d_to_3d(
    input_path: str,
    out_dir: str = "outputs",
    save_pointcloud: bool = True,
) -> Depth3DResult:
    os.makedirs(out_dir, exist_ok=True)

    image = load_image_bgr(input_path)
    gray, depth_colormap = generate_depth_map(image)
    points_3d = depth_to_pointcloud(gray)

    # Save outputs for report
    depth_path = os.path.join(out_dir, "depth_map.png")
    cv2.imwrite(depth_path, depth_colormap)

    if save_pointcloud:
        pc_path = os.path.join(out_dir, "pointcloud.npy")
        np.save(pc_path, points_3d)

    return Depth3DResult(
        original_bgr=image,
        gray=gray,
        depth_colormap_bgr=depth_colormap,
        points_3d=points_3d,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="2D -> 3D (pseudo depth) demo")
    parser.add_argument("--input", required=True, help="input image path (e.g., odong.jpg)")
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--no_pc", action="store_true", help="do not save pointcloud.npy")
    args = parser.parse_args()

    result = run_2d_to_3d(args.input, args.out, save_pointcloud=not args.no_pc)
    print("[OK] depth_map saved to:", os.path.join(args.out, "depth_map.png"))
    if not args.no_pc:
        print("[OK] pointcloud saved to:", os.path.join(args.out, "pointcloud.npy"))
    print("[INFO] depth_map shape:", result.depth_colormap_bgr.shape)
    print("[INFO] pointcloud shape:", result.points_3d.shape)


if __name__ == "__main__":
    main()
