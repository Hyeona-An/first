# tests/test_3d_processing.py
import numpy as np
import pytest

from first.src.depth import generate_depth_map, depth_to_pointcloud


def test_generate_depth_map_valid_image():
    # 검정색 빈 이미지 (문서 예시) :contentReference[oaicite:6]{index=6}
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gray, depth_map = generate_depth_map(image)

    assert gray.shape == (100, 100)
    assert depth_map.shape == image.shape, "출력 크기가 입력 크기와 다릅니다."
    assert isinstance(depth_map, np.ndarray), "출력 데이터 타입이 ndarray가 아닙니다."


def test_generate_depth_map_none_raises():
    with pytest.raises(ValueError):
        generate_depth_map(None)


def test_depth_to_pointcloud_shape_and_type():
    gray = np.zeros((50, 80), dtype=np.uint8)
    points = depth_to_pointcloud(gray)

    assert points.shape == (50, 80, 3)
    assert points.dtype == np.float32
    # Z축이 gray 기반인지 간단 검증
    assert float(points[0, 0, 2]) == 0.0
