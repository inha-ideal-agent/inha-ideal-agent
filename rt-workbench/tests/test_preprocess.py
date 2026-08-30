"""전처리 모듈 테스트 — 로드/CLAHE 형상·dtype 보존, 컬러→그레이 변환."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from rtworkbench.preprocess import apply_clahe, load_grayscale, to_display_rgb


def _write_png(path, arr: np.ndarray) -> str:
    assert cv2.imwrite(str(path), arr)
    return str(path)


# --- load_grayscale ---------------------------------------------------------


def test_load_grayscale_from_path(tmp_path):
    src = np.arange(0, 60 * 80, dtype=np.uint32).reshape(60, 80) % 256
    src = src.astype(np.uint8)
    p = _write_png(tmp_path / "gray.png", src)

    img = load_grayscale(p)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape == (60, 80)
    assert np.array_equal(img, src)  # PNG 무손실 → 픽셀 보존


def test_load_grayscale_converts_color(tmp_path):
    # 채널별로 값이 다른 BGR 컬러 이미지 → 2차원 그레이로 변환되어야 한다
    color = np.zeros((40, 50, 3), dtype=np.uint8)
    color[..., 0] = 200  # B
    color[..., 1] = 90  # G
    color[..., 2] = 30  # R
    p = _write_png(tmp_path / "color.png", color)

    img = load_grayscale(p)
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert img.shape == (40, 50)
    # 그레이 변환값은 세 채널의 가중 평균 — 어떤 단일 채널과도 전부 일치하지 않는 중간값
    assert 30 < int(img.mean()) < 200


def test_load_grayscale_from_bytes(tmp_path):
    src = np.full((32, 48), 127, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", src)
    assert ok

    img = load_grayscale(buf.tobytes())
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert np.array_equal(img, src)


def test_load_grayscale_invalid_bytes_raises():
    with pytest.raises(ValueError):
        load_grayscale(b"this is not an image")


def test_load_grayscale_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        load_grayscale(tmp_path / "no_such_file.png")


def test_load_grayscale_unsupported_type_raises():
    with pytest.raises(ValueError):
        load_grayscale(12345)  # type: ignore[arg-type]


# --- apply_clahe ------------------------------------------------------------


def test_apply_clahe_preserves_shape_dtype_and_input():
    # 저대비 그라데이션 이미지
    row = np.linspace(100, 140, 200).astype(np.uint8)
    img = np.tile(row, (80, 1))
    before = img.copy()

    out = apply_clahe(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert np.array_equal(img, before)  # 입력 불변
    assert out is not img
    # CLAHE 는 저대비 입력의 대비(표준편차)를 키운다
    assert out.std() > img.std()


def test_apply_clahe_rejects_color_input():
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_clahe(color)


def test_apply_clahe_non_uint8_returns_uint8():
    img = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    out = apply_clahe(img)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


# --- to_display_rgb ---------------------------------------------------------


def test_to_display_rgb_shape_and_channels():
    gray = np.arange(0, 30 * 40, dtype=np.uint32).reshape(30, 40) % 256
    gray = gray.astype(np.uint8)

    rgb = to_display_rgb(gray)
    assert rgb.shape == (30, 40, 3)
    assert rgb.dtype == np.uint8
    # 그레이 → RGB: 세 채널 모두 원본과 동일
    for ch in range(3):
        assert np.array_equal(rgb[..., ch], gray)


def test_to_display_rgb_passthrough_3ch():
    src = np.random.default_rng(0).integers(0, 255, (20, 20, 3), dtype=np.uint8)
    out = to_display_rgb(src)
    assert out.shape == src.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out, src)
    assert out is not src  # 복사본 반환


def test_to_display_rgb_rejects_bad_shape():
    with pytest.raises(ValueError):
        to_display_rgb(np.zeros((5, 5, 4), dtype=np.uint8))


def test_apply_clahe_rejects_empty_input():
    """회귀 방지: 크기 0 배열은 OpenCV CLAHE 내부 무한 정지/오동작 — ValueError로 거부."""
    for shape in [(0, 10), (10, 0), (0, 0)]:
        with pytest.raises(ValueError):
            apply_clahe(np.zeros(shape, dtype=np.uint8))
