"""2클릭 자(measure.py) 테스트 — 거리/캘리브레이션/변환/측정."""

from __future__ import annotations

import pytest

from rtworkbench.measure import calibrate, distance_px, measure, px_to_mm


# ---------------------------------------------------------------- distance_px


def test_distance_px_pythagoras():
    assert distance_px((0, 0), (3, 4)) == pytest.approx(5.0)


def test_distance_px_axis_and_order_invariance():
    assert distance_px((10, 5), (10, 25)) == pytest.approx(20.0)
    # 점 순서를 바꿔도 거리는 같다
    assert distance_px((3, 4), (0, 0)) == pytest.approx(distance_px((0, 0), (3, 4)))


def test_distance_px_same_point_is_zero():
    assert distance_px((7.5, 7.5), (7.5, 7.5)) == 0.0


# ---------------------------------------------------------------- calibrate


def test_calibrate_normal():
    # 100px 기준물이 실제 10mm → 0.1 mm/px (납마커 시나리오)
    assert calibrate((0, 0), (100, 0), 10.0) == pytest.approx(0.1)


def test_calibrate_diagonal():
    # 대각선 (3,4) 거리 5px, 실제 20mm → 4.0 mm/px
    assert calibrate((0, 0), (3, 4), 20.0) == pytest.approx(4.0)


def test_calibrate_same_point_raises():
    with pytest.raises(ValueError):
        calibrate((5, 5), (5, 5), 10.0)


def test_calibrate_nonpositive_length_raises():
    with pytest.raises(ValueError):
        calibrate((0, 0), (100, 0), 0.0)
    with pytest.raises(ValueError):
        calibrate((0, 0), (100, 0), -3.0)


# ---------------------------------------------------------------- px_to_mm


def test_px_to_mm_normal():
    assert px_to_mm(50.0, 0.1) == pytest.approx(5.0)


def test_px_to_mm_none_scale():
    # 캘리브레이션 전(mm_per_px=None)이면 None
    assert px_to_mm(50.0, None) is None


# ---------------------------------------------------------------- measure


def test_measure_with_scale():
    length_px, length_mm = measure((0, 0), (3, 4), 0.5)
    assert length_px == pytest.approx(5.0)
    assert length_mm == pytest.approx(2.5)


def test_measure_without_scale():
    # 스케일 미확정: px 길이는 계산되고 mm는 None
    length_px, length_mm = measure((0, 0), (0, 10), None)
    assert length_px == pytest.approx(10.0)
    assert length_mm is None
