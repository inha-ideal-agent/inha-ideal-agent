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


# ---------------------------------------------------------------- 용접선 축 위치 (100 mm 창 이동 판정 입력)


def test_weld_axis_position_is_midpoint_times_scale_default_x():
    from rtworkbench.measure import weld_axis_position_mm

    # 측정선 (100,10)-(140,30): 중점 x=120 → 0.5 mm/px → 60 mm ; y 축이면 중점 y=20 → 10 mm
    assert weld_axis_position_mm((100.0, 10.0), (140.0, 30.0), 0.5) == 60.0
    assert weld_axis_position_mm((100.0, 10.0), (140.0, 30.0), 0.5, axis="x") == 60.0
    assert weld_axis_position_mm((100.0, 10.0), (140.0, 30.0), 0.5, axis="y") == 10.0
    assert weld_axis_position_mm((100.0, 10.0), (140.0, 30.0), None) is None  # 스케일 미확정


def test_weld_axis_invalid_axis_raises():
    from rtworkbench.measure import positions_from_measurements, weld_axis_position_mm

    with pytest.raises(ValueError):
        weld_axis_position_mm((0.0, 0.0), (1.0, 1.0), 0.1, axis="z")
    with pytest.raises(ValueError):
        positions_from_measurements([], 0.1, axis="diag")


def test_positions_from_measurements_maps_defect_ids_and_supports_axis_y():
    from rtworkbench.measure import positions_from_measurements
    from rtworkbench.models import Measurement

    ms = [
        Measurement(defect_id="a", p1=(0.0, 100.0), p2=(20.0, 100.0), length_px=20.0, length_mm=2.0),
        Measurement(defect_id="b", p1=(300.0, 40.0), p2=(340.0, 60.0), length_px=44.7, length_mm=4.47),
    ]
    assert positions_from_measurements(ms, 0.1) == {"a": 1.0, "b": 32.0}  # x 중점 10·320 px
    assert positions_from_measurements(ms, 0.1, axis="y") == {"a": 10.0, "b": 5.0}  # y 중점 100·50 px
    assert positions_from_measurements(ms, None) == {}  # 스케일 미확정 → 룰 엔진 폴백
