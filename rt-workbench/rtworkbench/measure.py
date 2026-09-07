"""2클릭 자(Ruler) — 픽셀→mm 스케일 캘리브레이션과 길이 측정.

픽셀→mm 문제의 해답: 필름 안에 항상 함께 찍히는 기지(旣知) 길이 기준물
(IQI 와이어 간격, 납마커 등)을 판독원이 2클릭으로 지정해 스케일을 확정한다.
'사람이 확정한다'는 v3 철학과 정합 — AI가 추정한 스케일은 없다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # 순환 import 회피 — 실행 시에는 타입만 필요
    from rtworkbench.models import Measurement

Point = tuple[float, float]

# 용접선 축 — 필름 이미지의 어느 축을 따라 용접부가 놓였는가 ('x' = 가로, 'y' = 세로)
WELD_AXES: tuple[str, ...] = ("x", "y")


def distance_px(p1: Point, p2: Point) -> float:
    """두 픽셀 좌표 사이의 유클리드 거리(px)."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calibrate(p1: Point, p2: Point, known_length_mm: float) -> float:
    """기지 길이 기준물의 양 끝 2클릭으로 mm/px 스케일을 계산한다.

    Raises:
        ValueError: 두 점이 같거나 기지 길이가 0 이하일 때.
    """
    d = distance_px(p1, p2)
    if d <= 0:
        raise ValueError("캘리브레이션 두 점이 동일합니다. 서로 다른 두 점을 클릭하세요.")
    if known_length_mm <= 0:
        raise ValueError("기준물 실제 길이(mm)는 0보다 커야 합니다.")
    return known_length_mm / d


def px_to_mm(length_px: float, mm_per_px: float | None) -> float | None:
    """픽셀 길이를 mm로 변환. 스케일 미확정이면 None."""
    if mm_per_px is None:
        return None
    return length_px * mm_per_px


def measure(p1: Point, p2: Point, mm_per_px: float | None) -> tuple[float, float | None]:
    """2클릭 측정: (길이 px, 길이 mm 또는 None)을 반환."""
    d = distance_px(p1, p2)
    return d, px_to_mm(d, mm_per_px)


def weld_axis_position_mm(
    p1: Point, p2: Point, mm_per_px: float | None, axis: str = "x"
) -> float | None:
    """측정선(2클릭) 중점을 용접선 축에 투영한 위치(mm) — 100 mm 창 이동 그룹 판정의 입력.

    가정: 용접선은 이미지 축 하나와 평행하게 놓여 있다(필름을 용접선 방향으로 스캔하는 관행).
    기본 'x'(가로로 놓인 용접부), 세로로 놓였으면 axis='y'. 위치 = 결함 중심(측정선 중점)의 해당 축
    픽셀 좌표 × 스케일. 스케일 미확정(None)이면 None — 룰 엔진은 전체를 한 구간으로 폴백한다.

    Raises:
        ValueError: axis 가 'x'/'y' 가 아닐 때.
    """
    if axis not in WELD_AXES:
        raise ValueError(f"용접선 축(axis)은 {'/'.join(WELD_AXES)} 중 하나여야 합니다: {axis!r}")
    if mm_per_px is None:
        return None
    k = 0 if axis == "x" else 1
    return (p1[k] + p2[k]) / 2.0 * mm_per_px


def positions_from_measurements(
    measurements: Iterable["Measurement"], mm_per_px: float | None, axis: str = "x"
) -> dict[str, float]:
    """측정 목록 → {defect_id: 용접선 축 위치 mm}. 스케일 미확정이면 {} (룰 엔진이 전체 구간으로 폴백).

    RuleEngine.evaluate_all(..., positions=...) 에 그대로 넘긴다.
    """
    if axis not in WELD_AXES:
        raise ValueError(f"용접선 축(axis)은 {'/'.join(WELD_AXES)} 중 하나여야 합니다: {axis!r}")
    if mm_per_px is None:
        return {}
    out: dict[str, float] = {}
    for m in measurements:
        pos = weld_axis_position_mm(m.p1, m.p2, mm_per_px, axis)
        if pos is not None:
            out[m.defect_id] = pos
    return out
