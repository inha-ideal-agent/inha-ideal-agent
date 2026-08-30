"""2클릭 자(Ruler) — 픽셀→mm 스케일 캘리브레이션과 길이 측정.

픽셀→mm 문제의 해답: 필름 안에 항상 함께 찍히는 기지(旣知) 길이 기준물
(IQI 와이어 간격, 납마커 등)을 판독원이 2클릭으로 지정해 스케일을 확정한다.
'사람이 확정한다'는 v3 철학과 정합 — AI가 추정한 스케일은 없다.
"""

from __future__ import annotations

import math

Point = tuple[float, float]


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
