"""탐지 엔진 테스트 — recall 우선 검증 + 후보 폭주 방지 + YOLO 폴백 처리.

합성 필름은 scripts/generate_samples.py 의 generate_film() 을 직접 호출해 만든다
(scripts 디렉터리는 패키지가 아니므로 importlib 로 로드).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rtworkbench.detection import (
    CVFallbackDetector,
    YoloDetector,
    detector_status,
    get_detector,
)
from rtworkbench.models import DEFECT_TYPES

# --- 합성 샘플 생성기 로드 ---------------------------------------------------

_GEN_PATH = Path(__file__).resolve().parent.parent / "scripts" / "generate_samples.py"
_spec = importlib.util.spec_from_file_location("generate_samples_for_test", _GEN_PATH)
assert _spec is not None and _spec.loader is not None
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


# --- 매칭 유틸 (recall 판정: IoU>0.1 또는 중심점 포함) ------------------------


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _center(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _contains(bbox, pt) -> bool:
    return bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]


def _matched(gt_bbox, cand_bbox) -> bool:
    """IoU > 0.1 이거나, 어느 한쪽 bbox 중심이 상대 bbox 안이면 '후보 제시됨'."""
    return (
        _iou(gt_bbox, cand_bbox) > 0.1
        or _contains(gt_bbox, _center(cand_bbox))
        or _contains(cand_bbox, _center(gt_bbox))
    )


# --- CVFallbackDetector -----------------------------------------------------

# (seed, 결함 구성) — 생성기의 데모 샘플 구성과 동일 계열
RECALL_CASES = [
    (102, ["porosity", "porosity", "slag_inclusion"]),
    (103, ["crack", "porosity"]),
    (104, ["lack_of_fusion", "porosity", "porosity"]),
    (106, ["slag_inclusion", "lack_of_fusion", "crack", "porosity"]),
]


def test_cv_fallback_recall_over_60_percent():
    """recall 우선 검증: 시딩 결함의 60% 이상을 후보로 제시해야 한다."""
    det = CVFallbackDetector()
    total, hit = 0, 0
    for seed, spec in RECALL_CASES:
        img, gt = gen.generate_film(seed=seed, defects_spec=spec)
        cands = det.detect(img, max_candidates=30)
        for g in gt:
            total += 1
            if any(_matched(g["bbox"], c.bbox) for c in cands):
                hit += 1
    assert total == 12
    recall = hit / total
    assert recall >= 0.6, f"recall={recall:.2f} ({hit}/{total}) — 미탐은 배에 남는다"


def test_cv_fallback_low_contrast_lack_of_fusion_not_missed():
    """저대비 융합불량(놓치기 가장 쉬운 유형)이 단독으로도 잡히는지."""
    det = CVFallbackDetector()
    img, gt = gen.generate_film(seed=104, defects_spec=["lack_of_fusion"])
    cands = det.detect(img, max_candidates=30)
    assert any(_matched(gt[0]["bbox"], c.bbox) for c in cands)


def test_cv_fallback_no_defect_film_capped():
    """무결함 필름에서 후보 수가 폭주하지 않는지 (<= max_candidates)."""
    det = CVFallbackDetector()
    img, gt = gen.generate_film(seed=101, defects_spec=[])
    assert gt == []
    assert len(det.detect(img, max_candidates=30)) <= 30
    assert len(det.detect(img, max_candidates=10)) <= 10
    assert det.detect(img, max_candidates=0) == []


def test_cv_fallback_candidate_contract():
    """후보 필드 계약: source='ai', status='proposed', bbox 는 이미지 안, conf 범위."""
    det = CVFallbackDetector()
    img, _ = gen.generate_film(seed=103, defects_spec=["crack", "porosity"])
    h, w = img.shape
    cands = det.detect(img, max_candidates=30)
    assert cands, "결함 필름에서 후보가 하나도 없다"
    ids = set()
    for c in cands:
        assert c.source == "ai"
        assert c.status == "proposed"
        assert c.defect_type in DEFECT_TYPES
        assert c.id and c.id not in ids
        ids.add(c.id)
        x1, y1, x2, y2 = c.bbox
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h
        assert 0.0 < c.confidence <= 1.0


def test_cv_fallback_rejects_bad_input():
    det = CVFallbackDetector()
    with pytest.raises(ValueError):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))  # 컬러 입력 거부


def test_generate_film_is_deterministic():
    """같은 seed → 같은 이미지/GT (테스트 재현성 계약)."""
    img_a, gt_a = gen.generate_film(seed=42, defects_spec=["porosity", "crack"])
    img_b, gt_b = gen.generate_film(seed=42, defects_spec=["porosity", "crack"])
    assert np.array_equal(img_a, img_b)
    assert gt_a == gt_b


# --- YoloDetector (이 환경엔 ultralytics 없음 — '사용 불가'가 정상 경로) ------


def test_yolo_unavailable_without_exception():
    """ultralytics 미설치 환경: 예외 없이 available=False, detect()는 빈 목록."""
    det = YoloDetector(weights_path="/nonexistent/dir/best.pt")
    assert det.available is False
    assert det.note  # 사유 문자열 존재
    img, _ = gen.generate_film(seed=103, defects_spec=["crack"])
    assert det.detect(img) == []  # 예외를 던지지 않는다


def test_get_detector_falls_back_to_cv():
    det = get_detector()
    assert isinstance(det, CVFallbackDetector)
    assert det.name == "cv-fallback"


def test_detector_status_keys_and_values():
    st = detector_status()
    assert {"backend", "yolo_available", "weights_path", "note"} <= set(st.keys())
    assert st["yolo_available"] is False
    assert st["backend"] == "cv-fallback"
    assert isinstance(st["weights_path"], str)
    assert st["note"]


def test_cv_fallback_detect_rejects_empty_input():
    """회귀 방지: 크기 0 배열은 apply_clahe 경유 무한 정지 — ValueError로 거부."""
    det = CVFallbackDetector()
    for shape in [(0, 10), (10, 0), (0, 0)]:
        with pytest.raises(ValueError):
            det.detect(np.zeros(shape, dtype=np.uint8))
