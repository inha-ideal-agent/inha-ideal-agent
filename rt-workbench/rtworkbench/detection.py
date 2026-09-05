"""2차 눈(Second Reader) — 결함 후보 탐지 엔진.

원칙 3: 탐지는 부품이다. recall(민감도) 우선 — 오탐은 판독원의 클릭 한 번으로
기각되지만, 미탐은 배에 남는다. 후보는 항상 '제안(proposed)' 상태로만 생성되며,
합부 판정은 결정론적 룰 엔진 + 판독원의 몫이다(원칙 1 — AI는 판정하지 않는다).

백엔드 2종:
  - YoloDetector      : Ultralytics YOLO 가중치(config.YOLO_WEIGHTS_PATH; YOLO26s 권장,
                        YOLOv8/YOLO11 가중치도 같은 API)가 있고
                        import 가능할 때만 사용. import 실패는 예외가 아니라 '사용 불가'.
  - CVFallbackDetector: OpenCV 국소 대비 기반 안전망 — 모델 가중치 없이도
                        데모가 항상 동작하게 한다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from rtworkbench import config
from rtworkbench.models import DEFECT_TYPES, DefectCandidate, new_id
from rtworkbench.preprocess import apply_clahe

__all__ = [
    "DetectorBackend",
    "CVFallbackDetector",
    "YoloDetector",
    "get_detector",
    "detector_status",
]


@runtime_checkable
class DetectorBackend(Protocol):
    """탐지 백엔드 계약. name 은 'yolo' | 'cv-fallback'."""

    name: str

    def detect(self, img_gray: np.ndarray, max_candidates: int = 30) -> list[DefectCandidate]:
        """8-bit 그레이스케일 원본을 받아 결함 후보 목록을 반환한다.

        recall 우선(오탐 허용, 미탐 최소화). bbox 는 원본 픽셀 좌표(x1,y1,x2,y2).
        source='ai', status='proposed', id 는 models.new_id() 로 생성한다.
        """
        ...


# ---------------------------------------------------------------------------
# CV 폴백 탐지기
# ---------------------------------------------------------------------------


class CVFallbackDetector:
    """OpenCV 기반 폴백 탐지기.

    파이프라인:
      CLAHE → 약한 디노이즈 → 미디언 블러로 배경 추정 → 배경과의 차(어두운 쪽만)
      → 느슨한 임계(recall 우선) → 모폴로지 연결 → 연결 성분 분석
      → 컨투어 기하(원형도/종횡비/두께/대비)로 유형 추정 → 휴리스틱 confidence.

    recall 우선 설계: 임계는 느슨하게 잡고, confidence 순으로 상위 max_candidates 만
    제안하되, MIN_CONFIDENCE 미만의 저신뢰 후보는 제안하지 않는다(합성 필름 6장 실측:
    임계 0.5 에서 재현율 14/14, 장당 오탐 3.8건 — 임계 없이는 장당 오탐 27.7건).
    면적 6px 미만의 노이즈는 제외.
    """

    name = "cv-fallback"

    # 튜닝 파라미터 (합성/실 필름 공통의 보수적 기본값)
    MIN_AREA_PX = 6  # 이보다 작은 성분은 노이즈로 간주
    BG_MEDIAN_KSIZE = 31  # 배경 추정 미디언 블러 커널(홀수)
    DENOISE_SIGMA = 1.1  # 임계 전 가우시안 디노이즈
    THRESH_FLOOR = 8.0  # 임계 하한(느슨하게 — recall 우선)
    THRESH_SIGMA = 2.0  # 임계 = max(floor, mean + sigma*std)
    MIN_CONFIDENCE = 0.5  # 이 미만의 후보는 제안하지 않음 (recall 유지 + 오탐 억제 운영점)

    def detect(
        self,
        img_gray: np.ndarray,
        max_candidates: int = 30,
        min_confidence: float | None = None,
    ) -> list[DefectCandidate]:
        """min_confidence: None 이면 MIN_CONFIDENCE, 0.0 이면 필터 없이 상위 N 제안."""
        if not isinstance(img_gray, np.ndarray) or img_gray.ndim != 2 or img_gray.size == 0:
            raise ValueError("detect 입력은 크기 0이 아닌 2차원 그레이스케일 ndarray 여야 합니다.")
        if img_gray.dtype != np.uint8:
            img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

        h, w = img_gray.shape

        # 1) 대비 향상 + 약한 디노이즈 (노이즈 임계 초과율을 낮추면서 결함 신호는 보존)
        enh = apply_clahe(img_gray, clip_limit=3.0, tile_grid=8)
        den = cv2.GaussianBlur(enh, (0, 0), self.DENOISE_SIGMA)

        # 2) 배경 추정(미디언) → 어두운 쪽 차이만 취함 (RT 결함 지시는 어둡게 나타남)
        bg = cv2.medianBlur(den, self.BG_MEDIAN_KSIZE)
        diff = cv2.subtract(bg, den)  # uint8, 음수는 0

        # 3) 느슨한 적응 임계 — recall 우선
        d = diff.astype(np.float32)
        thr = max(self.THRESH_FLOOR, float(d.mean() + self.THRESH_SIGMA * d.std()))
        mask = (d >= thr).astype(np.uint8)

        # 4) 모폴로지 — 지그재그 균열 조각/군집 기공을 하나의 지시로 연결
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 5) 연결 성분 분석
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        band = self._estimate_bead_band(den)

        scored: list[tuple[float, DefectCandidate]] = []
        for i in range(1, num):
            x, y, cw, ch, area = (int(v) for v in stats[i])
            if area < self.MIN_AREA_PX:
                continue  # 너무 작은 노이즈 제외

            comp = (labels[y : y + ch, x : x + cw] == i).astype(np.uint8)
            contrast = float(d[y : y + ch, x : x + cw][comp > 0].mean())

            long_side, short_side, circularity = self._shape_stats(comp, area)
            elong = (long_side + 1.0) / (short_side + 1.0)

            defect_type = self._classify(
                area=area,
                long_side=long_side,
                short_side=short_side,
                elong=elong,
                circularity=circularity,
                contrast=contrast,
            )
            conf = self._confidence(area, contrast, cy=y + ch / 2.0, band=band)

            cand = DefectCandidate(
                id=new_id(),
                defect_type=defect_type,
                bbox=(float(x), float(y), float(min(x + cw, w)), float(min(y + ch, h))),
                confidence=conf,
                source="ai",
                status="proposed",
            )
            scored.append((conf * 1000.0 + math.sqrt(area), cand))

        # 6) 저신뢰 후보 제거 후 점수 순 상위 N 컷 (recall 우선이되 후보 폭주는 방지)
        min_conf = self.MIN_CONFIDENCE if min_confidence is None else float(min_confidence)
        scored = [t for t in scored if t[1].confidence >= min_conf]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [c for _, c in scored[: max(0, int(max_candidates))]]

    # -- 내부 유틸 ----------------------------------------------------------

    @staticmethod
    def _shape_stats(comp: np.ndarray, area: int) -> tuple[float, float, float]:
        """성분 마스크에서 (장축, 단축, 원형도)를 계산한다."""
        pts = np.column_stack(np.nonzero(comp))[:, ::-1].astype(np.float32)  # (x, y)
        if len(pts) >= 5:
            (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
            long_side, short_side = max(rw, rh), min(rw, rh)
        else:
            long_side = float(max(comp.shape))
            short_side = float(min(comp.shape))

        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        circularity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            perim = cv2.arcLength(cnt, True)
            if perim > 1e-6:
                # 픽셀 수 면적 기반 원형도 (얇은 성분에서 contourArea 과소평가 보정)
                circularity = float(min(1.0, 4.0 * math.pi * area / (perim * perim)))
        return float(long_side), float(short_side), circularity

    @staticmethod
    def _classify(
        *,
        area: int,
        long_side: float,
        short_side: float,
        elong: float,
        circularity: float,
        contrast: float,
    ) -> str:
        """컨투어 기하 기반 유형 추정 — 참고용 제안일 뿐, 최종 분류는 판독원이 한다."""
        # 종횡비 매우 크고 가늘며 대비 뚜렷 → 균열
        if elong >= 4.5 and short_side <= 13.0 and contrast >= 40.0:
            return "crack"
        # 길쭉하고 저대비 → 융합불량
        if elong >= 2.8 and contrast < 32.0:
            return "lack_of_fusion"
        # 원형도 높고 소형 → 기공
        if circularity >= 0.45 and long_side <= 26.0 and elong <= 2.4:
            return "porosity"
        return "unknown"

    @staticmethod
    def _confidence(
        area: int, contrast: float, cy: float, band: tuple[float, float] | None
    ) -> float:
        """면적/대비 기반 휴리스틱 confidence (0.3~0.9).

        용접 비드 밴드 안이면 소폭 가산 — 비드 밖 후보도 버리지는 않는다(recall 우선).
        """
        contrast_n = min(contrast / 60.0, 1.0)
        area_n = min(area / 400.0, 1.0)
        conf = 0.3 + 0.45 * contrast_n + 0.1 * area_n
        if band is not None and band[0] <= cy <= band[1]:
            conf += 0.05
        return float(np.clip(conf, 0.3, 0.9))

    @staticmethod
    def _estimate_bead_band(den: np.ndarray) -> tuple[float, float] | None:
        """행 평균 밝기로 용접 비드(밝은 수평 띠)의 세로 구간을 추정한다.

        실패하면 None — 이때는 밴드 가산 없이 전체 이미지 후보를 동일 취급.
        """
        h = den.shape[0]
        rows = den.mean(axis=1)
        thr = rows.mean() + 0.25 * rows.std()
        above = rows > thr
        if not above.any():
            return None
        # 가장 긴 연속 구간 탐색
        best_len, best_start = 0, -1
        cur_len, cur_start = 0, 0
        for i, v in enumerate(above):
            if v:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len < 10 or best_len > 0.8 * h:
            return None
        margin = 25.0
        return (best_start - margin, best_start + best_len + margin)


# ---------------------------------------------------------------------------
# YOLO 탐지기 (선택적 부품)
# ---------------------------------------------------------------------------


class YoloDetector:
    """Ultralytics YOLO 백엔드 (YOLO26s 권장 — ultralytics 8.4.x; YOLOv8/11 .pt도 동일 API).

    ultralytics 미설치 또는 가중치 부재는 예외가 아니라 '사용 불가'로 처리한다
    (available=False, detect()는 빈 목록). get_detector()가 폴백을 선택한다.
    """

    name = "yolo"

    # YOLO 클래스명(소문자) → 공용 결함 유형 코드 (부분 일치 별칭 포함)
    _CLASS_ALIASES: dict[str, str] = {
        "porosity": "porosity",
        "pore": "porosity",
        "gas": "porosity",
        "cluster": "cluster_porosity",
        "crack": "crack",
        "lack_of_fusion": "lack_of_fusion",
        "lof": "lack_of_fusion",
        "fusion": "lack_of_fusion",
        "incomplete_penetration": "incomplete_penetration",
        "penetration": "incomplete_penetration",
        "slag": "slag_inclusion",
        "inclusion": "slag_inclusion",
        "undercut": "undercut",
    }

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights_path = Path(weights_path) if weights_path else Path(config.YOLO_WEIGHTS_PATH)
        self._model = None
        self._available: bool | None = None
        self._note = ""

    @property
    def available(self) -> bool:
        """백엔드 사용 가능 여부 (최초 접근 시 1회 검사, 이후 캐시)."""
        if self._available is None:
            self._available = self._check()
        return self._available

    @property
    def note(self) -> str:
        """사용 가능/불가 사유 (UI 표시용 한국어)."""
        _ = self.available  # 검사 강제
        return self._note

    def _check(self) -> bool:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as exc:  # ImportError 포함 — 어떤 실패든 '사용 불가'
            self._note = f"ultralytics 미설치 또는 로드 실패({type(exc).__name__}) — CV 폴백 사용"
            return False
        if not self.weights_path.is_file():
            self._note = f"YOLO 가중치 없음({self.weights_path}) — CV 폴백 사용"
            return False
        try:
            self._model = YOLO(str(self.weights_path))
        except Exception as exc:
            self._note = f"YOLO 가중치 로드 실패({exc}) — CV 폴백 사용"
            return False
        self._note = f"YOLO 백엔드 사용 중 (가중치: {self.weights_path.name})"
        return True

    def detect(self, img_gray: np.ndarray, max_candidates: int = 30) -> list[DefectCandidate]:
        if not self.available:
            return []  # 사용 불가 — 예외를 던지지 않는다

        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # recall 우선: conf 임계를 낮게 잡고 상위 N 컷은 아래에서 수행
        results = self._model.predict(rgb, conf=0.10, verbose=False)

        cands: list[DefectCandidate] = []
        for res in results:
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_name = str(names.get(int(box.cls[0]), "unknown")).lower()
                cands.append(
                    DefectCandidate(
                        id=new_id(),
                        defect_type=self._map_class(cls_name),
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        source="ai",
                        status="proposed",
                    )
                )
        cands.sort(key=lambda c: c.confidence, reverse=True)
        return cands[: max(0, int(max_candidates))]

    @classmethod
    def _map_class(cls, cls_name: str) -> str:
        if cls_name in DEFECT_TYPES:
            return cls_name
        for alias, dtype in cls._CLASS_ALIASES.items():
            if alias in cls_name:
                return dtype
        return "unknown"


# ---------------------------------------------------------------------------
# 팩토리 / 상태
# ---------------------------------------------------------------------------


def get_detector() -> DetectorBackend:
    """YOLO 사용 가능하면 YoloDetector, 아니면 CVFallbackDetector 를 반환한다."""
    yolo = YoloDetector()
    if yolo.available:
        return yolo
    return CVFallbackDetector()


def detector_status() -> dict:
    """UI 표시용 백엔드 상태 요약."""
    yolo = YoloDetector()
    ok = yolo.available
    return {
        "backend": "yolo" if ok else "cv-fallback",
        "yolo_available": ok,
        "weights_path": str(config.YOLO_WEIGHTS_PATH),
        "note": yolo.note
        if ok
        else f"{yolo.note} · CV 폴백은 국소 대비 기반 안전망(recall 우선)입니다.",
    }
