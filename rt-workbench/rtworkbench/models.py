"""RT 판독 워크벤치 — 공용 데이터 모델.

모든 모듈(탐지, 측정, 룰 판정, 소견서, 아카이브, UI)이 공유하는 계약(contract).
이 파일의 시그니처를 바꾸면 전체 모듈이 함께 바뀌어야 하므로 신중히 수정할 것.

설계 원칙 (계획서 §3):
  1. AI는 판정하지 않는다 — 합부는 룰 엔진 + 판독원, LLM은 설명/작문 전담.
  2. 기존 이미지·기존 절차 위에 — 입력은 스캔된 필름 이미지 또는 디지털 RT 이미지.
  3. 탐지는 부품이다 — 신규성은 판독 워크플로우 전체의 통합.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

# 결함 유형 코드 → 한국어 표기 (RT에서 관찰되는 대표 유형)
DEFECT_TYPES: dict[str, str] = {
    "porosity": "기공",
    "cluster_porosity": "군집 기공",
    "crack": "균열",
    "lack_of_fusion": "융합불량",
    "incomplete_penetration": "용입부족",
    "slag_inclusion": "슬래그 개재물",
    "undercut": "언더컷",
    "unknown": "미분류 지시",
}

QUALITY_LEVELS: tuple[str, ...] = ("B", "C", "D")  # ISO 5817 계열: B(엄격) > C > D

CANDIDATE_STATUS = ("proposed", "accepted", "rejected")  # AI 제안 / 판독원 채택 / 기각

# 그룹 판정(누적 길이·투영 면적률) verdict의 defect_id 접두어 — "GROUP:<defect_type>"
GROUP_ID_PREFIX = "GROUP:"


def new_id(prefix: str = "df") -> str:
    """짧은 고유 ID 생성 (UI 표시용)."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class DefectCandidate:
    """결함 후보 1건. AI(2차 눈)가 제안하거나 판독원이 직접 추가한다."""

    id: str
    defect_type: str  # DEFECT_TYPES의 키
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) — 원본 이미지 픽셀 좌표
    confidence: float  # 0.0~1.0, 판독원 직접 추가 시 1.0
    source: str = "ai"  # 'ai' | 'human'
    status: str = "proposed"  # CANDIDATE_STATUS 중 하나
    note: str = ""

    @property
    def type_ko(self) -> str:
        return DEFECT_TYPES.get(self.defect_type, self.defect_type)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DefectCandidate":
        d = dict(d)
        d["bbox"] = tuple(d["bbox"])
        return cls(**d)


@dataclass
class Measurement:
    """판독원의 2클릭 자(ruler) 측정 결과. 사람이 확정한 값만 존재한다."""

    defect_id: str  # 대상 DefectCandidate.id
    p1: tuple[float, float]  # 원본 이미지 픽셀 좌표
    p2: tuple[float, float]
    length_px: float
    length_mm: float | None  # 캘리브레이션 전이면 None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["p1"], d["p2"] = list(self.p1), list(self.p2)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Measurement":
        d = dict(d)
        d["p1"], d["p2"] = tuple(d["p1"]), tuple(d["p2"])
        return cls(**d)


@dataclass
class RuleVerdict:
    """결정론적 룰 엔진의 판정 결과 1건. LLM은 이 값을 생성/수정할 수 없다."""

    defect_id: str
    defect_type: str
    size_mm: float
    quality_level: str  # 'B' | 'C' | 'D'
    thickness_mm: float
    limit_mm: float | None  # None → '허용 불가' 유형 (예: 균열)
    passed: bool
    clause: str  # 근거 조항 표기 (데모 기준표의 항목 ID)
    detail: str  # 판정 근거 설명 (한국어, 룰 엔진이 결정론적으로 생성)
    unit: str = "mm"  # size_mm/limit_mm의 단위: 'mm' | '%'(투영 면적률 그룹 판정)

    @property
    def is_group(self) -> bool:
        """그룹 판정(유형별 누적 길이·투영 면적률) 행 여부."""
        return self.defect_id.startswith(GROUP_ID_PREFIX)

    @property
    def type_ko(self) -> str:
        return DEFECT_TYPES.get(self.defect_type, self.defect_type)

    @property
    def display_id(self) -> str:
        """표시용 ID — 그룹 판정은 '합계(기공)' 형태, 단일 판정은 defect_id 그대로."""
        return f"합계({self.type_ko})" if self.is_group else self.defect_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuleVerdict":
        return cls(**d)  # 구버전 기록에 unit이 없으면 기본값 'mm'


@dataclass
class InspectionContext:
    """검사 1건의 메타데이터. 판독원이 사이드바에서 입력한다."""

    film_id: str = ""
    block: str = ""  # 예: "3번 블록"
    weld_id: str = ""  # 용접 이음부 식별자
    joint_type: str = "맞대기(Butt)"
    thickness_mm: float = 12.0
    quality_level: str = "B"
    inspector: str = ""
    technique: str = "RT (필름 스캔)"
    scale_mm_per_px: float | None = None  # 2클릭 캘리브레이션 결과
    scale_ref: str = ""  # 캘리브레이션 기준물 (예: "납마커 10mm")
    eval_length_mm: float = 100.0  # 평가 길이 — 누적 길이·면적률 그룹 판정의 기준 구간
    weld_width_mm: float = 20.0  # 용접부 폭 — 투영 면적(평가 길이 × 폭) 산정용

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InspectionContext":
        return cls(**d)  # 구버전 기록에 eval_length_mm/weld_width_mm가 없으면 기본값


@dataclass
class InspectionRecord:
    """승인 완료된 검사 1건 전체 — 아카이브(DB) 저장 단위이자 자기개선 루프의 라벨 원천."""

    record_id: str
    context: InspectionContext
    candidates: list[DefectCandidate]
    measurements: list[Measurement]
    verdicts: list[RuleVerdict]
    overall_passed: bool
    report_text: str  # 판독원이 승인한 최종 소견서 본문
    report_source: str  # 'claude' | 'gemini' | 'template' | 'cache' — 초안 생성 경로
    image_name: str  # 원본 파일명 (이미지 자체는 로컬 보관, DB에는 경로/이름만)
    image_size: tuple[int, int]  # (width, height) px — 라벨 export 시 정규화에 필요
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    elapsed_seconds: float | None = None  # 이미지 로드→승인 소요시간 (창출 효과 정량화)

    def to_json(self) -> str:
        d = {
            "record_id": self.record_id,
            "context": self.context.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "measurements": [m.to_dict() for m in self.measurements],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "overall_passed": self.overall_passed,
            "report_text": self.report_text,
            "report_source": self.report_source,
            "image_name": self.image_name,
            "image_size": list(self.image_size),
            "created_at": self.created_at,
            "elapsed_seconds": self.elapsed_seconds,
        }
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "InspectionRecord":
        d = json.loads(s)
        return cls(
            record_id=d["record_id"],
            context=InspectionContext.from_dict(d["context"]),
            candidates=[DefectCandidate.from_dict(c) for c in d["candidates"]],
            measurements=[Measurement.from_dict(m) for m in d["measurements"]],
            verdicts=[RuleVerdict.from_dict(v) for v in d["verdicts"]],
            overall_passed=d["overall_passed"],
            report_text=d["report_text"],
            report_source=d["report_source"],
            image_name=d["image_name"],
            image_size=tuple(d["image_size"]),
            created_at=d["created_at"],
            elapsed_seconds=d.get("elapsed_seconds"),  # 구버전 백업엔 없음 — None 허용
        )
