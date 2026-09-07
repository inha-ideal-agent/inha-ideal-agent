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
from dataclasses import asdict, dataclass, field, fields
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


def criteria_label(name: str, version: str = "") -> str:
    """기준표 표기 '이름 (v버전)'. 이름이 없으면 ''(기준표 미기록 — 구버전 기록)."""
    if not name:
        return ""
    return f"{name} (v{version})" if version else name


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
    # 그룹 판정에서 선택된 '최악 평가 구간' [start, end] (mm, 용접선 축 좌표 — IACS UR W33 100 mm 판독 구간).
    # 위치 정보로 창 이동 판정을 했을 때만 채워진다. None = 단일 판정, 또는 위치 정보가 없어
    # 전체를 한 구간으로 합산한 그룹 판정(폴백). 구버전 기록에는 없으므로 기본값 None.
    window_start_mm: float | None = None
    window_end_mm: float | None = None

    @property
    def is_group(self) -> bool:
        """그룹 판정(유형별 누적 길이·투영 면적률) 행 여부."""
        return self.defect_id.startswith(GROUP_ID_PREFIX)

    @property
    def window_mm(self) -> tuple[float, float] | None:
        """창 이동 그룹 판정이 고른 최악 구간 (start, end) mm — 없으면 None."""
        if self.window_start_mm is None or self.window_end_mm is None:
            return None
        return (self.window_start_mm, self.window_end_mm)

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
        # 구버전 기록: unit이 없으면 'mm', window_*가 없으면 None(전체 구간 판정)
        return cls(**d)


@dataclass(frozen=True)
class ReportItem:
    """IACS UR W33 §8 보고 항목 1개의 메타데이터 — UI 라벨/PDF/payload/README가 공유한다."""

    key: str  # ReportDetails 필드명
    label_ko: str  # 한국어 라벨 (UI·PDF·payload)
    label_en: str  # W33 원문 표현 (UI help 텍스트·PDF 병기)
    group: str  # 'general'(§8.2 일반 항목) | 'rt'(§8.5 RT 촬영 조건)
    unit: str = ""  # 표시 단위 (예: 'mm', 'kV', 's', '°') — 값 뒤에 붙인다
    placeholder: str = ""  # UI 입력 예시


REPORT_GROUPS: dict[str, str] = {
    "general": "일반 항목 (§8.2)",
    "rt": "RT 촬영 조건 (§8.5)",
}

# IACS UR W33 Rev.1/Corr.1 §8.2(일반) 중 InspectionContext에 없는 항목 + §8.5(RT 전용) 전체.
# 이미 InspectionContext/InspectionRecord에 있는 §8.2 항목(검사일=created_at, 용접부 위치=block/weld_id,
# 이음 종류, 모재 두께, 합격 기준=quality_level+criteria_*, 결과=verdicts, 합부=overall_passed,
# 평가자=inspector)은 여기서 중복 입력받지 않는다.
REPORT_ITEMS: tuple[ReportItem, ...] = (
    # ---- §8.2 일반 항목
    ReportItem("hull_number", "선체 번호", "Hull number", "general", placeholder="예: H-2031"),
    ReportItem("weld_length_mm", "검사 용접 길이", "Length of weld inspected", "general",
               unit="mm", placeholder="예: 300"),
    ReportItem("personnel_qualification", "검사자 자격 등급",
               "Qualification level of personnel (name/signature excluded here)", "general",
               placeholder="예: ISO 9712 RT Level 2"),
    ReportItem("steel_grade", "강재 등급", "Steel grade", "general", placeholder="예: AH36"),
    ReportItem("welding_process", "용접 방법", "Welding process", "general",
               placeholder="예: FCAW (136)"),
    ReportItem("testing_standard", "검사 규격", "Testing standards used", "general",
               placeholder="예: ISO 17636-1 Class B"),
    ReportItem("equipment", "검사 장비·배치", "Testing equipment and arrangement used", "general",
               placeholder="예: Ir-192 감마선 장비, 단벽 단상"),
    ReportItem("limitations_viewing", "제한사항·관찰 조건·온도",
               "Test limitations, viewing conditions and temperature", "general",
               placeholder="예: 제한 없음, 판독기 휘도 적합, 18℃"),
    ReportItem("repairs_count", "보수 횟수(2회 초과 시)",
               "Number of repairs if specific area repaired more than twice", "general",
               unit="회"),
    # ---- §8.5 RT 전용 항목
    ReportItem("source_type_size", "선원 종류·크기", "Type and size of radiation source", "rt",
               placeholder="예: Ir-192, 2.0×2.0 mm"),
    ReportItem("xray_kv", "X선 관전압", "X-ray voltage", "rt", unit="kV", placeholder="예: 200"),
    ReportItem("film_type", "필름 종류·카세트당 매수",
               "Type of film/designation and number of film per cassette", "rt",
               placeholder="예: AGFA D4, 1매"),
    ReportItem("exposures_count", "촬영 매수", "Number of radiographs (exposures)", "rt",
               unit="매", placeholder="예: 3"),
    ReportItem("screens", "증감지 종류", "Type of intensifying screens", "rt",
               placeholder="예: Pb 0.1 mm 전·후면"),
    ReportItem("exposure_technique", "촬영 기법", "Exposure technique", "rt",
               placeholder="예: 단벽 단상 (SWSI)"),
    ReportItem("exposure_time_s", "노출 시간", "Time of exposure", "rt", unit="s",
               placeholder="예: 90"),
    ReportItem("sfd_mm", "선원-필름 거리(SFD)", "Source-to-film distance", "rt", unit="mm",
               placeholder="예: 700"),
    ReportItem("source_to_weld_mm", "선원-용접부 거리", "Distance from radiation source to weld",
               "rt", unit="mm", placeholder="예: 688"),
    ReportItem("weld_to_film_mm", "용접부-필름 거리",
               "Distance from source side of the weld to radiographic film", "rt", unit="mm",
               placeholder="예: 12"),
    ReportItem("beam_angle_deg", "빔 입사각(법선 기준)",
               "Angle of radiation beam through the weld (from normal)", "rt", unit="°",
               placeholder="예: 0"),
    ReportItem("iqi_sensitivity", "IQI 감도", "IQI sensitivity", "rt",
               placeholder="예: W13 (1.6%)"),
    ReportItem("iqi_type_position", "IQI 종류·위치",
               "Type and position of IQI (source side or film side)", "rt",
               placeholder="예: ISO 19232-1 선형, 선원측"),
    ReportItem("density", "농도", "Density", "rt", placeholder="예: 2.3~2.8"),
    ReportItem("geometric_unsharpness", "기하학적 불선명도", "Geometric un-sharpness", "rt",
               unit="mm", placeholder="예: 0.03"),
    ReportItem("rt_acceptance_class", "RT 합격 등급", "Specific acceptance class criteria for RT",
               "rt", placeholder="예: ISO 10675-1 Level 1"),
)

REPORT_ITEM_TOTAL: int = len(REPORT_ITEMS)  # 25 — UI 'n/25 입력' 표기의 분모

# 값에 붙여 쓰는 단위 (한글 조수사·도 기호). 나머지(mm·kV·s)는 한 칸 띄운다.
_ATTACHED_UNITS: frozenset[str] = frozenset({"회", "매", "°"})


@dataclass
class ReportDetails:
    """IACS UR W33 §8.2·8.5 선급 보고 항목 (선택 입력).

    판정에는 관여하지 않고 문서(PDF·소견서 payload·아카이브)에만 실린다.
    필드 순서/이름은 REPORT_ITEMS 와 1:1 대응한다. 수치 항목도 자유 서식 문자열로 받는다
    (예: '2.3~2.8', '700') — 보고서 그대로 옮겨 적는 용도이며 계산에 쓰지 않는다.
    """

    # §8.2 일반 항목 (InspectionContext에 없는 것)
    hull_number: str = ""
    weld_length_mm: str = ""
    personnel_qualification: str = ""
    steel_grade: str = ""
    welding_process: str = ""
    testing_standard: str = ""
    equipment: str = ""
    limitations_viewing: str = ""
    repairs_count: int = 0  # 특정 부위 2회 초과 보수 시 횟수 — 0 = 해당 없음(미입력)
    # §8.5 RT 전용 항목
    source_type_size: str = ""
    xray_kv: str = ""
    film_type: str = ""
    exposures_count: str = ""
    screens: str = ""
    exposure_technique: str = ""
    exposure_time_s: str = ""
    sfd_mm: str = ""
    source_to_weld_mm: str = ""
    weld_to_film_mm: str = ""
    beam_angle_deg: str = ""
    iqi_sensitivity: str = ""
    iqi_type_position: str = ""
    density: str = ""
    geometric_unsharpness: str = ""
    rt_acceptance_class: str = ""

    @staticmethod
    def _is_filled(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0  # repairs_count: 0 = 해당 없음
        return bool(str(value or "").strip())

    def value_text(self, item: ReportItem) -> str:
        """항목 값의 표시 문자열 (단위 포함). 미입력이면 ''."""
        raw = getattr(self, item.key, "")
        if not self._is_filled(raw):
            return ""
        txt = str(raw).strip()
        if item.unit and not txt.endswith(item.unit):
            # 한글 조수사(회·매)·도(°)는 붙여 쓰고, 단위 기호(mm·kV·s)는 띄어 쓴다
            sep = "" if item.unit in _ATTACHED_UNITS else " "
            txt = f"{txt}{sep}{item.unit}"
        return txt

    def filled_items(self) -> list[tuple[ReportItem, str]]:
        """입력된 항목만 (REPORT_ITEMS 순서) → [(item, 표시 문자열)]."""
        out: list[tuple[ReportItem, str]] = []
        for item in REPORT_ITEMS:
            txt = self.value_text(item)
            if txt:
                out.append((item, txt))
        return out

    def coverage(self) -> tuple[int, int]:
        """(입력된 항목 수, 전체 항목 수) — UI '선급 보고 항목 n/25 입력' 표기용."""
        return len(self.filled_items()), REPORT_ITEM_TOTAL

    def is_empty(self) -> bool:
        return not self.filled_items()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ReportDetails":
        """구버전 기록(키 없음/None)·미지의 키를 모두 허용한다 — 없는 항목은 기본값."""
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for k, v in d.items():
            if k not in known:
                continue  # 이후 판본에서 추가된 키는 무시
            if k == "repairs_count":
                try:
                    kwargs[k] = int(v or 0)
                except (TypeError, ValueError):
                    kwargs[k] = 0
            else:
                kwargs[k] = "" if v is None else str(v)
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# IACS UR W33 §8.2·§8.5 보고 항목 충족률 — 원문 불릿 25개(일반 13 + RT 12) 단위
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class W33Item:
    """IACS UR W33 §8.2/§8.5 원문 불릿 1개 — 충족률 'n/25'의 분모 단위.

    ReportDetails 필드(25개 입력 필드)와는 다르다: W33 불릿 하나가 여러 입력 필드를 묶기도 하고
    (촬영 기법+노출 시간+SFD), 검사 컨텍스트/판정 기록만으로 충족되는 불릿도 있다(검사일, 합격 기준…).
    """

    key: str
    label_ko: str  # 한국어 라벨 (UI 캡션 '미입력: …' · PDF)
    label_en: str  # W33 원문 불릿
    group: str  # 'general'(§8.2) | 'rt'(§8.5)
    rule_ko: str  # 충족 규칙(사람이 읽는 설명 — README·도움말과 동일)


# 충족 규칙 원칙: 불릿의 하위 정보 중 워크벤치가 담을 수 있는 것이 **모두** 있을 때 충족.
# 검사 컨텍스트/판정/승인 기록에 이미 있는 정보는 자동 충족(중복 입력 없음). 조건부 항목(보수 횟수)은
# 기본값 0(해당 없음)과 미입력을 구분할 수 없어 1 이상일 때만 충족(보수적 — 과대 계상 금지).
W33_ITEMS: tuple[W33Item, ...] = (
    # ---- §8.2 일반 항목 13개
    W33Item("date_of_testing", "검사일", "Date of testing", "general",
            "승인 시각(created_at)이 자동 기록 → 항상 충족"),
    W33Item("hull_location_length", "선체 번호·용접부 위치·검사 길이",
            "Hull number, location and length of weld inspected", "general",
            "선체 번호 + 위치(블록 또는 용접부 ID) + 검사 용접 길이 모두"),
    W33Item("personnel", "검사자 성명·자격 등급(·서명)",
            "Names, qualification level and signature of personnel that have performed the testing",
            "general", "판독원(성명) + 검사자 자격 등급 모두 — 서명은 출력물에 수기"),
    W33Item("component_id", "검사 부재 식별", "Identification of the component examined", "general",
            "블록"),
    W33Item("weld_id", "검사 용접부 식별", "Identification of the welds examined", "general",
            "용접부 ID"),
    W33Item("material_joint_process", "강재 등급·이음 종류·모재 두께·용접 방법",
            "Steel grade, type of joint, thickness of parent material, welding process", "general",
            "강재 등급 + 용접 방법 모두 (이음 종류·모재 두께는 컨텍스트 필수값 → 자동)"),
    W33Item("acceptance_criteria", "합격 기준", "Acceptance criteria", "general",
            "품질등급 + 적용 기준표(JSON 이름·버전) 항상 기록 → 항상 충족"),
    W33Item("testing_standards", "적용 검사 규격", "Testing standards used", "general",
            "검사 규격 — 기법명('RT (필름 스캔)')은 규격이 아니므로 제외"),
    W33Item("equipment", "검사 장비·배치", "Testing equipment and arrangement used", "general",
            "검사 장비·배치"),
    W33Item("limitations_viewing", "제한사항·관찰 조건·온도",
            "Any test limitations, viewing conditions and temperature", "general",
            "제한사항·관찰 조건·온도"),
    W33Item("results", "검사 결과(합격 기준 대비 지시 위치·크기)",
            "Results of testing with reference to acceptance criteria, location and size of "
            "reportable indications", "general",
            "판정 실행 후(verdict 표 — 결함 없음도 결과) 충족"),
    W33Item("acceptance_statement", "합부 선언·평가일·평가자",
            "Statement of acceptance / non-acceptance, evaluation date, name and signature of "
            "evaluator", "general",
            "판정 실행 + 판독원(성명) 모두 — 평가일은 승인 시각 자동"),
    W33Item("repairs", "보수 횟수(2회 초과 보수 시)",
            "Number of repairs if specific area repaired more than twice", "general",
            "보수 횟수 1 이상 — 조건부 항목, 0(해당 없음)은 기본값과 구분 불가라 미충족 처리"),
    # ---- §8.5 RT 전용 항목 12개
    W33Item("source_voltage", "선원 종류·크기(·X선 관전압)",
            "Type and size of radiation source (width of radiation source), X-ray voltage", "rt",
            "선원 종류·크기 — X선 관전압은 X선 장비일 때만 해당(감마선은 해당 없음)이라 필수 아님"),
    W33Item("film", "필름 종류·카세트당 매수",
            "Type of film/designation and number of film in each film holder/cassette", "rt",
            "필름 종류·카세트당 매수"),
    W33Item("exposures", "촬영 매수", "Number of radiographs (exposures)", "rt", "촬영 매수"),
    W33Item("screens", "증감지 종류", "Type of intensifying screens", "rt", "증감지 종류"),
    W33Item("exposure_technique_time_sfd", "촬영 기법·노출 시간·SFD",
            "Exposure technique, time of exposure and source-to-film distance", "rt",
            "촬영 기법 + 노출 시간 + 선원-필름 거리(SFD) 모두"),
    W33Item("source_to_weld", "선원-용접부 거리", "Distance from radiation source to weld", "rt",
            "선원-용접부 거리"),
    W33Item("weld_to_film", "용접부-필름 거리",
            "Distance from source side of the weld to radiographic film", "rt", "용접부-필름 거리"),
    W33Item("beam_angle", "빔 입사각(법선 기준)",
            "Angle of radiation beam through the weld (from normal)", "rt", "빔 입사각"),
    W33Item("iqi", "IQI 감도·종류·위치",
            "Sensitivity, type and position of IQI (source side or film side)", "rt",
            "IQI 감도 + IQI 종류·위치 모두"),
    W33Item("density", "농도", "Density", "rt", "농도"),
    W33Item("geometric_unsharpness", "기하학적 불선명도", "Geometric un-sharpness", "rt",
            "기하학적 불선명도"),
    W33Item("rt_acceptance_class", "RT 합격 등급", "Specific acceptance class criteria for RT", "rt",
            "RT 합격 등급"),
)

W33_ITEM_TOTAL: int = len(W33_ITEMS)  # 25 = §8.2 일반 13 + §8.5 RT 12 — UI 'n/25 충족'의 분모
W33_GROUP_COUNTS: dict[str, int] = {
    g: sum(1 for i in W33_ITEMS if i.group == g) for g in ("general", "rt")
}  # {'general': 13, 'rt': 12}


def _txt(value: Any) -> bool:
    return bool(str(value or "").strip())


def w33_item_filled(key: str, context: "InspectionContext", *, judged: bool = False) -> bool:
    """W33 불릿 1개의 충족 여부 — W33_ITEMS[*].rule_ko 와 1:1로 대응하는 결정론적 규칙.

    judged: 판정 실행 여부(verdict 표 존재). 승인 기록(InspectionRecord)이 있으면 항상 True.
    """
    r = context.report
    f = ReportDetails._is_filled
    location = _txt(context.block) or _txt(context.weld_id)
    rules: dict[str, Any] = {
        "date_of_testing": lambda: True,
        "hull_location_length": lambda: f(r.hull_number) and location and f(r.weld_length_mm),
        "personnel": lambda: _txt(context.inspector) and f(r.personnel_qualification),
        "component_id": lambda: _txt(context.block),
        "weld_id": lambda: _txt(context.weld_id),
        "material_joint_process": lambda: f(r.steel_grade) and f(r.welding_process),
        "acceptance_criteria": lambda: True,
        "testing_standards": lambda: f(r.testing_standard),
        "equipment": lambda: f(r.equipment),
        "limitations_viewing": lambda: f(r.limitations_viewing),
        "results": lambda: bool(judged),
        "acceptance_statement": lambda: bool(judged) and _txt(context.inspector),
        "repairs": lambda: f(r.repairs_count),
        "source_voltage": lambda: f(r.source_type_size),
        "film": lambda: f(r.film_type),
        "exposures": lambda: f(r.exposures_count),
        "screens": lambda: f(r.screens),
        "exposure_technique_time_sfd": lambda: (
            f(r.exposure_technique) and f(r.exposure_time_s) and f(r.sfd_mm)
        ),
        "source_to_weld": lambda: f(r.source_to_weld_mm),
        "weld_to_film": lambda: f(r.weld_to_film_mm),
        "beam_angle": lambda: f(r.beam_angle_deg),
        "iqi": lambda: f(r.iqi_sensitivity) and f(r.iqi_type_position),
        "density": lambda: f(r.density),
        "geometric_unsharpness": lambda: f(r.geometric_unsharpness),
        "rt_acceptance_class": lambda: f(r.rt_acceptance_class),
    }
    try:
        return bool(rules[key]())
    except KeyError:
        raise KeyError(f"알 수 없는 W33 항목 키: {key!r}") from None


def w33_coverage(
    context: "InspectionContext",
    record: "InspectionRecord | None" = None,
    *,
    judged: bool | None = None,
) -> tuple[int, int, list[str]]:
    """IACS UR W33 §8.2·8.5 보고 항목 충족률 → (충족 수, 25, 미충족 라벨 목록[W33_ITEMS 순서]).

    - record 가 주어지면 승인 기록 기준(판정 완료로 간주, context 는 record.context 를 써도 됨).
    - record 없이 작업 중인 검사(사이드바)라면 judged 로 판정 실행 여부를 넘긴다(기본 False).
    - 분모는 항상 25(원문 불릿 수). 구버전 기록도 컨텍스트만으로 계산되어 예외가 없다.
    """
    if judged is None:
        judged = record is not None
    missing = [
        item.label_ko for item in W33_ITEMS if not w33_item_filled(item.key, context, judged=judged)
    ]
    return W33_ITEM_TOTAL - len(missing), W33_ITEM_TOTAL, missing


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
    # 선급 보고 항목(IACS UR W33 §8.2·8.5) — 판정에 관여하지 않는 문서 전용 필드
    report: ReportDetails = field(default_factory=ReportDetails)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # report 는 중첩 dict 로 직렬화된다

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InspectionContext":
        # 구버전 기록에 eval_length_mm/weld_width_mm/report 가 없으면 기본값
        d = dict(d)
        report = d.pop("report", None)
        ctx = cls(**d)
        ctx.report = report if isinstance(report, ReportDetails) else ReportDetails.from_dict(report)
        return ctx


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
    # 판정에 적용된 기준표(규격 판본) — RuleEngine.criteria_name/criteria_version.
    # 구버전 기록(판본 분리 전)에는 없으므로 ''(미기록)을 허용한다.
    criteria_name: str = ""
    criteria_version: str = ""

    @property
    def criteria_label(self) -> str:
        """적용 기준표 표기 '이름 (v버전)' — 미기록이면 ''."""
        return criteria_label(self.criteria_name, self.criteria_version)

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
            "criteria_name": self.criteria_name,
            "criteria_version": self.criteria_version,
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
            # 판본 분리 전 기록엔 없음 — ''(미기록) 허용
            criteria_name=str(d.get("criteria_name") or ""),
            criteria_version=str(d.get("criteria_version") or ""),
        )
