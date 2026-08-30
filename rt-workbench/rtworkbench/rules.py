"""결정론적 룰 엔진 — 합부 판정. AI는 여기 관여하지 않는다.

구현 계약:

class RuleEngine:
    def __init__(self, criteria_path: str | Path = config.CRITERIA_PATH): ...
        '''criteria JSON 로드. 파일 형식 오류 시 ValueError.'''

    @property
    def meta(self) -> dict: ...
        '''기준표 meta 블록 (이름/버전/디스클레이머).'''

    def evaluate(self, defect_id: str, defect_type: str, size_mm: float,
                 thickness_mm: float, quality_level: str) -> RuleVerdict:
        '''단일 결함 판정.
        - permitted=false → limit_mm=None, passed=False
        - permitted=true → limit_mm = min(coef_thickness*t, cap_mm) 반올림 2자리,
          passed = size_mm <= limit_mm
        - 미등재 유형 → default_rule fail-safe: passed=False, limit_mm=None,
          detail에 수동 판정 필요 명시
        - quality_level이 B/C/D 외 → ValueError
        - size_mm < 0 또는 thickness_mm <= 0 → ValueError
        - detail은 한국어로 근거를 서술 (예: "두께 12.0mm × 계수 0.2 = 2.4mm ≤ cap 3.0mm
          → 허용 한계 2.4mm, 측정값 3.1mm 초과 → 불합격").'''

    def evaluate_all(self, items: list[tuple[str, str, float]],
                     thickness_mm: float, quality_level: str) -> list[RuleVerdict]:
        '''items = [(defect_id, defect_type, size_mm), ...] 일괄 판정.'''

def overall_pass(verdicts: list[RuleVerdict]) -> bool:
    '''전체 합부: 모든 verdict가 passed일 때만 True. 빈 목록은 True(결함 없음).'''

────────────────────────────────────────────────────────────────────
반올림/경계 판정 정책 (부동소수 오차와 모순되지 않도록 고정):

  1) 허용 한계는 먼저 확정한다:  limit_mm = round(min(coef × t, cap), 2)
     — verdict에 저장되는 한계값과 비교에 쓰는 한계값이 항상 동일하다.
  2) 측정값도 같은 자릿수로 맞춘 뒤 비교한다:
       passed = round(size_mm, 2) <= limit_mm
     — 예: 두께 12mm × 계수 0.2는 이진 부동소수로 2.4000000000000004가 되는데,
       한계만 2.4로 반올림하고 측정값 2.4000000000000004를 그대로 비교하면
       '같은 값인데 불합격'이라는 모순이 생긴다. 양쪽을 0.01mm 단위로 정렬해
       경계 동치(측정 == 한계)가 반드시 합격이 되도록 한다.
  3) verdict.size_mm에는 비교에 실제 사용한 값 round(size_mm, 2)를 저장한다.
     — 저장된 (size_mm, limit_mm, passed) 세 값만 보고도
       passed == (size_mm <= limit_mm)가 항상 성립한다(자기모순 없는 기록).
     0.01mm(=10μm)는 필름 스캔 측정의 물리적 정밀도보다 충분히 촘촘하다.
"""

from __future__ import annotations

import json
from pathlib import Path

from rtworkbench import config
from rtworkbench.models import QUALITY_LEVELS, RuleVerdict


def _fmt_mm(x: float) -> str:
    """mm 수치 표기: 소수 최대 2자리, 최소 1자리 (2.40→'2.4', 3.00→'3.0', 2.41→'2.41')."""
    s = f"{x:.2f}"
    return s[:-1] if s.endswith("0") else s


class RuleEngine:
    """데이터 주도 결정론적 룰 엔진.

    기준표 JSON(criteria)만 교체하면 실제 규격으로 전환 가능 — 코드 무변경.
    LLM/탐지기는 이 클래스의 입력에도 출력에도 관여하지 않는다.
    """

    def __init__(self, criteria_path: str | Path = config.CRITERIA_PATH):
        path = Path(criteria_path)
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"기준표 JSON 파싱 실패: {path} — {e}") from e

        if not isinstance(data, dict):
            raise ValueError(f"기준표 형식 오류: 최상위가 객체(dict)가 아닙니다 — {path}")
        for key in ("meta", "rules", "default_rule"):
            if key not in data:
                raise ValueError(f"기준표 형식 오류: 필수 블록 '{key}' 누락 — {path}")

        if not isinstance(data["meta"], dict):
            raise ValueError(f"기준표 형식 오류: 'meta'가 객체(dict)가 아닙니다 — {path}")

        default = data["default_rule"]
        if not isinstance(default, dict):
            raise ValueError(
                f"기준표 형식 오류: 'default_rule'이 객체(dict)가 아닙니다 — {path}"
            )
        for key in ("clause", "detail"):
            if not isinstance(default.get(key), str):
                raise ValueError(
                    f"기준표 형식 오류: 'default_rule'에 문자열 '{key}'가 없습니다 — {path}"
                )

        quality_levels = tuple(data["meta"].get("quality_levels", QUALITY_LEVELS))
        self._validate_rules(data["rules"], quality_levels, path)

        self._data: dict = data
        self._quality_levels: tuple[str, ...] = quality_levels

    @staticmethod
    def _validate_rules(rules: dict, quality_levels: tuple[str, ...], path: Path) -> None:
        """규칙 블록의 구조 검증. permitted=true인데 계수/cap이 없으면 형식 오류.

        선언된 모든 품질등급(quality_levels)이 각 규칙의 levels에 존재해야 한다 —
        누락 시 evaluate 도중 KeyError가 아니라 로드 시점 ValueError로 거부한다.
        """
        if not isinstance(rules, dict):
            raise ValueError(f"기준표 형식 오류: 'rules'가 객체(dict)가 아닙니다 — {path}")
        for dtype, rule in rules.items():
            if not isinstance(rule, dict):
                raise ValueError(
                    f"기준표 형식 오류: '{dtype}' 규칙이 객체(dict)가 아닙니다 — {path}"
                )
            levels = rule.get("levels")
            if not isinstance(levels, dict) or "clause" not in rule:
                raise ValueError(
                    f"기준표 형식 오류: '{dtype}'에 clause/levels가 없습니다 — {path}"
                )
            missing = [lv for lv in quality_levels if lv not in levels]
            if missing:
                raise ValueError(
                    f"기준표 형식 오류: '{dtype}'의 levels에 선언된 품질등급 "
                    f"{'/'.join(missing)} 항목이 없습니다 — {path}"
                )
            for lv, entry in levels.items():
                if not isinstance(entry, dict) or "permitted" not in entry:
                    raise ValueError(
                        f"기준표 형식 오류: '{dtype}' 등급 '{lv}'에 permitted가 없습니다 — {path}"
                    )
                if entry["permitted"]:
                    for k in ("coef_thickness", "cap_mm"):
                        if not isinstance(entry.get(k), (int, float)):
                            raise ValueError(
                                f"기준표 형식 오류: '{dtype}' 등급 '{lv}'의 '{k}'가 "
                                f"숫자가 아닙니다 — {path}"
                            )

    @property
    def meta(self) -> dict:
        """기준표 meta 블록 (이름/버전/디스클레이머)."""
        return self._data["meta"]

    def evaluate(
        self,
        defect_id: str,
        defect_type: str,
        size_mm: float,
        thickness_mm: float,
        quality_level: str,
    ) -> RuleVerdict:
        """단일 결함 합부 판정 (결정론적 — 같은 입력이면 항상 같은 출력)."""
        # ── 입력 검증 (fail-safe 경로라도 입력이 틀리면 판정 자체를 거부한다) ──
        if quality_level not in self._quality_levels:
            raise ValueError(
                f"품질등급은 {'/'.join(self._quality_levels)} 중 하나여야 합니다: "
                f"'{quality_level}'"
            )
        if size_mm < 0:
            raise ValueError(f"측정 크기(size_mm)는 0 이상이어야 합니다: {size_mm}")
        if thickness_mm <= 0:
            raise ValueError(f"모재 두께(thickness_mm)는 0보다 커야 합니다: {thickness_mm}")

        # 반올림 정책 (모듈 docstring 참조): 측정값을 0.01mm 단위로 정렬해 저장/비교
        size_r = round(size_mm, 2)

        rule = self._data["rules"].get(defect_type)

        # ── 미등재 유형 → default_rule fail-safe: 자동 합격 처리 금지 ──
        if rule is None:
            default = self._data["default_rule"]
            return RuleVerdict(
                defect_id=defect_id,
                defect_type=defect_type,
                size_mm=size_r,
                quality_level=quality_level,
                thickness_mm=thickness_mm,
                limit_mm=None,
                passed=False,
                clause=default["clause"],
                detail=f"'{defect_type}' — {default['detail']}",
            )

        name_ko = rule.get("name_ko", defect_type)
        clause = rule["clause"]
        level = rule["levels"][quality_level]

        # ── 불허 유형: 크기와 무관하게 불합격, 한계 없음 ──
        if not level["permitted"]:
            return RuleVerdict(
                defect_id=defect_id,
                defect_type=defect_type,
                size_mm=size_r,
                quality_level=quality_level,
                thickness_mm=thickness_mm,
                limit_mm=None,
                passed=False,
                clause=clause,
                detail=(
                    f"품질등급 {quality_level}에서 {name_ko}은(는) "
                    f"크기와 무관하게 허용되지 않음 → 불합격"
                ),
            )

        # ── 허용 유형: limit = round(min(coef×t, cap), 2), 비교는 size_r <= limit ──
        coef = level["coef_thickness"]
        cap = level["cap_mm"]
        by_thickness = coef * thickness_mm  # 두께 기반 원시 한계 (반올림 전)
        limit = round(min(by_thickness, cap), 2)  # 정책 1) 한계 먼저 확정
        passed = size_r <= limit  # 정책 2) 정렬된 측정값과 비교 — 경계 동치는 합격

        cmp_sym = "≤" if passed else ">"
        verdict_ko = "합격" if passed else "불합격"
        detail = (
            f"품질등급 {quality_level} · {name_ko}: "
            f"두께 {_fmt_mm(thickness_mm)}mm × 계수 {coef} = {_fmt_mm(by_thickness)}mm, "
            f"cap {_fmt_mm(cap)}mm → 허용 한계 {_fmt_mm(limit)}mm; "
            f"측정 {_fmt_mm(size_r)}mm {cmp_sym} 한계 → {verdict_ko}"
        )

        return RuleVerdict(
            defect_id=defect_id,
            defect_type=defect_type,
            size_mm=size_r,
            quality_level=quality_level,
            thickness_mm=thickness_mm,
            limit_mm=limit,
            passed=passed,
            clause=clause,
            detail=detail,
        )

    def evaluate_all(
        self,
        items: list[tuple[str, str, float]],
        thickness_mm: float,
        quality_level: str,
    ) -> list[RuleVerdict]:
        """items = [(defect_id, defect_type, size_mm), ...] 일괄 판정 (입력 순서 보존)."""
        return [
            self.evaluate(defect_id, defect_type, size_mm, thickness_mm, quality_level)
            for defect_id, defect_type, size_mm in items
        ]


def overall_pass(verdicts: list[RuleVerdict]) -> bool:
    """전체 합부: 모든 verdict가 passed일 때만 True. 빈 목록은 True(결함 없음)."""
    return all(v.passed for v in verdicts)
