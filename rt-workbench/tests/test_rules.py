"""rules.py — 결정론적 룰 엔진 테스트.

데모 기준표(demo_iso5817_like.json)의 수치를 그대로 사용한다:
  porosity B: coef 0.2, cap 3.0 / crack: 전 등급 불허 /
  lack_of_fusion: B·C 불허, D는 coef 0.25, cap 25.0
"""

from __future__ import annotations

import pytest

from rtworkbench.models import RuleVerdict
from rtworkbench.rules import RuleEngine, overall_pass


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    return RuleEngine()  # 기본 경로 = config.CRITERIA_PATH


# ──────────────────────────── meta ────────────────────────────


def test_meta_블록_노출(engine):
    assert "name" in engine.meta
    assert "disclaimer" in engine.meta
    assert engine.meta["quality_levels"] == ["B", "C", "D"]


# ─────────────────────── 불허 유형: 균열 ───────────────────────


@pytest.mark.parametrize("level", ["B", "C", "D"])
@pytest.mark.parametrize("size", [0.0, 0.01, 1.0, 100.0])
def test_균열은_모든_등급에서_어떤_크기든_불합격(engine, level, size):
    v = engine.evaluate("df-1", "crack", size, thickness_mm=12.0, quality_level=level)
    assert v.passed is False
    assert v.limit_mm is None
    assert v.clause == "DEMO-100"
    assert "허용되지 않음" in v.detail
    assert f"품질등급 {level}" in v.detail


# ────────────────── 허용 유형: 기공(porosity) ──────────────────


def test_기공_B_t12_한계는_두께지배_2p4(engine):
    # min(0.2×12, 3.0) = 2.4 → 두께 항이 지배
    v = engine.evaluate("df-1", "porosity", 1.0, thickness_mm=12.0, quality_level="B")
    assert v.limit_mm == 2.4
    assert v.passed is True
    assert v.clause == "DEMO-2011"


def test_기공_B_t12_경계값_2p4는_합격(engine):
    # 경계 동치: 측정 == 한계 → 합격 (ISO 계열 관례: 한계 '이하' 허용)
    v = engine.evaluate("df-1", "porosity", 2.4, thickness_mm=12.0, quality_level="B")
    assert v.passed is True
    assert "합격" in v.detail and "불합격" not in v.detail


def test_기공_B_t12_2p41은_불합격(engine):
    v = engine.evaluate("df-1", "porosity", 2.41, thickness_mm=12.0, quality_level="B")
    assert v.passed is False
    assert v.limit_mm == 2.4
    assert "불합격" in v.detail


def test_기공_B_t30_cap이_지배해_한계_3p0(engine):
    # min(0.2×30=6.0, 3.0) = 3.0 → cap이 지배
    v = engine.evaluate("df-1", "porosity", 3.0, thickness_mm=30.0, quality_level="B")
    assert v.limit_mm == 3.0
    assert v.passed is True
    v2 = engine.evaluate("df-1", "porosity", 3.01, thickness_mm=30.0, quality_level="B")
    assert v2.passed is False


def test_기공_detail_판독_근거_흐름(engine):
    v = engine.evaluate("df-1", "porosity", 3.1, thickness_mm=12.0, quality_level="B")
    # "두께 × 계수 = X, cap Y → 허용 한계 Z; 측정 W > 한계 → 불합격" 흐름 확인
    for token in ("두께 12.0mm", "계수 0.2", "2.4mm", "cap 3.0mm",
                  "허용 한계 2.4mm", "측정 3.1mm", "불합격"):
        assert token in v.detail, f"detail에 '{token}' 누락: {v.detail}"


# ──────────── 등급별 허용/불허 혼재: 융합불량(LoF) ────────────


@pytest.mark.parametrize("level", ["B", "C"])
def test_융합불량_B_C는_불허(engine, level):
    v = engine.evaluate("df-1", "lack_of_fusion", 0.5, thickness_mm=12.0, quality_level=level)
    assert v.passed is False
    assert v.limit_mm is None
    assert "허용되지 않음" in v.detail


def test_융합불량_D는_한계_계산(engine):
    # min(0.25×12=3.0, 25.0) = 3.0
    v = engine.evaluate("df-1", "lack_of_fusion", 3.0, thickness_mm=12.0, quality_level="D")
    assert v.limit_mm == 3.0
    assert v.passed is True
    v2 = engine.evaluate("df-1", "lack_of_fusion", 3.1, thickness_mm=12.0, quality_level="D")
    assert v2.passed is False


# ─────────────── 미등재 유형: fail-safe 원칙 ───────────────


@pytest.mark.parametrize("dtype", ["unknown", "완전히_없는_유형"])
def test_미등재_유형은_fail_safe_불합격(engine, dtype):
    v = engine.evaluate("df-1", dtype, 0.1, thickness_mm=12.0, quality_level="B")
    assert v.passed is False
    assert v.limit_mm is None
    assert v.clause == "DEMO-UNLISTED"
    assert "수동 판정" in v.detail  # 자동 합격 금지 + 판독원 수동 판정 요구


# ──────────────────────── 입력 검증 ────────────────────────


def test_품질등급_A는_ValueError(engine):
    with pytest.raises(ValueError):
        engine.evaluate("df-1", "porosity", 1.0, thickness_mm=12.0, quality_level="A")


def test_음수_크기는_ValueError(engine):
    with pytest.raises(ValueError):
        engine.evaluate("df-1", "porosity", -0.1, thickness_mm=12.0, quality_level="B")


@pytest.mark.parametrize("t", [0.0, -5.0])
def test_두께_0이하는_ValueError(engine, t):
    with pytest.raises(ValueError):
        engine.evaluate("df-1", "porosity", 1.0, thickness_mm=t, quality_level="B")


def test_미등재_유형이라도_입력이_틀리면_ValueError(engine):
    # fail-safe 경로에서도 입력 검증이 먼저다
    with pytest.raises(ValueError):
        engine.evaluate("df-1", "unknown", 1.0, thickness_mm=12.0, quality_level="X")


def test_기준표_JSON_형식_오류는_ValueError(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ 이것은 JSON이 아님", encoding="utf-8")
    with pytest.raises(ValueError):
        RuleEngine(bad)
    missing = tmp_path / "missing.json"
    missing.write_text('{"meta": {}, "rules": {}}', encoding="utf-8")  # default_rule 누락
    with pytest.raises(ValueError):
        RuleEngine(missing)


def test_기준표_규칙_levels에_선언된_등급_누락은_로드시_ValueError(tmp_path):
    """선언된 quality_levels(B/C/D) 중 일부가 규칙 levels에 없으면 로드 거부.

    회귀 방지: 과거엔 로드가 통과한 뒤 evaluate(..., 'D')에서 원시 KeyError.
    """
    bad = tmp_path / "incomplete_levels.json"
    bad.write_text(
        """{
          "meta": {"quality_levels": ["B", "C", "D"]},
          "rules": {"porosity": {"clause": "X-1",
                    "levels": {"B": {"permitted": false}}}},
          "default_rule": {"clause": "X-0", "detail": "수동 판정 필요"}
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        RuleEngine(bad)


def test_기준표_rule이_dict가_아니면_ValueError(tmp_path):
    bad = tmp_path / "rule_not_dict.json"
    bad.write_text(
        """{
          "meta": {"quality_levels": ["B"]},
          "rules": {"porosity": "이것은 dict가 아님"},
          "default_rule": {"clause": "X-0", "detail": "수동 판정 필요"}
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        RuleEngine(bad)


def test_기준표_default_rule에_clause_detail_없으면_ValueError(tmp_path):
    """회귀 방지: 과거엔 로드 통과 후 미등재 유형 fail-safe 경로에서 KeyError('clause')."""
    bad = tmp_path / "default_missing_keys.json"
    bad.write_text(
        '{"meta": {}, "rules": {}, "default_rule": {"fail_safe": true}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        RuleEngine(bad)
    bad2 = tmp_path / "default_not_dict.json"
    bad2.write_text('{"meta": {}, "rules": {}, "default_rule": "x"}', encoding="utf-8")
    with pytest.raises(ValueError):
        RuleEngine(bad2)


def test_기준표_meta가_dict가_아니면_ValueError(tmp_path):
    bad = tmp_path / "meta_not_dict.json"
    bad.write_text(
        '{"meta": [], "rules": {}, "default_rule": {"clause": "X-0", "detail": "d"}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        RuleEngine(bad)


# ─────────────── evaluate_all / overall_pass ───────────────


def test_evaluate_all_순서_보존(engine):
    items = [
        ("df-3", "porosity", 1.0),
        ("df-1", "crack", 0.5),
        ("df-2", "undercut", 0.1),
    ]
    verdicts = engine.evaluate_all(items, thickness_mm=12.0, quality_level="B")
    assert [v.defect_id for v in verdicts] == ["df-3", "df-1", "df-2"]
    assert [v.defect_type for v in verdicts] == ["porosity", "crack", "undercut"]


def test_overall_pass_전부_합격이면_True(engine):
    verdicts = engine.evaluate_all(
        [("df-1", "porosity", 1.0), ("df-2", "slag_inclusion", 0.5)],
        thickness_mm=12.0,
        quality_level="B",
    )
    assert overall_pass(verdicts) is True


def test_overall_pass_하나라도_불합격이면_False(engine):
    verdicts = engine.evaluate_all(
        [("df-1", "porosity", 1.0), ("df-2", "crack", 0.1)],
        thickness_mm=12.0,
        quality_level="B",
    )
    assert overall_pass(verdicts) is False


def test_overall_pass_빈_목록은_True_결함없음():
    assert overall_pass([]) is True


# ───────────── 부동소수 경계: 반올림 정책의 무모순성 ─────────────


def test_부동소수_경계_0p2x12는_합격(engine):
    # 0.2 * 12 == 2.4000000000000004 (이진 부동소수). 정책: 한계는 round(·, 2)로
    # 먼저 확정하고 측정값도 round(size, 2)로 정렬해 비교하므로,
    # '한계와 같은 값을 측정했는데 불합격'이라는 모순이 생기지 않아야 한다.
    raw = 0.2 * 12.0
    assert raw != 2.4  # 전제: 실제로 부동소수 오차가 존재
    v = engine.evaluate("df-1", "porosity", raw, thickness_mm=12.0, quality_level="B")
    assert v.limit_mm == 2.4
    assert v.passed is True


def test_저장값_자기모순_없음(engine):
    # verdict에 저장된 (size_mm, limit_mm, passed)만 보고도 판정이 재현돼야 한다:
    # passed == (size_mm <= limit_mm)
    for size in (0.0, 2.39, 2.4, 0.2 * 12.0, 2.401, 2.41, 3.0):
        v = engine.evaluate("df-1", "porosity", size, thickness_mm=12.0, quality_level="B")
        assert v.passed == (v.size_mm <= v.limit_mm), (size, v.size_mm, v.limit_mm, v.passed)


def test_verdict_필드_계약(engine):
    v = engine.evaluate("df-9", "porosity", 1.23, thickness_mm=12.0, quality_level="C")
    assert isinstance(v, RuleVerdict)
    assert v.defect_id == "df-9"
    assert v.defect_type == "porosity"
    assert v.size_mm == 1.23
    assert v.quality_level == "C"
    assert v.thickness_mm == 12.0
    # porosity C: min(0.25×12=3.0, 4.0) = 3.0
    assert v.limit_mm == 3.0
