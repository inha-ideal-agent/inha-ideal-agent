"""rules.py — 결정론적 룰 엔진 테스트.

데모 기준표(demo_iso5817_like.json)의 수치를 그대로 사용한다:
  porosity B: coef 0.2, cap 3.0 / crack: 전 등급 불허 /
  lack_of_fusion: B·C 불허, D는 coef 0.25, cap 25.0
  그룹 판정 — porosity area_ratio: B 1.0% / C 1.5% / D 2.0% ;
             slag_inclusion cumulative_length B: coef 0.5, cap 12.5 ;
             lack_of_fusion cumulative_length: B·C 불허, D coef 0.25, cap 25.0
"""

from __future__ import annotations

import math

import pytest

from rtworkbench.models import RuleVerdict
from rtworkbench.rules import RuleEngine, is_group_verdict, overall_pass


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
    singles = [v for v in verdicts if not is_group_verdict(v)]
    assert [v.defect_id for v in singles] == ["df-3", "df-1", "df-2"]
    assert [v.defect_type for v in singles] == ["porosity", "crack", "undercut"]
    # 그룹 verdict는 단일 verdict 뒤에 덧붙는다 (세 유형 중 group 블록은 기공만 보유)
    assert [v.defect_id for v in verdicts[len(singles):]] == ["GROUP:porosity"]


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


# ═══════════════ 그룹 판정: 유형별 누적 길이 · 투영 면적률 ═══════════════


def _group(verdicts: list[RuleVerdict], dtype: str) -> RuleVerdict:
    gs = [v for v in verdicts if v.defect_id == f"GROUP:{dtype}"]
    assert len(gs) == 1, [v.defect_id for v in verdicts]
    return gs[0]


# ───────────── 투영 면적률 (area_ratio): 기공 ─────────────


def test_면적률_스펙_예시_기공3건_12p57mm2는_0p63pct_합격(engine):
    # 3건 × π(d/2)² = 12.57mm² 가 되는 d ≈ 2.31 → 12.57 ÷ (100 × 20) × 100 = 0.63% ≤ 1.0%
    d = math.sqrt(12.57 * 4 / (3 * math.pi))
    verdicts = engine.evaluate_all(
        [("a", "porosity", d), ("b", "porosity", d), ("c", "porosity", d)],
        thickness_mm=12.0, quality_level="B", eval_length_mm=100.0, weld_width_mm=20.0,
    )
    g = _group(verdicts, "porosity")
    assert g.unit == "%"
    assert g.size_mm == 0.63 and g.limit_mm == 1.0 and g.passed is True
    assert g.clause == "DEMO-2012-AREA" and g.quality_level == "B" and g.thickness_mm == 12.0
    assert (
        "기공 3건 원 근사 면적 합 12.57mm² ÷ (평가길이 100mm × 용접부 폭 20mm) "
        "= 0.63% ≤ 한계 1.0% → 합격"
    ) in g.detail


def test_면적률_개별은_전부_합격이어도_합계_초과면_그룹_불합격(engine):
    # 5 × d=2.4(단일 한계와 동치) → 5 × π × 1.2² = 22.62mm² ÷ 2000mm² = 1.13% > 1.0%
    verdicts = engine.evaluate_all(
        [(f"p{i}", "porosity", 2.4) for i in range(5)], thickness_mm=12.0, quality_level="B"
    )
    assert all(v.passed for v in verdicts if not is_group_verdict(v))
    g = _group(verdicts, "porosity")
    assert g.size_mm == 1.13 and g.limit_mm == 1.0 and g.passed is False
    assert "22.62mm²" in g.detail and "> 한계 1.0%" in g.detail and "불합격" in g.detail
    assert overall_pass(verdicts) is False  # 그룹 verdict가 종합 판정에 반영된다


@pytest.mark.parametrize(
    "weld_w, ratio, passed",
    [(3.1416, 1.0, True), (3.14, 1.0, True), (3.12, 1.01, False)],
)
def test_면적률_경계_반올림_정책_경계_동치는_합격(engine, weld_w, ratio, passed):
    # d=2.0 → 면적 π mm² ; 분모 100 × W → 면적률 = 314.159… ÷ (100·W) %
    g = engine.evaluate_group(
        "porosity", [2.0], 12.0, "B", eval_length_mm=100.0, weld_width_mm=weld_w
    )
    assert g.size_mm == ratio and g.limit_mm == 1.0 and g.passed is passed
    assert g.passed == (g.size_mm <= g.limit_mm)  # 저장값 자기모순 없음


@pytest.mark.parametrize("level, limit", [("B", 1.0), ("C", 1.5), ("D", 2.0)])
def test_면적률_등급별_한계(engine, level, limit):
    g = engine.evaluate_group("porosity", [1.0], 12.0, level)
    assert g.limit_mm == limit and g.unit == "%" and g.quality_level == level


def test_평가구간이_바뀌면_면적률이_바뀐다(engine):
    # d=2.0: 100×20 → 0.157% ≈ 0.16 ; 50×10(1/4 면적) → 0.628% ≈ 0.63
    assert engine.evaluate_group("porosity", [2.0], 12.0, "B").size_mm == 0.16
    g = engine.evaluate_group(
        "porosity", [2.0], 12.0, "B", eval_length_mm=50.0, weld_width_mm=10.0
    )
    assert g.size_mm == 0.63 and "평가길이 50mm × 용접부 폭 10mm" in g.detail


# ───────────── 누적 길이 (cumulative_length): 슬래그 · 융합불량 ─────────────


def test_누적길이_슬래그_B_t12_한계_6p0_합격(engine):
    # min(0.5×12=6.0, 12.5) = 6.0 ; 2.0+2.0+1.5 = 5.5 ≤ 6.0
    verdicts = engine.evaluate_all(
        [("s1", "slag_inclusion", 2.0), ("s2", "slag_inclusion", 2.0), ("s3", "slag_inclusion", 1.5)],
        thickness_mm=12.0, quality_level="B",
    )
    g = _group(verdicts, "slag_inclusion")
    assert g.unit == "mm"
    assert g.size_mm == 5.5 and g.limit_mm == 6.0 and g.passed is True
    assert g.clause == "DEMO-302-CUM"
    for token in ("슬래그 개재물 3건", "(2.0 + 2.0 + 1.5) = 5.5mm", "계수 0.5",
                  "cap 12.5mm", "허용 한계 6.0mm", "누적 5.5mm ≤ 한계", "합격"):
        assert token in g.detail, f"detail에 '{token}' 누락: {g.detail}"


def test_누적길이_경계_동치는_합격_초과는_불합격(engine):
    g = engine.evaluate_group("slag_inclusion", [2.0, 2.0, 2.0], 12.0, "B")
    assert g.size_mm == 6.0 and g.limit_mm == 6.0 and g.passed is True
    g2 = engine.evaluate_group("slag_inclusion", [2.0, 2.0, 2.0, 1.0], 12.0, "B")
    assert g2.size_mm == 7.0 and g2.passed is False and "누적 7.0mm > 한계" in g2.detail
    assert g2.passed == (g2.size_mm <= g2.limit_mm)


def test_누적길이_개별은_전부_합격이어도_합계_초과면_종합_불합격(engine):
    # 단일 한계 min(0.2×12=2.4, 2.0) = 2.0 → 각각 합격 ; 누적 8.0 > 6.0 → 그룹 불합격
    verdicts = engine.evaluate_all(
        [(f"s{i}", "slag_inclusion", 2.0) for i in range(4)], thickness_mm=12.0, quality_level="B"
    )
    assert all(v.passed for v in verdicts if not is_group_verdict(v))
    assert _group(verdicts, "slag_inclusion").passed is False
    assert overall_pass(verdicts) is False


def test_누적길이_cap이_지배(engine):
    # min(0.5×30=15.0, 12.5) = 12.5
    assert engine.evaluate_group("slag_inclusion", [1.0], 30.0, "B").limit_mm == 12.5


@pytest.mark.parametrize("level", ["B", "C"])
def test_누적길이_융합불량_B_C는_불허_permitted_false(engine, level):
    g = engine.evaluate_group("lack_of_fusion", [1.0, 2.0], 12.0, level)
    assert g.passed is False and g.limit_mm is None
    assert g.size_mm == 3.0 and g.unit == "mm"  # 집계값은 기록한다
    assert g.clause == "DEMO-401-CUM"
    assert "허용되지 않음" in g.detail and f"품질등급 {level}" in g.detail


def test_누적길이_융합불량_D는_허용(engine):
    # min(0.25×12=3.0, 25.0) = 3.0 ; 1.0+2.0 = 3.0 → 경계 동치 합격
    g = engine.evaluate_group("lack_of_fusion", [1.0, 2.0], 12.0, "D")
    assert g.limit_mm == 3.0 and g.passed is True


# ───────────── 공통 계약 ─────────────


def test_group_블록_없는_유형은_그룹_verdict가_없다(engine):
    verdicts = engine.evaluate_all(
        [("c1", "crack", 1.0), ("u1", "undercut", 0.1), ("cp", "cluster_porosity", 1.0)],
        thickness_mm=12.0, quality_level="B",
    )
    assert len(verdicts) == 3 and not any(is_group_verdict(v) for v in verdicts)
    assert engine.evaluate_group("crack", [1.0], 12.0, "B") is None
    assert engine.evaluate_group("porosity", [], 12.0, "B") is None  # 집계 대상 없음
    assert engine.evaluate_group("없는_유형", [1.0], 12.0, "B") is None


def test_그룹_verdict는_유형별_1건_최초등장_순서(engine):
    verdicts = engine.evaluate_all(
        [("s1", "slag_inclusion", 1.0), ("p1", "porosity", 1.0),
         ("s2", "slag_inclusion", 1.0), ("c1", "crack", 1.0)],
        thickness_mm=12.0, quality_level="B",
    )
    assert [v.defect_id for v in verdicts] == [
        "s1", "p1", "s2", "c1", "GROUP:slag_inclusion", "GROUP:porosity"
    ]


def test_그룹_verdict_ID_및_표시명(engine):
    g = engine.evaluate_group("porosity", [1.0], 12.0, "B")
    assert g.defect_id == "GROUP:porosity" and g.defect_type == "porosity"
    assert is_group_verdict(g) and g.is_group and g.display_id == "합계(기공)"
    s = engine.evaluate("df-1", "porosity", 1.0, thickness_mm=12.0, quality_level="B")
    assert not is_group_verdict(s) and not s.is_group and s.display_id == "df-1"


def test_unit_필드_기본값_mm_면적률은_pct_직렬화_왕복(engine):
    s = engine.evaluate("df-1", "porosity", 1.0, thickness_mm=12.0, quality_level="B")
    assert s.unit == "mm" and s.to_dict()["unit"] == "mm"
    g = engine.evaluate_group("porosity", [1.0], 12.0, "B")
    assert g.unit == "%" and RuleVerdict.from_dict(g.to_dict()) == g
    legacy = {k: v for k, v in s.to_dict().items() if k != "unit"}  # unit 없는 구버전 기록
    assert RuleVerdict.from_dict(legacy).unit == "mm"


def test_evaluate_all_3인자_호출_호환_기본_평가구간_100x20(engine):
    items = [("a", "porosity", 2.0), ("b", "slag_inclusion", 1.0)]
    old = engine.evaluate_all(items, 12.0, "B")
    new = engine.evaluate_all(items, 12.0, "B", eval_length_mm=100.0, weld_width_mm=20.0)
    assert old == new and len(old) == 4
    assert "평가길이 100mm × 용접부 폭 20mm" in _group(old, "porosity").detail


@pytest.mark.parametrize(
    "kw", [{"eval_length_mm": 0.0}, {"eval_length_mm": -1.0}, {"weld_width_mm": 0.0}]
)
def test_평가길이_용접부폭_0이하는_ValueError(engine, kw):
    with pytest.raises(ValueError):
        engine.evaluate_all([("a", "porosity", 1.0)], 12.0, "B", **kw)
    with pytest.raises(ValueError):
        engine.evaluate_group("porosity", [1.0], 12.0, "B", **kw)


def test_그룹_판정도_입력검증(engine):
    with pytest.raises(ValueError):
        engine.evaluate_group("porosity", [1.0, -0.1], 12.0, "B")
    with pytest.raises(ValueError):
        engine.evaluate_group("porosity", [1.0], 12.0, "A")
    with pytest.raises(ValueError):
        engine.evaluate_group("porosity", [1.0], 0.0, "B")


# ───────────── 기준표 group 블록 검증 ─────────────

_GROUP_BASE = (
    '{"meta": {"quality_levels": ["B"]}, '
    '"rules": {"porosity": {"clause": "X-1", '
    '"levels": {"B": {"permitted": true, "coef_thickness": 0.2, "cap_mm": 3.0}}, '
    '"group": %s}}, '
    '"default_rule": {"clause": "X-0", "detail": "수동 판정 필요"}}'
)


@pytest.mark.parametrize(
    "group_json",
    [
        '"문자열"',  # dict 아님
        '{"mode": "volume_ratio", "clause": "G", "levels": {"B": {"limit_pct": 1}}}',  # 미지 mode
        '{"mode": "area_ratio", "levels": {"B": {"limit_pct": 1}}}',  # clause 누락
        '{"mode": "area_ratio", "clause": "G"}',  # levels 누락
        '{"mode": "area_ratio", "clause": "G", "levels": {}}',  # 선언 등급 B 누락
        '{"mode": "area_ratio", "clause": "G", "levels": {"B": {}}}',  # limit_pct 누락
        '{"mode": "area_ratio", "clause": "G", "levels": {"B": {"limit_pct": "1"}}}',  # 숫자 아님
        '{"mode": "cumulative_length", "clause": "G", '
        '"levels": {"B": {"permitted": true, "coef_thickness": 0.5}}}',  # cap_mm 누락
        '{"mode": "cumulative_length", "clause": "G", "levels": {"B": "x"}}',  # 등급 entry dict 아님
    ],
)
def test_기준표_group_블록_형식_오류는_로드시_ValueError(tmp_path, group_json):
    bad = tmp_path / "bad_group.json"
    bad.write_text(_GROUP_BASE % group_json, encoding="utf-8")
    with pytest.raises(ValueError):
        RuleEngine(bad)


def test_기준표_group_불허_등급은_한계_수치_없이_로드된다(tmp_path):
    ok = tmp_path / "ok_group.json"
    ok.write_text(
        _GROUP_BASE % '{"mode": "cumulative_length", "clause": "G", '
        '"levels": {"B": {"permitted": false}}}',
        encoding="utf-8",
    )
    g = RuleEngine(ok).evaluate_group("porosity", [1.0], 12.0, "B")
    assert g.passed is False and g.limit_mm is None and g.clause == "G"
