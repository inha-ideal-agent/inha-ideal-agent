"""선급 보고 항목(IACS UR W33 §8.2·8.5) — ReportDetails 모델 / 커버리지 / 구버전 기록 호환 테스트."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import fields

import pytest

from rtworkbench.db import Archive
from rtworkbench.models import (
    REPORT_GROUPS,
    REPORT_ITEM_TOTAL,
    REPORT_ITEMS,
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    ReportDetails,
)

# W33 §8.2 일반 항목 중 InspectionContext에 없는 것(8) + 보수 횟수(1) + §8.5 RT 전용(16)
EXPECTED_GENERAL_KEYS = (
    "hull_number", "weld_length_mm", "personnel_qualification", "steel_grade",
    "welding_process", "testing_standard", "equipment", "limitations_viewing", "repairs_count",
)
EXPECTED_RT_KEYS = (
    "source_type_size", "xray_kv", "film_type", "exposures_count", "screens",
    "exposure_technique", "exposure_time_s", "sfd_mm", "source_to_weld_mm", "weld_to_film_mm",
    "beam_angle_deg", "iqi_sensitivity", "iqi_type_position", "density",
    "geometric_unsharpness", "rt_acceptance_class",
)


def _record(ctx: InspectionContext, rid: str = "rec-rpt") -> InspectionRecord:
    return InspectionRecord(
        record_id=rid, context=ctx,
        candidates=[DefectCandidate(id="d1", defect_type="porosity", bbox=(1.0, 1.0, 5.0, 5.0),
                                    confidence=0.9, status="accepted")],
        measurements=[], verdicts=[], overall_passed=True,
        report_text="본문", report_source="template", image_name="film.png", image_size=(100, 50),
    )


# ---------------------------------------------------------------- 항목 목록(W33 대응)


def test_report_items_cover_w33_generic_and_rt_specific_items():
    """항목 목록 = §8.2 일반 9개 + §8.5 RT 전용 16개 = 25, 데이터클래스 필드와 1:1."""
    assert REPORT_ITEM_TOTAL == 25
    assert tuple(i.key for i in REPORT_ITEMS if i.group == "general") == EXPECTED_GENERAL_KEYS
    assert tuple(i.key for i in REPORT_ITEMS if i.group == "rt") == EXPECTED_RT_KEYS
    assert [i.key for i in REPORT_ITEMS] == [f.name for f in fields(ReportDetails)]
    assert set(i.group for i in REPORT_ITEMS) == set(REPORT_GROUPS)
    for item in REPORT_ITEMS:
        assert item.label_ko and item.label_en  # 한국어 라벨 + W33 원문(help)


def test_report_details_defaults_are_empty():
    r = ReportDetails()
    assert r.is_empty()
    assert r.coverage() == (0, 25)
    assert r.filled_items() == []
    assert r.repairs_count == 0


# ---------------------------------------------------------------- 커버리지/표시


def test_coverage_counts_filled_items_and_units():
    r = ReportDetails(hull_number="H-2031", sfd_mm="700", repairs_count=3, xray_kv=" 200 ",
                      beam_angle_deg="0", exposures_count="3매")
    assert r.coverage() == (6, 25)
    shown = {item.key: value for item, value in r.filled_items()}
    assert shown["hull_number"] == "H-2031"
    assert shown["sfd_mm"] == "700 mm"  # 단위 기호는 띄어 쓴다
    assert shown["xray_kv"] == "200 kV"  # 공백 트리밍
    assert shown["repairs_count"] == "3회"  # 조수사는 붙여 쓴다
    assert shown["beam_angle_deg"] == "0°"
    assert shown["exposures_count"] == "3매"  # 이미 단위가 붙어 있으면 중복하지 않는다
    # 순서는 REPORT_ITEMS 순서(일반 → RT)
    assert [item.key for item, _ in r.filled_items()] == [
        "hull_number", "repairs_count", "xray_kv", "exposures_count", "sfd_mm", "beam_angle_deg",
    ]


def test_whitespace_only_and_zero_repairs_are_not_filled():
    r = ReportDetails(hull_number="   ", repairs_count=0)
    assert r.coverage() == (0, 25) and r.is_empty()


# ---------------------------------------------------------------- 직렬화 왕복 / 구버전 호환


def test_context_roundtrip_with_report():
    ctx = InspectionContext(
        film_id="F-1", block="3번 블록",
        report=ReportDetails(hull_number="H-1", steel_grade="AH36", repairs_count=2,
                             iqi_sensitivity="W13", density="2.3~2.8"),
    )
    d = ctx.to_dict()
    assert isinstance(d["report"], dict) and d["report"]["hull_number"] == "H-1"
    back = InspectionContext.from_dict(json.loads(json.dumps(d, ensure_ascii=False)))
    assert back == ctx
    assert back.report.coverage() == (5, 25)


def test_context_from_dict_without_report_key_gets_defaults():
    """구버전 기록(report 키 없음)도 로드되고 보고 항목은 0/25."""
    legacy = {"film_id": "F-old", "block": "5번 블록", "weld_id": "W-9", "thickness_mm": 10.0,
              "quality_level": "C", "inspector": "홍길동"}
    ctx = InspectionContext.from_dict(legacy)
    assert ctx.film_id == "F-old" and ctx.thickness_mm == 10.0
    assert ctx.report == ReportDetails()
    assert ctx.report.coverage() == (0, 25)


def test_report_from_dict_tolerates_none_unknown_keys_and_string_count():
    assert ReportDetails.from_dict(None) == ReportDetails()
    assert ReportDetails.from_dict({}) == ReportDetails()
    r = ReportDetails.from_dict({"hull_number": "H-1", "future_item": "x",
                                 "repairs_count": "4", "density": None})
    assert r.hull_number == "H-1" and r.repairs_count == 4 and r.density == ""
    assert ReportDetails.from_dict({"repairs_count": "abc"}).repairs_count == 0


def test_record_json_roundtrip_and_legacy_json_without_report():
    ctx = InspectionContext(film_id="F-1", report=ReportDetails(hull_number="H-7", sfd_mm="700"))
    rec = _record(ctx)
    back = InspectionRecord.from_json(rec.to_json())
    assert back.context.report == ctx.report
    assert back.to_json() == rec.to_json()

    legacy = json.loads(rec.to_json())
    del legacy["context"]["report"]
    old = InspectionRecord.from_json(json.dumps(legacy, ensure_ascii=False))
    assert old.context.report == ReportDetails()
    assert old.context.film_id == "F-1"


# ---------------------------------------------------------------- DB (마이그레이션 안전)


def test_archive_roundtrip_preserves_report(tmp_path):
    db = Archive(tmp_path / "t.db")
    ctx = InspectionContext(film_id="F-1", report=ReportDetails(
        hull_number="H-2031", personnel_qualification="RT Level 2", rt_acceptance_class="Level 1"))
    db.save(_record(ctx))
    got = db.get("rec-rpt")
    assert got is not None and got.context.report == ctx.report
    assert got.context.report.coverage() == (3, 25)


def test_archive_loads_legacy_row_whose_payload_has_no_report(tmp_path):
    """payload_json 에 context.report 가 없는(추가 전) 행도 get/export/import 가 그대로 동작한다."""
    db = Archive(tmp_path / "legacy.db")
    legacy = json.loads(_record(InspectionContext(film_id="F-legacy"), "rec-legacy").to_json())
    del legacy["context"]["report"]
    payload = json.dumps(legacy, ensure_ascii=False)
    assert '"report"' not in payload
    with sqlite3.connect(tmp_path / "legacy.db") as conn:
        conn.execute(
            "INSERT INTO inspections (record_id, created_at, overall_passed, payload_json) "
            "VALUES (?, ?, ?, ?)",
            ("rec-legacy", legacy["created_at"], 1, payload),
        )
    got = db.get("rec-legacy")
    assert got is not None and got.context.film_id == "F-legacy"
    assert got.context.report == ReportDetails() and got.context.report.coverage() == (0, 25)

    # 백업(JSON Lines) → 복원도 구버전 줄을 그대로 받아들인다
    dst = Archive(tmp_path / "dst.db")
    assert dst.import_json(db.export_all_json()) == 1
    assert dst.get("rec-legacy").context.report.is_empty()


# ---------------------------------------------------------------- W33 §8 원문 불릿 충족률 (n/25)

from rtworkbench.models import (  # noqa: E402
    W33_GROUP_COUNTS,
    W33_ITEM_TOTAL,
    W33_ITEMS,
    RuleVerdict,
    w33_coverage,
    w33_item_filled,
)


def _full_report() -> ReportDetails:
    return ReportDetails(
        hull_number="H-2031", weld_length_mm="300", personnel_qualification="ISO 9712 RT Level 2",
        steel_grade="AH36", welding_process="FCAW (136)", testing_standard="ISO 17636-1 Class B",
        equipment="Ir-192 감마선 장비, 단벽 단상", limitations_viewing="제한 없음, 18℃", repairs_count=3,
        source_type_size="Ir-192, 2.0×2.0 mm", xray_kv="", film_type="AGFA D4, 1매", exposures_count="3",
        screens="Pb 0.1 mm", exposure_technique="SWSI", exposure_time_s="90", sfd_mm="700",
        source_to_weld_mm="688", weld_to_film_mm="12", beam_angle_deg="0", iqi_sensitivity="W13 (1.6%)",
        iqi_type_position="ISO 19232-1 선형, 선원측", density="2.3~2.8", geometric_unsharpness="0.03",
        rt_acceptance_class="ISO 10675-1 Level 1",
    )


def test_w33_items_are_the_25_bullets_13_general_12_rt():
    """분모 = W33 §8.2 일반 13 + §8.5 RT 12 = 25 원문 불릿(입력 필드 25개와 다른 단위)."""
    assert W33_ITEM_TOTAL == 25 and W33_GROUP_COUNTS == {"general": 13, "rt": 12}
    assert [i.group for i in W33_ITEMS] == ["general"] * 13 + ["rt"] * 12  # 원문 순서
    assert len({i.key for i in W33_ITEMS}) == 25
    for item in W33_ITEMS:
        assert item.label_ko and item.label_en and item.rule_ko
    # 원문 불릿 대조(§8.5): 촬영 기법+노출 시간+SFD 가 한 불릿, 선원-용접부/용접부-필름 거리는 각각 별도
    rt_en = [i.label_en for i in W33_ITEMS if i.group == "rt"]
    assert rt_en[4].startswith("Exposure technique, time of exposure and source-to-film distance")
    assert rt_en[5] == "Distance from radiation source to weld"
    assert rt_en[6] == "Distance from source side of the weld to radiographic film"
    assert rt_en[0].startswith("Type and size of radiation source") and "X-ray voltage" in rt_en[0]


def test_w33_coverage_empty_context_counts_only_always_true_items():
    """빈 컨텍스트: 검사일(승인 시각 자동)·합격 기준(품질등급+기준표 항상 기록)만 충족 → 2/25."""
    n, total, missing = w33_coverage(InspectionContext())
    assert (n, total) == (2, 25) and len(missing) == 23
    assert "검사일" not in missing and "합격 기준" not in missing
    assert missing[0] == "선체 번호·용접부 위치·검사 길이" and missing[-1] == "RT 합격 등급"  # W33_ITEMS 순서
    # 판정 전엔 결과/합부 선언 불릿이 미충족
    assert "검사 결과(합격 기준 대비 지시 위치·크기)" in missing and "합부 선언·평가일·평가자" in missing


def test_w33_coverage_full_context_after_judgment_is_25_of_25():
    ctx = InspectionContext(block="3번 블록", weld_id="W-102", inspector="홍길동", report=_full_report())
    assert w33_coverage(ctx, judged=False) == (23, 25, ["검사 결과(합격 기준 대비 지시 위치·크기)",
                                                        "합부 선언·평가일·평가자"])
    assert w33_coverage(ctx, judged=True) == (25, 25, [])
    # 승인 기록이 있으면 판정 완료로 간주 (judged 생략)
    rec = _record(ctx)
    rec.verdicts = [RuleVerdict("d1", "porosity", 1.0, "B", 12.0, 2.4, True, "DEMO-2011", "ok")]
    assert w33_coverage(rec.context, rec) == (25, 25, [])
    assert w33_coverage(ctx, judged=True)[0] == 25 and ctx.report.coverage() == (24, 25)  # 필드 수와는 다른 값


@pytest.mark.parametrize(
    "key, ctx_kwargs, report_kwargs, expected",
    [
        # 선체 번호·위치·길이: 셋 모두 있어야 충족
        ("hull_location_length", {}, {"hull_number": "H"}, False),
        ("hull_location_length", {"weld_id": "W"}, {"hull_number": "H"}, False),
        ("hull_location_length", {"weld_id": "W"}, {"hull_number": "H", "weld_length_mm": "300"}, True),
        ("hull_location_length", {"block": "B3"}, {"hull_number": "H", "weld_length_mm": "300"}, True),
        ("hull_location_length", {}, {"hull_number": "H", "weld_length_mm": "300"}, False),
        # 검사자: 성명 + 자격 등급
        ("personnel", {"inspector": "홍길동"}, {}, False),
        ("personnel", {}, {"personnel_qualification": "L2"}, False),
        ("personnel", {"inspector": "홍길동"}, {"personnel_qualification": "L2"}, True),
        # 강재·이음·두께·용접 방법: 강재 등급 + 용접 방법 (이음·두께는 컨텍스트 필수값)
        ("material_joint_process", {}, {"steel_grade": "AH36"}, False),
        ("material_joint_process", {}, {"welding_process": "FCAW"}, False),
        ("material_joint_process", {}, {"steel_grade": "AH36", "welding_process": "FCAW"}, True),
        # 검사 규격: 기법명(technique 기본값)만으로는 충족되지 않는다
        ("testing_standards", {"technique": "RT (필름 스캔)"}, {}, False),
        ("testing_standards", {}, {"testing_standard": "ISO 17636-1"}, True),
        # 조건부 항목: 보수 횟수 0 은 미충족
        ("repairs", {}, {"repairs_count": 0}, False),
        ("repairs", {}, {"repairs_count": 3}, True),
        # RT: 선원 종류·크기만 있으면 충족(관전압은 X선 장비에만 해당), 관전압만으론 불충족
        ("source_voltage", {}, {"xray_kv": "200"}, False),
        ("source_voltage", {}, {"source_type_size": "Ir-192"}, True),
        # RT: 촬영 기법+노출 시간+SFD 세 필드가 한 불릿
        ("exposure_technique_time_sfd", {}, {"sfd_mm": "700"}, False),
        ("exposure_technique_time_sfd", {}, {"exposure_technique": "SWSI", "exposure_time_s": "90"}, False),
        ("exposure_technique_time_sfd", {}, {"exposure_technique": "SWSI", "exposure_time_s": "90",
                                             "sfd_mm": "700"}, True),
        # RT: IQI 감도 + 종류·위치
        ("iqi", {}, {"iqi_sensitivity": "W13"}, False),
        ("iqi", {}, {"iqi_sensitivity": "W13", "iqi_type_position": "선원측"}, True),
        # 거리 두 불릿은 서로 독립
        ("source_to_weld", {}, {"weld_to_film_mm": "12"}, False),
        ("source_to_weld", {}, {"source_to_weld_mm": "688"}, True),
        ("weld_to_film", {}, {"weld_to_film_mm": "12"}, True),
    ],
)
def test_w33_item_rules(key, ctx_kwargs, report_kwargs, expected):
    ctx = InspectionContext(**ctx_kwargs, report=ReportDetails(**report_kwargs))
    assert w33_item_filled(key, ctx) is expected


def test_w33_results_and_statement_depend_on_judgment_and_inspector():
    ctx = InspectionContext(inspector="")
    assert w33_item_filled("results", ctx) is False and w33_item_filled("results", ctx, judged=True) is True
    assert w33_item_filled("acceptance_statement", ctx, judged=True) is False  # 평가자 성명 없음
    ctx.inspector = "홍길동"
    assert w33_item_filled("acceptance_statement", ctx, judged=True) is True
    assert w33_item_filled("acceptance_statement", ctx, judged=False) is False
    with pytest.raises(KeyError):
        w33_item_filled("no_such_item", ctx)


def test_w33_coverage_on_legacy_record_without_report_does_not_fail(tmp_path):
    """보고 항목이 없는 구버전 기록도 컨텍스트만으로 계산된다(예외 없음)."""
    legacy = InspectionContext.from_dict({"film_id": "F-old", "block": "5번 블록", "inspector": "홍길동"})
    rec = _record(legacy, "rec-legacy")
    n, total, missing = w33_coverage(rec.context, rec)
    # 검사일·합격 기준·부재 식별(블록)·검사 결과·합부 선언(판정+판독원) = 5
    assert (n, total) == (5, 25) and "검사 부재 식별" not in missing and "검사 용접부 식별" in missing
