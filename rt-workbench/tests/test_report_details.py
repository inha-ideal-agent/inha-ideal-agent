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
