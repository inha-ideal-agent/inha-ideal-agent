"""보완 기능 회귀 테스트 — 판독 소요시간 기록 / JSON 백업·복원 / 평균 통계."""

from __future__ import annotations

import pytest

from rtworkbench.db import Archive
from rtworkbench.models import (
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    Measurement,
    RuleVerdict,
)


def _record(record_id: str, *, elapsed: float | None, image_name: str = "film_a.png") -> InspectionRecord:
    cand = DefectCandidate(
        id=f"df-{record_id}", defect_type="porosity", bbox=(10.0, 20.0, 30.0, 40.0),
        confidence=0.8, source="ai", status="accepted",
    )
    meas = Measurement(defect_id=cand.id, p1=(10.0, 30.0), p2=(30.0, 30.0),
                      length_px=20.0, length_mm=4.0)
    verdict = RuleVerdict(
        defect_id=cand.id, defect_type="porosity", size_mm=4.0, quality_level="B",
        thickness_mm=12.0, limit_mm=2.4, passed=False, clause="DEMO-2011", detail="테스트",
    )
    return InspectionRecord(
        record_id=record_id,
        context=InspectionContext(film_id="F-1", block="3번 블록", weld_id="W-1"),
        candidates=[cand], measurements=[meas], verdicts=[verdict],
        overall_passed=False, report_text="소견", report_source="template",
        image_name=image_name, image_size=(1400, 520), elapsed_seconds=elapsed,
    )


def test_elapsed_seconds_roundtrip(tmp_path):
    """elapsed_seconds가 저장→재조회에서 보존된다."""
    db = Archive(tmp_path / "t.db")
    db.save(_record("rec-el1", elapsed=123.4))
    got = db.get("rec-el1")
    assert got is not None and got.elapsed_seconds == 123.4


def test_elapsed_none_and_legacy_json(tmp_path):
    """elapsed 미기록(None) 허용 + 구버전 JSON(키 없음)도 파싱된다."""
    db = Archive(tmp_path / "t.db")
    db.save(_record("rec-el2", elapsed=None))
    assert db.get("rec-el2").elapsed_seconds is None

    legacy = _record("rec-legacy", elapsed=None).to_json()
    legacy = legacy.replace(', "elapsed_seconds": null', "")
    assert "elapsed_seconds" not in legacy
    rec = InspectionRecord.from_json(legacy)
    assert rec.elapsed_seconds is None


def test_stats_avg_elapsed(tmp_path):
    """평균 판독 소요시간 = 기록된 값들의 평균 (None 제외)."""
    db = Archive(tmp_path / "t.db")
    db.save(_record("r1", elapsed=60.0))
    db.save(_record("r2", elapsed=180.0, image_name="film_b.png"))
    db.save(_record("r3", elapsed=None, image_name="film_c.png"))
    assert db.stats()["avg_elapsed_seconds"] == pytest.approx(120.0)


def test_stats_avg_elapsed_empty(tmp_path):
    assert Archive(tmp_path / "t.db").stats()["avg_elapsed_seconds"] is None


def test_export_import_roundtrip(tmp_path):
    """JSON Lines 백업 → 빈 DB 복원 → 내용 동일."""
    src = Archive(tmp_path / "src.db")
    src.save(_record("r1", elapsed=60.0))
    src.save(_record("r2", elapsed=None, image_name="film_b.png"))
    backup = src.export_all_json()
    assert backup.count("\n") == 2  # 줄당 1건 + 마지막 개행

    dst = Archive(tmp_path / "dst.db")
    assert dst.import_json(backup) == 2
    for rid in ("r1", "r2"):
        assert dst.get(rid).to_json() == src.get(rid).to_json()
    # 재복원 시 record_id 교체 — 중복 없음
    assert dst.import_json(backup) == 2
    assert dst.stats()["total"] == 2


def test_import_broken_line(tmp_path):
    """파손된 줄은 줄 번호와 함께 ValueError."""
    db = Archive(tmp_path / "t.db")
    good = _record("r1", elapsed=None).to_json()
    with pytest.raises(ValueError, match="2번째"):
        db.import_json(good + "\n{broken json\n")
    assert db.get("r1") is not None  # 이전 줄은 저장됨 (문서화된 동작)


def test_export_empty(tmp_path):
    assert Archive(tmp_path / "t.db").export_all_json() == ""


def test_legacy_db_migration(tmp_path):
    """elapsed_seconds 컬럼이 없는 구버전 DB도 열리고 저장된다 (ALTER 마이그레이션)."""
    import sqlite3

    path = tmp_path / "old.db"
    # 구버전 스키마 흉내: 컬럼 없이 생성
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE inspections (record_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "film_id TEXT NOT NULL DEFAULT '', block TEXT NOT NULL DEFAULT '', "
            "weld_id TEXT NOT NULL DEFAULT '', joint_type TEXT NOT NULL DEFAULT '', "
            "thickness_mm REAL, quality_level TEXT NOT NULL DEFAULT '', "
            "inspector TEXT NOT NULL DEFAULT '', overall_passed INTEGER NOT NULL, "
            "defect_count INTEGER NOT NULL DEFAULT 0, report_text TEXT NOT NULL DEFAULT '', "
            "report_source TEXT NOT NULL DEFAULT '', image_name TEXT NOT NULL DEFAULT '', "
            "image_w INTEGER, image_h INTEGER, payload_json TEXT NOT NULL)"
        )
    db = Archive(path)  # ALTER로 컬럼 추가되어야 함
    db.save(_record("r1", elapsed=42.0))
    assert db.get("r1").elapsed_seconds == 42.0
