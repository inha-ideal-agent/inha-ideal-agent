"""Archive(SQLite 아카이브 + YOLO 라벨 export) 테스트."""

from __future__ import annotations

import pytest

from rtworkbench.db import Archive, YOLO_CLASSES
from rtworkbench.models import (
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    Measurement,
    RuleVerdict,
)


# ---------------------------------------------------------------- 픽스처/팩토리


def make_record(
    record_id: str = "rec-0001",
    *,
    block: str = "3번 블록",
    weld_id: str = "BW-12",
    passed: bool = True,
    created_at: str = "2026-08-30T10:00:00",
    image_name: str = "film_001.png",
    image_size: tuple[int, int] = (200, 100),
    candidates: list[DefectCandidate] | None = None,
    verdicts: list[RuleVerdict] | None = None,
    report_text: str = "판정 결과 이상 없음.",
) -> InspectionRecord:
    """테스트용 InspectionRecord 팩토리."""
    if candidates is None:
        candidates = [
            DefectCandidate(
                id="df-a1",
                defect_type="porosity",
                bbox=(50.0, 20.0, 150.0, 80.0),
                confidence=0.8,
                source="ai",
                status="accepted",
            ),
        ]
    if verdicts is None:
        verdicts = [
            RuleVerdict(
                defect_id="df-a1",
                defect_type="porosity",
                size_mm=1.5,
                quality_level="B",
                thickness_mm=12.0,
                limit_mm=2.4,
                passed=True,
                clause="DEMO-2011",
                detail="한도 이내",
            ),
        ]
    ctx = InspectionContext(
        film_id=f"F-{record_id}",
        block=block,
        weld_id=weld_id,
        thickness_mm=12.0,
        quality_level="B",
        inspector="홍길동",
        scale_mm_per_px=0.1,
        scale_ref="납마커 10mm",
    )
    measurements = [
        Measurement(defect_id="df-a1", p1=(50.0, 20.0), p2=(150.0, 80.0),
                    length_px=116.6, length_mm=11.66),
    ]
    return InspectionRecord(
        record_id=record_id,
        context=ctx,
        candidates=candidates,
        measurements=measurements,
        verdicts=verdicts,
        overall_passed=passed,
        report_text=report_text,
        report_source="template",
        image_name=image_name,
        image_size=image_size,
        created_at=created_at,
    )


@pytest.fixture
def archive(tmp_path):
    return Archive(db_path=tmp_path / "test.db")


# ---------------------------------------------------------------- 저장/조회


def test_save_get_roundtrip(archive):
    """save→get 왕복 직렬화: to_json 동등성."""
    rec = make_record()
    archive.save(rec)
    loaded = archive.get(rec.record_id)
    assert loaded is not None
    assert loaded.to_json() == rec.to_json()


def test_get_missing_returns_none(archive):
    assert archive.get("없는-ID") is None


def test_resave_replaces_no_duplicate(archive):
    """같은 record_id 재저장 시 교체 — 검색 결과에 중복 없음, 내용은 최신본."""
    rec = make_record("rec-x", report_text="첫 저장")
    archive.save(rec)
    rec2 = make_record("rec-x", report_text="수정된 소견서", passed=False)
    archive.save(rec2)

    df = archive.search()
    assert len(df) == 1
    loaded = archive.get("rec-x")
    assert loaded.report_text == "수정된 소견서"
    assert loaded.overall_passed is False
    # defects 테이블도 교체됐는지 — 통계로 확인 (candidate 1건만 존재해야 함)
    st = archive.stats()
    assert st["ai_proposed"] == 1


# ---------------------------------------------------------------- 검색


@pytest.fixture
def populated(archive):
    """검색 시나리오용 3건: 블록/결함유형/합부/날짜가 서로 다름."""
    # A: 3번 블록, 기공 accepted, 합격, 8/28
    archive.save(make_record(
        "rec-A", block="3번 블록", weld_id="V-01", passed=True,
        created_at="2026-08-28T09:00:00",
        report_text="수직 이음에서 기공 1건, 한도 이내.",
    ))
    # B: 5번 블록, 균열 accepted + 기공 rejected, 불합격, 8/29
    archive.save(make_record(
        "rec-B", block="5번 블록", weld_id="H-02", passed=False,
        created_at="2026-08-29T09:00:00",
        image_name="film_002.png",
        candidates=[
            DefectCandidate(id="df-b1", defect_type="crack",
                            bbox=(10, 10, 60, 20), confidence=0.7,
                            source="ai", status="accepted"),
            DefectCandidate(id="df-b2", defect_type="porosity",
                            bbox=(0, 0, 5, 5), confidence=0.3,
                            source="ai", status="rejected"),
        ],
        verdicts=[
            RuleVerdict(defect_id="df-b1", defect_type="crack", size_mm=5.0,
                        quality_level="B", thickness_mm=12.0, limit_mm=None,
                        passed=False, clause="DEMO-100", detail="균열은 허용 불가"),
        ],
        report_text="균열 검출로 불합격.",
    ))
    # C: 3번 블록 후속 검사, 결함 없음, 합격, 8/30
    archive.save(make_record(
        "rec-C", block="3번 블록", weld_id="V-03", passed=True,
        created_at="2026-08-30T09:00:00",
        image_name="film_003.png",
        candidates=[], verdicts=[],
        report_text="지시 없음.",
    ))
    return archive


def test_search_block_partial_match(populated):
    df = populated.search(block="3번")
    assert set(df["record_id"]) == {"rec-A", "rec-C"}


def test_search_defect_type_accepted_only(populated):
    """defect_type 필터는 accepted 결함만 매칭 (rec-B의 rejected 기공은 제외)."""
    df = populated.search(defect_type="porosity")
    assert set(df["record_id"]) == {"rec-A"}
    df2 = populated.search(defect_type="crack")
    assert set(df2["record_id"]) == {"rec-B"}


def test_search_passed_filter(populated):
    df = populated.search(passed=False)
    assert set(df["record_id"]) == {"rec-B"}
    df2 = populated.search(passed=True)
    assert set(df2["record_id"]) == {"rec-A", "rec-C"}


def test_search_text(populated):
    """text는 report_text/film_id/weld_id/block LIKE 검색."""
    df = populated.search(text="수직 이음")
    assert set(df["record_id"]) == {"rec-A"}
    # weld_id에도 걸리는지
    df2 = populated.search(text="H-02")
    assert set(df2["record_id"]) == {"rec-B"}


def test_search_date_range(populated):
    df = populated.search(date_from="2026-08-29")
    assert set(df["record_id"]) == {"rec-B", "rec-C"}
    # 날짜만 준 date_to는 그 날 포함(경계 inclusive)
    df2 = populated.search(date_from="2026-08-28", date_to="2026-08-29")
    assert set(df2["record_id"]) == {"rec-A", "rec-B"}


def test_search_sorted_newest_first(populated):
    df = populated.search()
    assert list(df["record_id"]) == ["rec-C", "rec-B", "rec-A"]


def test_search_summary_columns(populated):
    """요약 컬럼 구성 + defect_types 한국어 콤마 결합 + defect_count=accepted 수."""
    df = populated.search()
    assert list(df.columns) == [
        "record_id", "created_at", "film_id", "block", "weld_id",
        "quality_level", "overall_passed", "defect_count", "defect_types",
        "inspector",
    ]
    row_a = df[df["record_id"] == "rec-A"].iloc[0]
    assert row_a["defect_types"] == "기공"
    assert row_a["defect_count"] == 1
    row_b = df[df["record_id"] == "rec-B"].iloc[0]
    assert row_b["defect_types"] == "균열"  # rejected 기공은 표기 안 함
    assert row_b["defect_count"] == 1  # accepted만 집계
    row_c = df[df["record_id"] == "rec-C"].iloc[0]
    assert row_c["defect_types"] == ""
    assert row_c["defect_count"] == 0


def test_search_multi_filter_and(populated):
    """복합 필터는 AND — '3번 블록 기공 이력' 시나리오."""
    df = populated.search(block="3번", defect_type="porosity")
    assert set(df["record_id"]) == {"rec-A"}


# ---------------------------------------------------------------- 통계


def test_stats_empty(archive):
    st = archive.stats()
    assert st["total"] == 0
    assert st["acceptance_rate"] is None
    assert st["defect_type_counts"] == {}


def test_stats(archive):
    """AI 3건 제안(2 채택/1 기각) + 사람 추가 1건 → 채택률 2/3."""
    archive.save(make_record(
        "rec-s1", passed=False, created_at="2026-08-29T10:00:00",
        candidates=[
            DefectCandidate(id="d1", defect_type="porosity",
                            bbox=(10, 10, 20, 20), confidence=0.9,
                            source="ai", status="accepted"),
            DefectCandidate(id="d2", defect_type="crack",
                            bbox=(30, 30, 90, 40), confidence=0.6,
                            source="ai", status="accepted"),
            DefectCandidate(id="d3", defect_type="porosity",
                            bbox=(0, 0, 4, 4), confidence=0.2,
                            source="ai", status="rejected"),
            DefectCandidate(id="d4", defect_type="undercut",
                            bbox=(50, 5, 80, 10), confidence=1.0,
                            source="human", status="accepted"),
        ],
        verdicts=[],
    ))
    archive.save(make_record("rec-s2", passed=True, created_at="2026-08-30T10:00:00"))

    st = archive.stats()
    assert st["total"] == 2
    assert st["passed"] == 1
    assert st["failed"] == 1
    # accepted만: rec-s1(기공1, 균열1, 언더컷1) + rec-s2(기공1)
    assert st["defect_type_counts"] == {"porosity": 2, "crack": 1, "undercut": 1}
    assert st["ai_proposed"] == 4  # rec-s1의 3건 + rec-s2의 1건
    assert st["ai_accepted"] == 3
    assert st["ai_rejected"] == 1
    assert st["human_added"] == 1  # 미탐 신호
    assert st["acceptance_rate"] == pytest.approx(3 / 4)


# ---------------------------------------------------------------- YOLO export


def test_export_yolo_labels(archive, tmp_path):
    """정규화 좌표 수치 검증 + classes.txt + rejected/unknown 제외 + 클립."""
    archive.save(make_record(
        "rec-y1", image_name="scan_A.png", image_size=(200, 100),
        candidates=[
            # 기공 accepted: bbox (50,20,150,80) → cx .5 cy .5 w .5 h .6
            DefectCandidate(id="y1", defect_type="porosity",
                            bbox=(50, 20, 150, 80), confidence=0.8,
                            source="ai", status="accepted"),
            # 균열 accepted, 좌상단이 범위 밖 → 클립 후 cx .25 cy .25 w .5 h .5
            DefectCandidate(id="y2", defect_type="crack",
                            bbox=(-20, -10, 100, 50), confidence=0.9,
                            source="human", status="accepted"),
            # rejected → 제외
            DefectCandidate(id="y3", defect_type="porosity",
                            bbox=(0, 0, 10, 10), confidence=0.2,
                            source="ai", status="rejected"),
            # 미분류 지시(unknown) accepted → 학습 클래스 아님, 제외
            DefectCandidate(id="y4", defect_type="unknown",
                            bbox=(0, 0, 10, 10), confidence=1.0,
                            source="human", status="accepted"),
        ],
        verdicts=[],
    ))

    out = tmp_path / "yolo"
    result = archive.export_yolo_labels(out)
    assert result["records"] == 1
    assert result["labels"] == 2
    assert result["out_dir"] == str(out)

    # classes.txt: DEFECT_TYPES 키 순서, 'unknown' 제외
    classes = (out / "classes.txt").read_text(encoding="utf-8").splitlines()
    assert classes == list(YOLO_CLASSES)
    assert "unknown" not in classes
    assert classes[0] == "porosity"
    assert classes[2] == "crack"

    lines = (out / "scan_A.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2  # rejected/unknown 제외됨

    # 라벨 수치 검증
    cls0, cx0, cy0, w0, h0 = lines[0].split()
    assert int(cls0) == 0  # porosity
    assert float(cx0) == pytest.approx(0.5)
    assert float(cy0) == pytest.approx(0.5)
    assert float(w0) == pytest.approx(0.5)
    assert float(h0) == pytest.approx(0.6)

    cls1, cx1, cy1, w1, h1 = lines[1].split()
    assert int(cls1) == 2  # crack
    assert float(cx1) == pytest.approx(0.25)
    assert float(cy1) == pytest.approx(0.25)
    assert float(w1) == pytest.approx(0.5)
    assert float(h1) == pytest.approx(0.5)

    # 모든 값 0~1 범위
    for line in lines:
        for v in line.split()[1:]:
            assert 0.0 <= float(v) <= 1.0


def test_export_yolo_empty_record_makes_empty_file(archive, tmp_path):
    """accepted 결함이 없는 record도 빈 라벨 파일(음성 예시) 생성."""
    archive.save(make_record(
        "rec-y2", image_name="scan_B.png", candidates=[], verdicts=[],
    ))
    out = tmp_path / "yolo2"
    result = archive.export_yolo_labels(out)
    assert result["records"] == 1
    assert result["labels"] == 0
    assert (out / "scan_B.txt").read_text(encoding="utf-8") == ""


def test_export_yolo_same_stem_records_merge_no_overwrite(archive, tmp_path):
    """회귀 방지: 같은 image_name stem의 레코드끼리 라벨을 덮어쓰지 않고 병합.

    과거엔 레코드별 write_text 로 나중 레코드가 앞 레코드의 라벨 파일을 통째로
    덮어써 라벨이 소실되고, 반환 'labels'가 디스크 실제 줄 수와 불일치했다.
    """
    archive.save(make_record(
        "rec-m1", image_name="scan.png", image_size=(200, 100),
        candidates=[DefectCandidate(id="m1", defect_type="porosity",
                                    bbox=(50, 20, 150, 80), confidence=0.8,
                                    source="ai", status="accepted")],
        verdicts=[],
    ))
    archive.save(make_record(
        "rec-m2", image_name="scan.png", image_size=(200, 100),
        candidates=[DefectCandidate(id="m2", defect_type="crack",
                                    bbox=(0, 0, 100, 50), confidence=0.9,
                                    source="human", status="accepted")],
        verdicts=[],
    ))

    out = tmp_path / "yolo_merge"
    result = archive.export_yolo_labels(out)
    assert result["records"] == 2

    lines = (out / "scan.txt").read_text(encoding="utf-8").splitlines()
    classes = sorted(int(line.split()[0]) for line in lines)
    assert len(lines) == 2  # 두 레코드의 라벨이 모두 남는다
    assert classes == [YOLO_CLASSES.index("porosity"), YOLO_CLASSES.index("crack")]
    # 반환 카운트 = 디스크에 실제 기록된 줄 수
    assert result["labels"] == 2
