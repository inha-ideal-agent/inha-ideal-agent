"""소견서 LLM 어댑터(report_llm) + PDF 출력(report_pdf) 테스트.

네트워크 호출 없이 검증한다: monkeypatch 로 API 키를 빈 값으로 강제하여
키-없음 경로(캐시 → 스킵 → 스킵 → 템플릿)만 타게 한다.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from rtworkbench import config
from rtworkbench.models import (
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    Measurement,
    ReportDetails,
    RuleVerdict,
)
from rtworkbench.report_llm import ReportWriter, build_payload, template_report
from rtworkbench.report_pdf import build_pdf

INSPECTOR_NAME = "홍길동"
IMAGE_NAME = "필름_스캔_001.png"


# ---------------------------------------------------------------- 픽스처


@pytest.fixture
def context() -> InspectionContext:
    return InspectionContext(
        film_id="F-2026-0830-01",
        block="3번 블록",
        weld_id="W-104",
        joint_type="맞대기(Butt)",
        thickness_mm=12.0,
        quality_level="B",
        inspector=INSPECTOR_NAME,  # 실명 — payload 에 절대 포함되면 안 된다
        technique="RT (필름 스캔)",
        scale_mm_per_px=0.1,
        scale_ref="납마커 10mm",
    )


@pytest.fixture
def verdicts() -> list[RuleVerdict]:
    return [
        RuleVerdict(
            defect_id="df-aaaa0001",
            defect_type="porosity",
            size_mm=3.2,
            quality_level="B",
            thickness_mm=12.0,
            limit_mm=2.4,
            passed=False,
            clause="DEMO-2011",
            detail="크기 3.20mm > 허용한계 2.40mm",
        ),
        RuleVerdict(
            defect_id="df-aaaa0002",
            defect_type="slag_inclusion",
            size_mm=1.1,
            quality_level="B",
            thickness_mm=12.0,
            limit_mm=2.0,
            passed=True,
            clause="DEMO-301",
            detail="크기 1.10mm 이하 허용한계 2.00mm",
        ),
    ]


@pytest.fixture
def measurements() -> list[Measurement]:
    return [
        Measurement(
            defect_id="df-aaaa0001",
            p1=(10.0, 10.0),
            p2=(42.0, 10.0),
            length_px=32.0,
            length_mm=3.2,
        )
    ]


@pytest.fixture
def no_keys(monkeypatch):
    """API 키를 빈 값으로 강제 — 실제 네트워크 호출이 절대 나가지 않게 한다."""
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")


@pytest.fixture
def record(context, verdicts, measurements) -> InspectionRecord:
    candidates = [
        DefectCandidate(
            id="df-aaaa0001",
            defect_type="porosity",
            bbox=(10.0, 10.0, 42.0, 20.0),
            confidence=0.83,
            status="accepted",
        ),
        DefectCandidate(
            id="df-aaaa0002",
            defect_type="slag_inclusion",
            bbox=(100.0, 50.0, 120.0, 60.0),
            confidence=1.0,
            source="human",
            status="accepted",
        ),
    ]
    report_text = template_report(build_payload(context, verdicts, measurements))
    return InspectionRecord(
        record_id="rec-12345678",
        context=context,
        candidates=candidates,
        measurements=measurements,
        verdicts=verdicts,
        overall_passed=False,
        report_text=report_text,
        report_source="template",
        image_name=IMAGE_NAME,
        image_size=(2048, 1024),
    )


# ---------------------------------------------------------------- build_payload


def test_build_payload_excludes_identifying_info(context, verdicts, measurements):
    payload = build_payload(context, verdicts, measurements)
    # 검사원 실명 제외 (비식별 원칙)
    assert INSPECTOR_NAME not in payload
    # 이미지 파일명/경로 제외 (build_payload 는 이미지를 알지 못한다)
    assert IMAGE_NAME not in payload
    assert ".png" not in payload
    assert "/home/" not in payload


def test_build_payload_includes_business_ids_and_verdicts(context, verdicts, measurements):
    payload = build_payload(context, verdicts, measurements)
    # 업무 식별자 + 검사 조건
    assert "F-2026-0830-01" in payload
    assert "W-104" in payload
    assert "3번 블록" in payload
    assert "맞대기(Butt)" in payload
    assert "12 mm" in payload
    assert "품질 등급: B" in payload
    # 결함별 수치/조항/합부
    assert "기공" in payload
    assert "3.20 mm" in payload
    assert "2.40 mm" in payload
    assert "DEMO-2011" in payload
    assert "슬래그 개재물" in payload
    assert "불합격" in payload
    # 종합 합부 (불합격 1건 → 전체 불합격)
    assert "[종합 판정]" in payload
    assert "불합격 1건" in payload


def test_build_payload_no_defects(context):
    payload = build_payload(context, [], [])
    assert "유의미한 결함 지시 없음" in payload
    assert "[종합 판정]" in payload
    assert "합격" in payload


# ---------------------------------------------------------------- ReportWriter


def test_writer_falls_back_to_template_without_keys(
    no_keys, tmp_path, context, verdicts, measurements
):
    cache_path = tmp_path / "llm_cache.json"
    writer = ReportWriter(cache_path=cache_path)
    payload = build_payload(context, verdicts, measurements)

    text, source = writer.generate(payload)
    assert source == "template"
    assert "RT 판독 소견서" in text  # 한국어 소견서 형태

    # 템플릿 결과는 캐시에 저장되지 않는다 (캐시는 API 성공 시에만 저장)
    assert not cache_path.exists()

    # 같은 payload 재호출도 여전히 template (캐시 히트가 아니어야 한다)
    text2, source2 = writer.generate(payload)
    assert source2 == "template"
    assert text2 == text


def test_writer_cache_hit(no_keys, tmp_path, context, verdicts, measurements):
    payload = build_payload(context, verdicts, measurements)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    cache_path = tmp_path / "llm_cache.json"
    cache_path.write_text(
        json.dumps(
            {key: {"text": "사전 캐시된 소견서 본문", "source": "claude"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    writer = ReportWriter(cache_path=cache_path)
    text, source = writer.generate(payload)
    assert source == "cache"
    assert text == "사전 캐시된 소견서 본문"


def test_writer_ignores_corrupt_cache(no_keys, tmp_path, context, verdicts, measurements):
    # 깨진 캐시 파일이 있어도 예외 없이 템플릿으로 폴백해야 한다
    cache_path = tmp_path / "llm_cache.json"
    cache_path.write_text("{{{{ 깨진 JSON", encoding="utf-8")
    writer = ReportWriter(cache_path=cache_path)
    text, source = writer.generate(build_payload(context, verdicts, measurements))
    assert source == "template"
    assert text


# ---------------------------------------------------------------- template_report


def test_template_report_with_defects(context, verdicts, measurements):
    report = template_report(build_payload(context, verdicts, measurements))
    for section in ("1. 검사 개요", "2. 결함 소견", "3. 판정", "4. 비고"):
        assert section in report
    assert "기공" in report
    assert "불합격" in report
    # 판정 주체 고지 + 데모 기준표 유의 문구
    assert "자격 판독원" in report
    assert "데모 기준" in report


def test_template_report_without_defects(context):
    report = template_report(build_payload(context, [], []))
    for section in ("1. 검사 개요", "2. 결함 소견", "3. 판정", "4. 비고"):
        assert section in report
    assert "유의미한 결함 지시 없음" in report
    assert "자격 판독원" in report


def test_template_report_never_fails_on_arbitrary_payload():
    # 섹션 마커가 없는 임의 텍스트도 소견서 형태로 감싼다
    report = template_report("아무 구조도 없는 자유 텍스트")
    assert "1. 검사 개요" in report
    assert "아무 구조도 없는 자유 텍스트" in report
    assert "4. 비고" in report


# ---------------------------------------------------------------- build_pdf


def test_build_pdf_bytes(record):
    pdf = build_pdf(record)
    assert isinstance(pdf, bytes)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1500


def test_build_pdf_no_defects_korean(context):
    # 결함 없는 한글 record 에서도 예외 없이 생성되어야 한다
    report_text = template_report(build_payload(context, [], []))
    rec = InspectionRecord(
        record_id="rec-empty001",
        context=context,
        candidates=[],
        measurements=[],
        verdicts=[],
        overall_passed=True,
        report_text=report_text,
        report_source="template",
        image_name="빈_필름.png",
        image_size=(1000, 800),
    )
    pdf = build_pdf(rec)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1500


def test_build_pdf_unmeasurable_defect(context):
    # limit_mm=None(허용 불가 유형, 예: 균열) 도 표에 문제 없이 들어가야 한다
    verdict = RuleVerdict(
        defect_id="df-crack001",
        defect_type="crack",
        size_mm=5.0,
        quality_level="B",
        thickness_mm=12.0,
        limit_mm=None,
        passed=False,
        clause="DEMO-100",
        detail="균열은 크기와 무관하게 허용 불가",
    )
    rec = InspectionRecord(
        record_id="rec-crack001",
        context=context,
        candidates=[],
        measurements=[],
        verdicts=[verdict],
        overall_passed=False,
        report_text=template_report(build_payload(context, [verdict], [])),
        report_source="template",
        image_name="필름_균열.png",
        image_size=(1600, 900),
    )
    pdf = build_pdf(rec)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1500


# ---------------------------------------------------------------- 회귀 방지 (캐시/타임아웃)


def test_save_cache_atomic_and_concurrent_no_lost_update(tmp_path):
    """회귀 방지: _save_cache 는 원자적 교체 + 락 — 동시 저장에도 항목이 유실되지 않는다."""
    import threading

    cache_path = tmp_path / "llm_cache.json"
    # 사전 시딩된 항목이 이후 저장에도 살아남아야 한다
    cache_path.write_text(
        json.dumps({"seed": {"text": "사전 캐시", "source": "claude"}}), encoding="utf-8"
    )
    writer = ReportWriter(cache_path=cache_path)

    threads = [
        threading.Thread(target=writer._save_cache, args=(f"k{i}", f"text-{i}", "claude"))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = json.loads(cache_path.read_text(encoding="utf-8"))  # 파일은 항상 유효 JSON
    assert "seed" in data  # 기존 항목 보존
    for i in range(8):
        assert data[f"k{i}"]["text"] == f"text-{i}"  # lost-update 없음
    assert not list(tmp_path.glob("*.tmp"))  # 임시 파일 잔존 없음


def test_try_anthropic_uses_bounded_timeout(monkeypatch, tmp_path):
    """회귀 방지: Anthropic 클라이언트에 짧은 timeout/재시도 제한을 명시한다.

    SDK 기본값(read 600s × 재시도 2회)이면 API 스톨 시 UI가 수십 분 동결된다.
    """
    import sys
    import types

    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop here — 생성자 kwargs 만 검증")

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")

    writer = ReportWriter(cache_path=tmp_path / "cache.json")
    assert writer._try_anthropic("payload") is None  # 예외는 삼키고 폴백(None)

    timeout = captured.get("timeout")
    assert isinstance(timeout, (int, float)) and timeout <= 60
    assert captured.get("max_retries") == 0


# ---------------------------------------------------------------- 그룹 판정 (단위 %)


def _group_verdict() -> RuleVerdict:
    return RuleVerdict(
        defect_id="GROUP:porosity",
        defect_type="porosity",
        size_mm=0.63,
        quality_level="B",
        thickness_mm=12.0,
        limit_mm=1.0,
        passed=True,
        clause="DEMO-2012-AREA",
        detail="기공 3건 원 근사 면적 합 12.57mm² ÷ (평가길이 100mm × 용접부 폭 20mm) = 0.63% ≤ 한계 1.0% → 합격",
        unit="%",
    )


def test_build_payload_group_verdict_uses_percent_unit_and_sum_label(context, verdicts, measurements):
    payload = build_payload(context, verdicts + [_group_verdict()], measurements)
    assert "합계(기공)" in payload
    assert "크기 0.63 %" in payload and "허용한계 1.00 %" in payload
    assert "0.63 mm" not in payload
    # 단일 verdict 표기는 그대로 mm
    assert "3.20 mm" in payload and "허용한계 2.40 mm" in payload


def test_build_pdf_with_group_verdict(context, verdicts):
    rec = InspectionRecord(
        record_id="rec-group001",
        context=context,
        candidates=[],
        measurements=[],
        verdicts=verdicts + [_group_verdict()],
        overall_passed=False,
        report_text="본문",
        report_source="template",
        image_name=IMAGE_NAME,
        image_size=(800, 600),
    )
    pdf = build_pdf(rec)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


# ---------------------------------------------------------------- 적용 기준표(규격 판본) 표기


def _pdf_text(pdf: bytes) -> str:
    pymupdf = pytest.importorskip("pymupdf")
    return "".join(page.get_text() for page in pymupdf.open(stream=pdf, filetype="pdf"))


def test_build_pdf_footer_carries_criteria_name_and_version(record):
    record.criteria_name = "데모 기준표 — 판본 테스트"
    record.criteria_version = "9.9-test"
    text = _pdf_text(build_pdf(record))
    assert "적용 기준표" in text
    assert "데모 기준표 — 판본 테스트" in text and "9.9-test" in text
    assert "데모 기준" in text  # 데모 기준표 디스클레이머는 그대로


def test_build_pdf_legacy_record_without_criteria_is_marked_unrecorded(record):
    assert record.criteria_name == "" and record.criteria_label == ""  # 판본 분리 전 기록
    text = _pdf_text(build_pdf(record))
    assert "미기록" in text


# ---------------------------------------------------------------- 선급 보고 항목 (IACS UR W33 §8)


def _filled_report() -> ReportDetails:
    return ReportDetails(
        hull_number="H-2031", personnel_qualification="ISO 9712 RT Level 2",
        steel_grade="AH36", repairs_count=3,
        source_type_size="Ir-192, 2x2 mm", xray_kv="200", sfd_mm="700",
        iqi_sensitivity="W13 (1.6%)", density="2.3~2.8", rt_acceptance_class="ISO 10675-1 Level 1",
    )


def test_build_payload_has_conditions_section_only_when_report_filled(context, verdicts, measurements):
    """'[촬영 조건]' 섹션은 입력된 보고 항목이 있을 때만, 입력된 항목만 실린다."""
    assert context.report.is_empty()
    empty = build_payload(context, verdicts, measurements)
    assert "[촬영 조건]" not in empty and "선체 번호" not in empty

    context.report = _filled_report()
    payload = build_payload(context, verdicts, measurements)
    assert "[촬영 조건]" in payload
    assert "선체 번호: H-2031" in payload
    assert "검사자 자격 등급: ISO 9712 RT Level 2" in payload
    assert "보수 횟수(2회 초과 시): 3회" in payload
    assert "X선 관전압: 200 kV" in payload and "선원-필름 거리(SFD): 700 mm" in payload
    assert "농도: 2.3~2.8" in payload and "RT 합격 등급: ISO 10675-1 Level 1" in payload
    assert "용접 방법" not in payload  # 미입력 항목은 나열하지 않는다
    # 섹션 순서: 검사 개요 → 촬영 조건 → 결함별 판정 → 종합 판정
    assert payload.index("[검사 개요]") < payload.index("[촬영 조건]") < payload.index("[결함별 판정]")
    # 비식별 원칙은 그대로 — 검사원 실명은 여전히 제외
    assert INSPECTOR_NAME not in payload


def test_build_payload_report_fields_do_not_change_verdict_lines(context, verdicts, measurements):
    """보고 항목은 문서 전용 — 결함별 판정/종합 판정 섹션 텍스트는 바뀌지 않는다."""
    before = build_payload(context, verdicts, measurements)
    context.report = _filled_report()
    after = build_payload(context, verdicts, measurements)
    tail = lambda s: s[s.index("[결함별 판정]"):]  # noqa: E731
    assert tail(before) == tail(after)


def test_template_report_carries_conditions_under_overview(context, verdicts, measurements):
    context.report = _filled_report()
    report = template_report(build_payload(context, verdicts, measurements))
    assert "촬영 조건 (IACS UR W33 §8 보고 항목)" in report
    assert "선체 번호: H-2031" in report
    assert report.index("1. 검사 개요") < report.index("선체 번호") < report.index("2. 결함 소견")
    # 비어 있으면 그 소제목도 없다
    context.report = ReportDetails()
    assert "촬영 조건" not in template_report(build_payload(context, verdicts, measurements))


def test_build_pdf_report_section_present_only_when_filled(record):
    """PDF '5. 선급 보고 항목' 섹션 — 입력이 있을 때만, 입력된 항목만 + n/25 표기."""
    assert record.context.report.is_empty()
    text = _pdf_text(build_pdf(record))
    assert "5. 선급 보고 항목" not in text
    # 기존 섹션은 그대로
    for section in ("1. 검사 정보", "2. 결함 판정", "3. 종합 판정", "4. 소견서 본문", "적용 기준표"):
        assert section in text

    record.context.report = _filled_report()
    pdf = build_pdf(record)
    assert pdf.startswith(b"%PDF")
    text = _pdf_text(pdf)
    assert "5. 선급 보고 항목(IACS UR W33 §8)" in text
    # 충족률은 W33 불릿 기준(승인 기록 = 판정 완료): 검사일·합격 기준·검사 결과(항상/판정) + 블록·용접부 ID
    # (컨텍스트) + 검사자(판독원+자격) + 합부 선언(판정+판독원) + 보수 3회 + 선원·농도·RT 등급 = 11/25.
    # 선체 번호(길이 없음)·SFD(기법·시간 없음)·IQI 감도(종류·위치 없음)·강재(용접 방법 없음)는 불릿 미충족
    # → 입력 필드 수 10/25(ReportDetails.coverage) 와 다른 값이다.
    assert record.context.report.coverage() == (10, 25)
    assert "보고 항목 충족 11/25" in text and "입력 항목 10/25" not in text
    assert "일반 13" in text and "RT 12" in text and "미충족:" in text
    assert "일반 항목 (§8.2)" in text and "RT 촬영 조건 (§8.5)" in text
    assert "H-2031" in text and "Hull number" in text  # 한국어 라벨 + W33 원문 병기
    assert "ISO 9712 RT Level 2" in text and "3회" in text
    assert "200 kV" in text and "700 mm" in text and "2.3~2.8" in text
    # 미입력 항목은 표에서 생략(W33 원문 라벨 'Welding process' 없음) — 한국어 '용접 방법'은 헤더의
    # 미충족 목록('강재 등급·이음 종류·모재 두께·용접 방법')에만 한 번 나온다
    assert "Welding process" not in text and text.count("용접 방법") == 1
    for section in ("1. 검사 정보", "2. 결함 판정", "3. 종합 판정", "4. 소견서 본문", "적용 기준표"):
        assert section in text
    # 가운뎃점(U+00B7)은 내장 CID 폰트에 글리프가 없어 사라지므로 아래아(U+318D)로 렌더링된다
    assert "선원 종류ㆍ크기" in text and "선원 종류크기" not in text
    assert "일반 13 ㆍ" in text  # 캡션의 가운뎃점도 동일 처리
    assert "표시ㆍ문서화" in text  # 기존 푸터('후보 표시·문서화')도 더 이상 글자가 빠지지 않는다
