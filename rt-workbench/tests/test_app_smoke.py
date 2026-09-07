"""app.py 스모크 테스트 — Streamlit AppTest로 예외 없이 부팅되는지 확인.

주의: 커스텀 컴포넌트(streamlit-image-coordinates)는 AppTest에서 None을 반환하므로
앱 스크립트는 None-안전해야 한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_boots_without_exception():
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, f"앱 부팅 중 예외 발생: {at.exception}"


def _boot_with_image():
    """이미지가 주입된 AppTest 세션 (업로드 모드 — 컴포넌트/업로더는 None 안전)."""
    import numpy as np

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    at.session_state["wb_image"] = np.full((100, 200), 128, dtype=np.uint8)
    at.session_state["wb_image_name"] = "test.png"
    at.session_state["wb_image_key"] = "test:injected"
    at.run(timeout=60)
    return at


def _button(at, key):
    return next(b for b in at.button if b.key == key)


def test_unmeasured_accepted_defect_blocks_judgment_and_approval():
    """회귀 방지(false PASS): 미측정 채택 결함(crack 포함)은 판정 확정·승인 불가."""
    from rtworkbench.models import DefectCandidate

    at = _boot_with_image()
    at.session_state["wb_candidates"] = [
        DefectCandidate(id="c-crack", defect_type="crack", bbox=(10, 10, 50, 30),
                        confidence=0.9, source="ai", status="accepted"),
    ]
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_overall"] is None  # 합격 확정 금지
    assert at.session_state["wb_verdicts"] == []
    assert at.error  # 미측정 안내
    assert _button(at, "wb_btn_approve").disabled  # 승인 불가


def test_meta_scale_apply_invalidates_judgment_and_recomputes_mm():
    """회귀 방지: 스케일 적용은 기존 판정 무효화 + 측정 mm 재계산."""
    import pytest as _pytest

    from rtworkbench.models import Measurement

    at = _boot_with_image()
    at.session_state["wb_sample_meta"] = {"mm_per_px": 0.05, "marker": {"length_mm": 5}}
    at.session_state["wb_measurements"] = {
        # 옛(잘못된) 스케일로 계산된 mm=10.0 — 새 스케일 0.05로는 50px=2.5mm
        "c1": Measurement(defect_id="c1", p1=(0.0, 0.0), p2=(0.0, 50.0),
                          length_px=50.0, length_mm=10.0),
    }
    at.session_state["wb_overall"] = True  # 스케일 적용 전의 낡은 판정
    at.run(timeout=60)
    _button(at, "wb_btn_meta_scale").click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_overall"] is None  # 판정 무효화
    m = at.session_state["wb_measurements"]["c1"]
    assert m.length_mm == _pytest.approx(2.5)  # 새 스케일로 무조건 재계산


def test_thickness_change_invalidates_judgment():
    """회귀 방지: 판정 후 모재 두께 변경 시 기존 verdict/overall 무효화."""
    at = _boot_with_image()
    at.session_state["wb_overall"] = True
    at.run(timeout=60)
    at.number_input(key="wb_ctx_thickness").set_value(6.0)
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_overall"] is None


def test_new_image_clears_report_text():
    """회귀 방지: 이미지 교체 시 이전 필름 소견서 본문이 남지 않는다."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    at.radio(key="wb_src_mode").set_value("샘플 이미지")
    at.run(timeout=60)
    assert at.session_state["wb_image_name"] == "sample_01.png"
    at.text_area(key="wb_report_text").set_value("필름 A 전용 소견서 본문")
    at.run(timeout=60)
    at.selectbox(key="wb_sample_sel").select_index(1)
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_image_name"] == "sample_02.png"
    assert at.session_state["wb_report_text"] == ""


def test_detection_success_notice_visible():
    """회귀 방지: AI 탐지 성공 안내(recall 우선/기각 1클릭)가 최종 렌더에 표시된다."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    at.radio(key="wb_src_mode").set_value("샘플 이미지")
    at.run(timeout=60)
    _button(at, "wb_btn_detect").click()
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_candidates"]  # 후보는 추가되고
    assert any("기각" in s.value for s in at.success)  # 안내문도 남는다


def test_eval_window_change_invalidates_judgment():
    """회귀 방지: 판정 후 평가 길이/용접부 폭 변경 시 기존 verdict/overall 무효화."""
    for key, value in (("wb_ctx_eval_len", 200.0), ("wb_ctx_weld_w", 30.0)):
        at = _boot_with_image()
        at.session_state["wb_overall"] = True
        at.run(timeout=60)
        at.number_input(key=key).set_value(value)
        at.run(timeout=60)
        assert not at.exception
        assert at.session_state["wb_overall"] is None, key


def test_judgment_appends_group_verdict_using_sidebar_eval_window():
    """판정 실행 → 단일 verdict 뒤에 그룹 verdict, 사이드바 평가 길이/용접부 폭이 반영된다."""
    from rtworkbench.models import DefectCandidate, Measurement

    at = _boot_with_image()
    at.session_state["wb_candidates"] = [
        DefectCandidate(id="p1", defect_type="porosity", bbox=(10, 10, 30, 30),
                        confidence=1.0, source="human", status="accepted"),
    ]
    at.session_state["wb_measurements"] = {
        "p1": Measurement(defect_id="p1", p1=(0.0, 0.0), p2=(0.0, 20.0),
                          length_px=20.0, length_mm=2.0),
    }
    at.session_state["wb_scale_mm_per_px"] = 0.1  # 20px → 2.0mm
    at.run(timeout=60)
    at.number_input(key="wb_ctx_eval_len").set_value(50.0)
    at.number_input(key="wb_ctx_weld_w").set_value(10.0)
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert not at.exception
    verdicts = at.session_state["wb_verdicts"]
    assert [v.defect_id for v in verdicts] == ["p1", "GROUP:porosity"]
    g = verdicts[-1]
    assert g.unit == "%" and "평가길이 50mm × 용접부 폭 10mm" in g.detail
    assert at.session_state["wb_overall"] is True
    # 스케일 확정 → 측정선 중점 × 스케일 = 용접선 축 위치 → 창 이동(최악 구간) 판정, 캡션에 구간 표시
    assert g.window_mm == (0.0, 50.0) and "최악 50mm 구간 [0.0~50.0 mm] 내 기공 1건" in g.detail
    assert any("최악 50mm 구간" in c.value and "기공 [0~50 mm]" in c.value for c in at.caption)


def test_judgment_with_measurements_uses_worst_100mm_window_from_measurement_positions():
    """측정선 중점 × 스케일 → 용접선(x축) 위치 → 100 mm 창 이동: 군집 3건만 담는 최악 구간으로 판정.

    detail 에 '구간' 이 들어가고, 판정 캡션에 선택된 구간이 표시된다. 스케일이 없으면 전체 구간 폴백.
    """
    from rtworkbench.models import DefectCandidate, Measurement

    at = _boot_with_image()
    # 스케일 0.1 mm/px: x 중점 100/300/500 px → 10/30/50 mm 군집, 2000/4000 px → 200/400 mm 원거리
    xs = {"p1": 100.0, "p2": 300.0, "p3": 500.0, "p4": 2000.0, "p5": 4000.0}
    at.session_state["wb_candidates"] = [
        DefectCandidate(id=k, defect_type="porosity", bbox=(x - 12, 10, x + 12, 34),
                        confidence=1.0, source="human", status="accepted")
        for k, x in xs.items()
    ]
    at.session_state["wb_measurements"] = {
        k: Measurement(defect_id=k, p1=(x - 12.0, 20.0), p2=(x + 12.0, 20.0), length_px=24.0, length_mm=2.4)
        for k, x in xs.items()
    }
    at.session_state["wb_scale_mm_per_px"] = 0.1
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert not at.exception
    verdicts = at.session_state["wb_verdicts"]
    g = verdicts[-1]
    assert g.defect_id == "GROUP:porosity" and "구간" in g.detail
    assert g.window_mm == (10.0, 110.0)
    assert "최악 100mm 구간 [10.0~110.0 mm] 내 기공 3건(필름 전체 5건 중)" in g.detail
    assert g.size_mm == 0.68 and g.passed is True  # 전체 합산이었다면 1.13% 불합격
    assert at.session_state["wb_overall"] is True
    caption = next(c.value for c in at.caption if "최악 100mm 구간" in c.value)
    assert "기공 [10~110 mm]" in caption and "IACS UR W33 100 mm 판독 구간" in caption

    # 스케일 미확정(주입된 mm 만 있음) → 위치 없음 → 전체를 한 구간으로 보수적 합산 + 안내 캡션
    at.session_state["wb_scale_mm_per_px"] = None
    at.session_state["wb_overall"] = None
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert not at.exception
    g = at.session_state["wb_verdicts"][-1]
    assert g.window_mm is None and "위치 정보 없음 → 전체를 한 구간으로 보수적 합산" in g.detail
    assert g.size_mm == 1.13 and g.passed is False
    assert any("전체를 한 구간으로 보수적 합산" in c.value and "기공" in c.value for c in at.caption)


# ─────────────── 선급 보고 항목(IACS UR W33 §8.2·8.5) — 사이드바 expander ───────────────

REPORT_EXPANDER_LABEL = "선급 보고 항목(IACS UR W33 §8.2·8.5)"


def _coverage_caption(at) -> str:
    return next(c.value for c in at.caption if c.value.startswith("📋 선급 보고 항목"))


def test_report_details_expander_exists_and_coverage_caption_renders():
    """사이드바 expander(접힘)에 25개 입력 위젯이 있고 W33 불릿 기준 'n/25 충족 — 미입력: …' 캡션이 보인다.

    빈 컨텍스트라도 검사일(승인 시각)·합격 기준(품질등급+기준표)은 항상 충족 → 2/25 에서 시작한다.
    """
    from rtworkbench.models import REPORT_ITEMS

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception
    exp = next(e for e in at.expander if e.label == REPORT_EXPANDER_LABEL)
    assert exp.proto.expanded is False  # 기본 접힘
    cap = _coverage_caption(at)
    assert cap.startswith("📋 선급 보고 항목(IACS UR W33 §8) 2/25 충족 — 미입력: ")
    missing = cap.split("미입력: ", 1)[1].split(", ")
    assert len(missing) == 23 and missing[0] == "선체 번호·용접부 위치·검사 길이" and missing[-1] == "RT 합격 등급"
    assert "검사일" not in missing and "합격 기준" not in missing  # 항상 충족되는 두 불릿은 미입력 목록에 없다
    # 항목마다 위젯 1개 — 보수 횟수만 number_input, 나머지는 text_input (help 에 W33 원문)
    for item in REPORT_ITEMS:
        key = f"wb_rpt_{item.key}"
        if item.key == "repairs_count":
            w = at.number_input(key=key)
            assert w.value == 0
        else:
            w = at.text_input(key=key)
            assert w.value == ""
            assert item.label_en in (w.help or "")
        assert item.label_ko in w.label

    # 선체 번호만으론 '선체 번호·위치·길이' 불릿 미충족, SFD 만으론 '촬영 기법·노출 시간·SFD' 미충족,
    # 보수 횟수 3 은 충족 → 2 + 1 = 3/25 (입력 필드 수 3 과 우연히 같지만 다른 단위)
    at.text_input(key="wb_rpt_hull_number").set_value("H-2031")
    at.text_input(key="wb_rpt_sfd_mm").set_value("700")
    at.number_input(key="wb_rpt_repairs_count").set_value(3)
    at.run(timeout=60)
    assert not at.exception
    cap = _coverage_caption(at)
    assert cap.startswith("📋 선급 보고 항목(IACS UR W33 §8) 3/25 충족 — 미입력: ")
    assert "보수 횟수" not in cap and "촬영 기법·노출 시간·SFD" in cap
    # 불릿의 나머지 하위 정보를 채우면 그 불릿이 충족된다: 용접부 ID + 검사 길이 → 선체 번호 불릿(+1),
    # 촬영 기법 + 노출 시간 → SFD 불릿(+1), 용접부 ID → 용접부 식별 불릿(+1) = 6/25
    at.text_input(key="wb_ctx_weld_id").set_value("BL3-V-012")
    at.text_input(key="wb_rpt_weld_length_mm").set_value("300")
    at.text_input(key="wb_rpt_exposure_technique").set_value("SWSI")
    at.text_input(key="wb_rpt_exposure_time_s").set_value("90")
    at.run(timeout=60)
    assert not at.exception
    cap = _coverage_caption(at)
    assert cap.startswith("📋 선급 보고 항목(IACS UR W33 §8) 6/25 충족 — 미입력: ")
    assert "선체 번호" not in cap and "SFD" not in cap and "검사 용접부 식별" not in cap


def test_report_details_do_not_invalidate_judgment_but_flow_into_record(monkeypatch, tmp_path):
    """보고 항목 변경은 판정을 무효화하지 않고(룰 입력 아님), 승인 기록·PDF에는 실린다."""
    from rtworkbench import db as archive_db
    from rtworkbench.models import DefectCandidate, Measurement

    # 앱은 archive_db.Archive() 를 기본 인자(config.DB_PATH — import 시점에 고정)로 부른다.
    # 실제 data/ DB 를 건드리지 않도록 기본 DB 경로를 임시 파일로 바꾼다.
    Archive = archive_db.Archive
    orig_init = Archive.__init__
    monkeypatch.setattr(
        Archive, "__init__",
        lambda self, db_path=tmp_path / "t.db": orig_init(self, db_path),
    )
    at = _boot_with_image()
    at.session_state["wb_candidates"] = [
        DefectCandidate(id="p1", defect_type="porosity", bbox=(10, 10, 30, 30),
                        confidence=1.0, source="human", status="accepted"),
    ]
    at.session_state["wb_measurements"] = {
        "p1": Measurement(defect_id="p1", p1=(0.0, 0.0), p2=(0.0, 20.0),
                          length_px=20.0, length_mm=2.0),
    }
    at.session_state["wb_scale_mm_per_px"] = 0.1
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert at.session_state["wb_overall"] is True
    verdicts_before = [v.to_dict() for v in at.session_state["wb_verdicts"]]
    at.session_state["wb_pdf_bytes"] = b"%PDF-stale"  # 미리 생성해 둔 PDF

    at.text_input(key="wb_rpt_hull_number").set_value("H-2031")
    at.text_input(key="wb_rpt_xray_kv").set_value("200")
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_overall"] is True  # 판정 유지
    assert [v.to_dict() for v in at.session_state["wb_verdicts"]] == verdicts_before
    assert at.session_state["wb_pdf_bytes"] is None  # 낡은 PDF만 버린다
    # W33 불릿 기준: 검사일·합격 기준(항상) + 검사 결과(판정 완료) = 3 — 선체 번호만·관전압만으론 불릿 미충족
    cap = _coverage_caption(at)
    assert cap.startswith("📋 선급 보고 항목(IACS UR W33 §8) 3/25 충족 — 미입력: ")
    assert "검사 결과" not in cap and "합부 선언·평가일·평가자" in cap  # 판독원 성명이 없어 합부 선언은 미충족

    _button(at, "wb_btn_approve").click()
    at.run(timeout=60)
    assert not at.exception
    rec = Archive(tmp_path / "t.db").get(at.session_state["wb_record_id"])
    assert rec is not None
    assert rec.context.report.hull_number == "H-2031" and rec.context.report.xray_kv == "200"
    assert rec.context.report.coverage() == (2, 25)  # 입력 필드 수(구 지표)는 그대로 쓸 수 있다
    from rtworkbench.models import w33_coverage
    assert w33_coverage(rec.context, rec)[:2] == (3, 25)
    pdf = at.session_state["wb_pdf_bytes"]
    assert pdf and pdf.startswith(b"%PDF")
    pymupdf = pytest.importorskip("pymupdf")
    text = "".join(pg.get_text() for pg in pymupdf.open(stream=pdf, filetype="pdf"))
    assert "5. 선급 보고 항목" in text and "H-2031" in text and "200 kV" in text
    assert "보고 항목 충족 3/25" in text  # PDF 5절 헤더도 W33 불릿 기준


# ─────────────── 기준표(규격 판본) 선택 — 사이드바 selectbox ───────────────

CRITERIA_2023 = "demo_iso5817_like"
CRITERIA_2014 = "demo_iso5817_2014_like"


def test_criteria_selectbox_exists_with_default_2023():
    """사이드바 '기준표(규격 판본)' selectbox — 기본 선택은 2023/2021 구조 데모(config.CRITERIA_PATH)."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception
    sb = at.selectbox(key="wb_criteria_id")
    assert sb.label == "기준표(규격 판본)"
    assert at.session_state["wb_criteria_id"] == CRITERIA_2023
    assert len(sb.options) >= 2  # 두 판본 모두 목록에 있고, 표시는 meta.name
    assert any("2023" in o for o in sb.options) and any("2014" in o for o in sb.options)
    assert any("📐" in c.value and "2023" in c.value for c in at.caption)  # 판본 메모 캡션


def test_criteria_change_invalidates_judgment():
    """회귀 방지: 판정 후 기준표(규격 판본) 변경 시 기존 verdict/overall 무효화."""
    at = _boot_with_image()
    at.session_state["wb_overall"] = True
    at.run(timeout=60)
    at.selectbox(key="wb_criteria_id").select(CRITERIA_2014)
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_criteria_id"] == CRITERIA_2014
    assert at.session_state["wb_overall"] is None
    assert at.session_state["wb_verdicts"] == []
    assert at.session_state["wb_criteria_used"] is None


def test_judgment_uses_selected_criteria_and_pdf_carries_it():
    """판정은 선택된 판본의 JSON으로 — 같은 결함이 2023 구조에선 불합격, 2014 구조에선 합격."""
    from rtworkbench.models import DefectCandidate, Measurement

    at = _boot_with_image()
    at.session_state["wb_candidates"] = [
        DefectCandidate(id="p1", defect_type="porosity", bbox=(10, 10, 30, 30),
                        confidence=1.0, source="human", status="accepted"),
    ]
    at.session_state["wb_measurements"] = {
        "p1": Measurement(defect_id="p1", p1=(0.0, 0.0), p2=(0.0, 28.0),
                          length_px=28.0, length_mm=2.8),
    }
    at.session_state["wb_scale_mm_per_px"] = 0.1  # 28px → 2.8mm ; t=12, B
    at.run(timeout=60)

    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert not at.exception
    v = at.session_state["wb_verdicts"][0]
    assert (v.limit_mm, v.passed, v.clause) == (2.4, False, "DEMO-2011")
    used = at.session_state["wb_criteria_used"]
    assert used["id"] == CRITERIA_2023 and "2023" in used["name"] and used["version"]
    assert any("적용 기준표" in c.value and "2023" in c.value for c in at.caption)

    at.selectbox(key="wb_criteria_id").select(CRITERIA_2014)
    at.run(timeout=60)
    assert at.session_state["wb_overall"] is None  # 판본 변경 → 재판정 필요
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert not at.exception
    v = at.session_state["wb_verdicts"][0]
    assert (v.limit_mm, v.passed, v.clause) == (3.0, True, "DEMO14-2011")
    assert at.session_state["wb_overall"] is True
    assert at.session_state["wb_criteria_used"]["id"] == CRITERIA_2014

    # PDF(승인 전 미리 생성)에도 적용 기준표가 실린다
    _button(at, "wb_btn_pdf_preview").click()
    at.run(timeout=60)
    assert not at.exception
    pdf = at.session_state["wb_pdf_bytes"]
    assert pdf and pdf.startswith(b"%PDF")
    pymupdf = pytest.importorskip("pymupdf")  # PDF 본문 추출 — 미설치 환경에선 여기까지만 검증
    text = "".join(pg.get_text() for pg in pymupdf.open(stream=pdf, filetype="pdf"))
    assert "적용 기준표" in text and "2014" in text and "0.2-demo-2014" in text


# ─────────────── 📖 기준표·정보 탭 — 용어 풀이 · 참고 자료 · 제안서 문구 정렬 ───────────────

README_PATH = APP_PATH.parent / "README.md"


def _all_text(elements) -> str:
    return "\n".join(e.value for e in elements)


def test_info_tab_renders_glossary_and_references_without_exception():
    """정보 탭: 맨 위 용어 풀이 캡션(8줄 이내)과 맨 끝 '참고 자료' 절이 예외 없이 렌더된다."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception, f"정보 탭 렌더 중 예외: {at.exception}"

    glossary = next(c.value for c in at.caption if "이 프로토타입은 무엇이고 무엇이 아닌가" in c.value)
    assert len([ln for ln in glossary.strip().splitlines() if ln.strip()]) <= 8
    for term in ("RT", "방사선투과검사", "판독", "선급", "KR", "DNV", "ABS", "IACS", "IQI"):
        assert term in glossary, term

    assert any(s.value == "참고 자료" for s in at.subheader)
    refs = next(m.value for m in at.markdown if "**규칙·규격**" in m.value)
    for group in ("**규칙·규격**", "**데이터셋**", "**논문**", "**도구**"):
        assert group in refs, group
    for key in ("IACS UR W33 Rev.1/Corr.1", "IACS UR W33 Rev.2", "2026.07.15", "2028.01.01", "100 mm",
                "ISO 10675-1:2021", "ISO 5817:2023", "ISO 17636-1/-2:2022", "ISO 14096-2", "ISO 9712:2021",
                "GDXray", "RIAWELC", "AI Hub 71761", "WeldVGG", "Palma-Ramírez", "Lu et al.",
                "YOLO26", "AGPL-3.0", "Streamlit", "Ollama", "vLLM", "EXAONE 3.5", "HyperCLOVA X SEED"):
        assert key in refs, key
    # 검색 스니펫으로만 확인한 항목은 표시가 붙는다
    assert "HyperCLOVA X SEED — NAVER 한국어 모델 후보 (검색 요약 기준)" in refs
    # AI Hub 수치는 활용 신청 시 데이터셋 페이지에서 확인한 값 — 계획서 S04와 같은 확인 상태(검색 요약 아님)
    aihub_line = refs.split("AI Hub 71761", 1)[1].splitlines()[0]
    assert "활용 신청 시 데이터셋 페이지에서 확인" in aihub_line and "검색 요약 기준" not in aihub_line


def test_references_list_matches_readme():
    """앱의 참고 자료 번호 항목은 README.md '## 참고 자료' 절과 한 줄도 다르지 않다."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    refs = next(m.value for m in at.markdown if "**규칙·규격**" in m.value)
    numbered = [ln.strip() for ln in refs.splitlines() if ln.strip()[:2].rstrip(".").isdigit()]
    assert len(numbered) == 18

    readme = README_PATH.read_text(encoding="utf-8")
    assert "## 참고 자료" in readme
    readme_section = readme.split("## 참고 자료", 1)[1]
    for line in numbered:
        assert line in readme_section, line


def test_info_tab_copy_matches_proposal_deck():
    """설계 원칙·탐지 백엔드·LLM 정책 문구가 제안서 문구와 일치한다."""
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception

    principles = next(m.value for m in at.markdown if "**1. AI는 판정하지 않는다.**" in m.value)
    # 원칙 2는 계획서 S01 문구("기존 사진·기존 절차 위에")와 같은 낱말을 쓴다 — 사이드바 캡션도 동일
    assert "**2. 기존 사진·기존 절차 위에.**" in principles
    assert "스캔한 필름 사진 또는 디지털 RT(CR/DR) 사진" in principles
    assert any("② 기존 사진·기존 절차 위에" in c.value for c in at.caption)
    assert "ISO 14096-2" in principles and "조직 단위 전제조건" in principles

    captions = _all_text(at.caption)
    # 탐지 백엔드: 폴백은 학습 모델이 아닌 OpenCV 휴리스틱, YOLO 가중치(YOLO26s 권장)로 자동 전환
    det = next(c.value for c in at.caption if "OpenCV" in c.value and "YOLO26s" in c.value)
    assert "학습된 모델이 아닌" in det and "자동으로 YOLO 백엔드로 전환" in det
    # LLM 정책: 로컬 기본 모델 EXAONE 3.5(exaone3.5:7.8b), 클라우드는 옵트인
    llm = next(c.value for c in at.caption if "EXAONE 3.5" in c.value)
    assert "exaone3.5:7.8b" in llm and "옵트인" in llm and "RTWB_ALLOW_CLOUD_LLM=1" in llm
    assert "이 프로토타입은 무엇이고 무엇이 아닌가" in captions

    # 정책 표(st.table)에도 옵트인·기본 모델 표기가 실린다
    tables = "\n".join(t.value.to_string() for t in at.table)
    assert "옵트인" in tables
    assert "EXAONE 3.5" in tables  # 기본 모델(exaone3.5:7.8b)일 때 주석 표기


# ─────────────── 초안 생성 이후 보고 항목 변경 — 낡은 초안 표시 ───────────────


def test_report_item_edit_after_draft_marks_draft_stale_and_clears_payload():
    """초안 생성 뒤 보고 항목을 바꾸면 payload·생성 경로는 비워지고, 본문은 남되 '초안을 다시 생성' 경고가 뜬다.

    (이전에는 PDF만 버려서, 소견서 본문 4절의 '[촬영 조건]' 값과 PDF 5절 표의 값이 어긋난 채
    승인·아카이브될 수 있었다.)
    """
    from rtworkbench.models import DefectCandidate, Measurement

    at = _boot_with_image()
    at.session_state["wb_candidates"] = [
        DefectCandidate(id="p1", defect_type="porosity", bbox=(10, 10, 30, 30),
                        confidence=1.0, source="human", status="accepted"),
    ]
    at.session_state["wb_measurements"] = {
        "p1": Measurement(defect_id="p1", p1=(0.0, 0.0), p2=(0.0, 20.0),
                          length_px=20.0, length_mm=2.0),
    }
    at.session_state["wb_scale_mm_per_px"] = 0.1
    at.text_input(key="wb_rpt_sfd_mm").set_value("700")
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    assert at.session_state["wb_overall"] is True
    _button(at, "wb_btn_report").click()
    at.run(timeout=60)
    assert not at.exception
    assert "700 mm" in at.session_state["wb_report_payload"]
    assert "700 mm" in at.session_state["wb_report_text"]
    assert at.session_state["wb_report_source"]
    assert at.session_state["wb_report_stale"] is False
    assert not any("초안 생성 이후" in w.value for w in at.warning)

    # 보고 항목 수정 → 판정은 유지, payload·경로는 비움, 본문은 남김 + 경고
    at.text_input(key="wb_rpt_sfd_mm").set_value("800")
    at.run(timeout=60)
    assert not at.exception
    assert at.session_state["wb_overall"] is True
    assert at.session_state["wb_report_payload"] == ""
    assert at.session_state["wb_report_source"] == ""
    assert at.session_state["wb_pdf_bytes"] is None
    assert "700 mm" in at.session_state["wb_report_text"]  # 편집 중일 수 있는 본문은 보존
    assert at.session_state["wb_report_stale"] is True
    stale = [w.value for w in at.warning if "초안 생성 이후" in w.value]
    assert stale and "초안 생성" in stale[0] and "보고 항목" in stale[0]

    # 다시 초안 생성 → 새 값 반영, 경고 사라짐
    _button(at, "wb_btn_report").click()
    at.run(timeout=60)
    assert not at.exception
    assert "800 mm" in at.session_state["wb_report_payload"]
    assert "800 mm" in at.session_state["wb_report_text"] and "700 mm" not in at.session_state["wb_report_text"]
    assert at.session_state["wb_report_stale"] is False
    assert not any("초안 생성 이후" in w.value for w in at.warning)


def test_judgment_input_change_after_draft_marks_draft_stale():
    """초안 생성 뒤 판정 입력(두께)이 바뀌면 판정은 무효화되고, 남은 본문에는 낡음 경고가 붙는다.

    초안이 없던 상태(직접 쓴 본문)에서는 경고가 붙지 않는다.
    """
    from rtworkbench.models import DefectCandidate, Measurement

    at = _boot_with_image()
    at.session_state["wb_report_text"] = "직접 작성한 소견"  # 초안 없이 손으로 쓴 본문
    at.session_state["wb_candidates"] = [
        DefectCandidate(id="p1", defect_type="porosity", bbox=(10, 10, 30, 30),
                        confidence=1.0, source="human", status="accepted"),
    ]
    at.session_state["wb_measurements"] = {
        "p1": Measurement(defect_id="p1", p1=(0.0, 0.0), p2=(0.0, 20.0),
                          length_px=20.0, length_mm=2.0),
    }
    at.session_state["wb_scale_mm_per_px"] = 0.1
    at.run(timeout=60)
    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    at.number_input(key="wb_ctx_thickness").set_value(20.0)
    at.run(timeout=60)
    assert at.session_state["wb_overall"] is None
    assert at.session_state["wb_report_stale"] is False  # 생성한 초안이 없었으므로 경고 없음
    assert not any("초안 생성 이후" in w.value for w in at.warning)

    _button(at, "wb_btn_judge").click()
    at.run(timeout=60)
    _button(at, "wb_btn_report").click()
    at.run(timeout=60)
    assert at.session_state["wb_report_payload"]
    at.number_input(key="wb_ctx_thickness").set_value(25.0)
    at.run(timeout=60)
    assert at.session_state["wb_overall"] is None
    assert at.session_state["wb_report_stale"] is True
    assert any("초안 생성 이후" in w.value for w in at.warning)

    # 새 이미지 로드 → 본문과 함께 낡음 표시도 사라진다
    at.session_state["wb_image_key"] = "test:another"
    at.session_state["wb_image_name"] = "another.png"
    at.run(timeout=60)
