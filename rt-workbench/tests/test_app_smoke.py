"""app.py 스모크 테스트 — Streamlit AppTest로 예외 없이 부팅되는지 확인.

주의: 커스텀 컴포넌트(streamlit-image-coordinates)는 AppTest에서 None을 반환하므로
앱 스크립트는 None-안전해야 한다.
"""

from __future__ import annotations

from pathlib import Path

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
