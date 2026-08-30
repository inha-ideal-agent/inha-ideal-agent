"""RT 판독 워크벤치 — Streamlit UI.

"판독은 자격자가, 서류는 AI가." RT 판독원 1인의 책상 위 소프트웨어.

3대 설계 원칙:
  1. AI는 판정하지 않는다 — 합부는 결정론적 룰 엔진 + 판독원, LLM은 설명/작문 전담.
  2. 새 행동 요구 0 — 입력은 기존 필름 스캔 이미지 그대로.
  3. 탐지는 부품 — recall(민감도) 우선, 오탐은 클릭 한 번·미탐은 배에 남는다.

실행: streamlit run app.py  (프로젝트 루트에서)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

from rtworkbench import __version__, config, detection, measure, preprocess, rules
from rtworkbench import db as archive_db
from rtworkbench.models import (
    DEFECT_TYPES,
    QUALITY_LEVELS,
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    Measurement,
    new_id,
)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

DISPLAY_W = 980  # 이미지 표시 폭(px)

TYPE_KEYS: list[str] = list(DEFECT_TYPES.keys())

STATUS_KO = {"proposed": "제안", "accepted": "채택", "rejected": "기각"}
SOURCE_KO = {"ai": "AI", "human": "판독원"}

JOINT_TYPES = ["맞대기(Butt)", "필릿(Fillet)", "T형(Tee)", "겹치기(Lap)", "모서리(Corner)"]

CLICK_MODES = ("없음", "캘리브레이션", "결함 측정", "후보 추가")

# 오버레이 색상 (RGB)
COLOR_PROPOSED = (255, 150, 0)  # 주황 — AI 제안
COLOR_ACCEPTED = (255, 45, 45)  # 빨강 — 채택
COLOR_REJECTED = (150, 150, 150)  # 회색 점선 — 기각
COLOR_HUMAN = (60, 120, 255)  # 파랑 — 판독원 직접 추가
COLOR_MEASURE = (0, 220, 120)  # 초록 — 측정선
COLOR_CLICK = (255, 0, 255)  # 마젠타 — 진행 중 클릭 십자

PRINCIPLES_SHORT = "원칙: ① AI는 판정하지 않는다 ② 새 행동 요구 0 ③ 탐지는 부품(recall 우선)"

PRINCIPLES_FULL = """
**1. AI는 판정하지 않는다.**
합격/불합격은 결정론적 룰 엔진(기준표 JSON)과 자격 판독원이 결정한다.
LLM은 이미 확정된 판정 결과를 공식 문체로 정리하는 작문 보조일 뿐이다.

**2. 새 행동 요구 0.**
현장에 새 장비·새 촬영 절차를 요구하지 않는다. 입력은 지금도 존재하는
필름 스캔 이미지 파일 그대로다.

**3. 탐지는 부품이다.**
결함 후보 탐지기는 교체 가능한 부품이며 recall(민감도)을 우선한다.
오탐(false positive)은 판독원의 클릭 한 번으로 기각되지만,
미탐(false negative)은 배에 남는다.
"""


# ---------------------------------------------------------------------------
# 세션 상태 관리
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict = {
    "wb_image": None,  # 원본 그레이스케일 ndarray (H,W)
    "wb_image_name": "",
    "wb_image_key": "",  # 이미지 교체 감지용 식별자
    "wb_sample_meta": None,  # 샘플 .json 메타 (있을 때만)
    "wb_candidates": [],  # list[DefectCandidate]
    "wb_measurements": {},  # dict[defect_id, Measurement]
    "wb_scale_mm_per_px": None,
    "wb_scale_ref": "",
    "wb_verdicts": [],  # list[RuleVerdict]
    "wb_overall": None,  # bool | None — 판정 실행 전 None
    "wb_report_text": "",
    "wb_report_source": "",
    "wb_report_payload": "",
    "wb_pdf_bytes": None,
    "wb_record_id": "",
    "wb_archived": False,
    "wb_click_points": [],  # 진행 중 클릭 좌표 (원본 픽셀 기준)
    "wb_click_last_time": None,  # 직전 처리한 클릭 unix_time (재클릭/중복 방지)
    "wb_det_status": None,  # 탐지 백엔드 상태 캐시
}


def init_state() -> None:
    """세션 상태 기본값을 1회 초기화한다."""
    for key, val in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_for_new_image() -> None:
    """새 이미지 로드 시 하위 상태 전부 리셋.

    wb_click_last_time 은 유지 — 컴포넌트에 남은 직전 클릭 이벤트가
    새 이미지에서 재처리되는 것을 막는다.
    """
    st.session_state["wb_candidates"] = []
    st.session_state["wb_measurements"] = {}
    st.session_state["wb_scale_mm_per_px"] = None
    st.session_state["wb_scale_ref"] = ""
    st.session_state["wb_sample_meta"] = None
    st.session_state["wb_click_points"] = []
    st.session_state["wb_record_id"] = ""
    st.session_state["wb_archived"] = False
    # 이전 필름의 소견서 본문이 새 검사 기록에 저장되지 않도록 여기서만 비운다
    # (invalidate_judgment는 같은 이미지 내 재판정 시 편집 보호를 위해 남긴다).
    st.session_state["wb_report_text"] = ""
    invalidate_judgment()


def invalidate_judgment() -> None:
    """후보/측정이 바뀌면 판정·소견서·PDF 산출물을 무효화한다.

    소견서 본문(wb_report_text)은 판독원이 편집 중일 수 있으므로 지우지 않는다.
    """
    st.session_state["wb_verdicts"] = []
    st.session_state["wb_overall"] = None
    st.session_state["wb_report_source"] = ""
    st.session_state["wb_report_payload"] = ""
    st.session_state["wb_pdf_bytes"] = None


def on_click_mode_change() -> None:
    """클릭 모드 전환 시 진행 중이던 클릭 점을 버린다."""
    st.session_state["wb_click_points"] = []


# ---------------------------------------------------------------------------
# 그리기 유틸
# ---------------------------------------------------------------------------


def _draw_dashed_rect(
    img: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash: int = 8,
) -> None:
    """점선 사각형 (기각 후보 표시용)."""
    x1, y1 = p1
    x2, y2 = p2

    def _dashed_line(a: tuple[int, int], b: tuple[int, int]) -> None:
        dist = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        if dist <= 0:
            return
        n = max(1, dist // dash)
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1.0)
            q1 = (int(a[0] + (b[0] - a[0]) * t0), int(a[1] + (b[1] - a[1]) * t0))
            q2 = (int(a[0] + (b[0] - a[0]) * t1), int(a[1] + (b[1] - a[1]) * t1))
            cv2.line(img, q1, q2, color, thickness)

    _dashed_line((x1, y1), (x2, y1))
    _dashed_line((x2, y1), (x2, y2))
    _dashed_line((x2, y2), (x1, y2))
    _dashed_line((x1, y2), (x1, y1))


def _candidate_color(c: DefectCandidate) -> tuple[int, int, int]:
    if c.source == "human":
        return COLOR_HUMAN
    if c.status == "accepted":
        return COLOR_ACCEPTED
    if c.status == "rejected":
        return COLOR_REJECTED
    return COLOR_PROPOSED


def build_overlay_rgb(
    gray: np.ndarray,
    *,
    use_clahe: bool,
    clip_limit: float,
    show_candidates: bool,
) -> np.ndarray:
    """표시용 RGB 이미지 생성: CLAHE(옵션) + 후보 박스 + 측정선 + 진행 중 클릭 십자."""
    base = preprocess.apply_clahe(gray, clip_limit=clip_limit) if use_clahe else gray
    rgb = preprocess.to_display_rgb(base)
    h, w = gray.shape[:2]
    lt = max(2, int(round(w / DISPLAY_W * 2)))  # 표시 축소를 고려한 선 두께
    fs = max(0.45, w / DISPLAY_W * 0.55)  # 폰트 스케일

    # 후보 박스 + ID
    if show_candidates:
        for c in st.session_state["wb_candidates"]:
            color = _candidate_color(c)
            x1, y1, x2, y2 = (int(round(v)) for v in c.bbox)
            if c.status == "rejected":
                _draw_dashed_rect(rgb, (x1, y1), (x2, y2), color, max(1, lt - 1))
            else:
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, lt)
            label = c.id.split("-")[-1]  # 짧은 hex 부분만 표기
            ty = y1 - 6 if y1 - 6 > 12 else y2 + int(18 * fs)
            cv2.putText(rgb, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, lt)

    # 확정된 측정선
    for m in st.session_state["wb_measurements"].values():
        q1 = (int(round(m.p1[0])), int(round(m.p1[1])))
        q2 = (int(round(m.p2[0])), int(round(m.p2[1])))
        cv2.line(rgb, q1, q2, COLOR_MEASURE, lt)
        txt = f"{m.length_mm:.2f}mm" if m.length_mm is not None else f"{m.length_px:.0f}px"
        cv2.putText(
            rgb, txt, (min(q1[0], q2[0]), min(q1[1], q2[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, fs, COLOR_MEASURE, lt,
        )

    # 진행 중 클릭 점 (십자 마커)
    for px, py in st.session_state["wb_click_points"]:
        cv2.drawMarker(
            rgb, (int(round(px)), int(round(py))), COLOR_CLICK,
            markerType=cv2.MARKER_CROSS, markerSize=int(14 * fs / 0.5), thickness=lt,
        )
    return rgb


# ---------------------------------------------------------------------------
# 클릭 처리 (2클릭 상태머신)
# ---------------------------------------------------------------------------


def _finish_calibration(p1, p2) -> None:
    known_mm = float(st.session_state.get("wb_calib_mm", 10.0))
    try:
        scale = measure.calibrate(p1, p2, known_mm)
    except ValueError as exc:
        st.session_state["wb_click_points"] = []
        st.error(f"캘리브레이션 실패: {exc}")
        return
    st.session_state["wb_scale_mm_per_px"] = scale
    st.session_state["wb_scale_ref"] = f"기준물 {known_mm:g}mm 2클릭 (납마커 등)"
    st.session_state["wb_click_points"] = []
    refresh_measurement_mm()  # 기존 측정의 mm를 새 스케일로 재계산 (재캘리브레이션 포함)
    invalidate_judgment()  # 측정 mm가 생기거나 바뀌면 판정 입력이 바뀐 것
    st.rerun()


def _finish_measurement(p1, p2) -> None:
    target = st.session_state.get("wb_measure_target")
    st.session_state["wb_click_points"] = []
    if not target:
        st.error("측정 대상 후보를 먼저 선택하세요 (채택된 후보만 측정할 수 있습니다).")
        return
    try:
        length_px, length_mm = measure.measure(p1, p2, st.session_state["wb_scale_mm_per_px"])
    except Exception as exc:
        st.error(f"측정 계산 실패: {exc}")
        return
    m = Measurement(defect_id=target, p1=p1, p2=p2, length_px=length_px, length_mm=length_mm)
    st.session_state["wb_measurements"][target] = m  # 같은 defect_id 재측정 시 교체
    invalidate_judgment()
    st.rerun()


def _finish_add_candidate(p1, p2) -> None:
    st.session_state["wb_click_points"] = []
    dtype = st.session_state.get("wb_add_type", "unknown")
    x1, x2 = sorted((p1[0], p2[0]))
    y1, y2 = sorted((p1[1], p2[1]))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        st.error("bbox가 너무 작습니다 — 좌상단과 우하단을 조금 더 벌려 클릭하세요.")
        return
    cand = DefectCandidate(
        id=new_id(),
        defect_type=dtype,
        bbox=(x1, y1, x2, y2),
        confidence=1.0,
        source="human",
        status="accepted",  # 판독원이 직접 본 것 — 즉시 채택
        note="판독원 직접 추가 (AI 미탐 신호)",
    )
    st.session_state["wb_candidates"].append(cand)
    invalidate_judgment()
    st.rerun()


def handle_click(value: dict | None, orig_w: int, orig_h: int, mode: str) -> None:
    """streamlit_image_coordinates 반환값 처리.

    같은 지점 재클릭은 새 이벤트로 안 올 수 있으므로 unix_time으로 신규 클릭을
    판별한다(직전 처리한 unix_time을 session_state에 기억, 다르면 신규).
    """
    if not value:
        return
    t = value.get("unix_time")
    if t is None or t == st.session_state["wb_click_last_time"]:
        return  # 이미 처리한 이벤트
    st.session_state["wb_click_last_time"] = t

    if mode == "없음":
        return
    disp_w = value.get("width") or DISPLAY_W
    disp_h = value.get("height") or 1
    # 표시 좌표 → 원본 픽셀 좌표
    ox = float(value.get("x", 0)) * orig_w / disp_w
    oy = float(value.get("y", 0)) * orig_h / disp_h
    ox = min(max(ox, 0.0), float(orig_w - 1))
    oy = min(max(oy, 0.0), float(orig_h - 1))

    pts = st.session_state["wb_click_points"]
    pts.append((ox, oy))
    if len(pts) < 2:
        st.rerun()  # 첫 클릭 십자 마커 즉시 표시
        return

    p1, p2 = pts[0], pts[1]
    if mode == "캘리브레이션":
        _finish_calibration(p1, p2)
    elif mode == "결함 측정":
        _finish_measurement(p1, p2)
    elif mode == "후보 추가":
        _finish_add_candidate(p1, p2)
    else:
        st.session_state["wb_click_points"] = []


# ---------------------------------------------------------------------------
# 데이터 조립 유틸
# ---------------------------------------------------------------------------


def get_detector_status() -> dict:
    """탐지 백엔드 상태 (세션 1회 캐시 — import 재시도 비용 절약)."""
    if st.session_state.get("wb_det_status") is None:
        try:
            st.session_state["wb_det_status"] = detection.detector_status()
        except Exception as exc:
            st.session_state["wb_det_status"] = {
                "backend": "?", "yolo_available": False, "weights_path": "",
                "note": f"상태 조회 실패: {exc}",
            }
    return st.session_state["wb_det_status"]


def build_context_from_sidebar() -> InspectionContext:
    """사이드바 위젯 값 + 세션의 스케일로 InspectionContext 조립."""
    return InspectionContext(
        film_id=st.session_state.get("wb_ctx_film_id", ""),
        block=st.session_state.get("wb_ctx_block", ""),
        weld_id=st.session_state.get("wb_ctx_weld_id", ""),
        joint_type=st.session_state.get("wb_ctx_joint", JOINT_TYPES[0]),
        thickness_mm=float(st.session_state.get("wb_ctx_thickness", 12.0)),
        quality_level=st.session_state.get("wb_ctx_quality", "B"),
        inspector=st.session_state.get("wb_ctx_inspector", ""),
        scale_mm_per_px=st.session_state.get("wb_scale_mm_per_px"),
        scale_ref=st.session_state.get("wb_scale_ref", ""),
    )


def refresh_measurement_mm() -> None:
    """모든 측정의 mm를 현재 스케일로 (재)계산 — 소급 적용·재캘리브레이션 모두 커버.

    length_px가 원천 데이터이므로 무조건 재계산해도 항상 안전하다.
    (조건부 갱신이면 재캘리브레이션 시 옛 스케일의 mm가 고정되어 버린다.)
    """
    scale = st.session_state["wb_scale_mm_per_px"]
    if scale is None:
        return
    for m in st.session_state["wb_measurements"].values():
        m.length_mm = measure.px_to_mm(m.length_px, scale)


def assemble_record(report_text: str) -> InspectionRecord:
    """승인/PDF용 InspectionRecord 조립 (저장은 호출부에서)."""
    gray = st.session_state["wb_image"]
    if not st.session_state["wb_record_id"]:
        st.session_state["wb_record_id"] = new_id("rec")
    ctx = build_context_from_sidebar()
    source = st.session_state["wb_report_source"] or "template"
    return InspectionRecord(
        record_id=st.session_state["wb_record_id"],
        context=ctx,
        candidates=list(st.session_state["wb_candidates"]),
        measurements=list(st.session_state["wb_measurements"].values()),
        verdicts=list(st.session_state["wb_verdicts"]),
        overall_passed=bool(st.session_state["wb_overall"]),
        report_text=report_text,
        report_source=source,
        image_name=st.session_state["wb_image_name"],
        image_size=(int(gray.shape[1]), int(gray.shape[0])),
    )


def try_build_pdf(record: InspectionRecord) -> bytes | None:
    """PDF 생성 — 실패 시 st.error 후 None (report_pdf는 지연 import)."""
    try:
        from rtworkbench import report_pdf

        return report_pdf.build_pdf(record)
    except Exception as exc:
        st.error(f"PDF 생성에 실패했습니다: {exc}")
        return None


def verdicts_to_df(verdicts) -> pd.DataFrame:
    """RuleVerdict 목록 → 표시용 DataFrame."""
    rows = []
    for v in verdicts:
        rows.append(
            {
                "결함 ID": v.defect_id,
                "유형": DEFECT_TYPES.get(v.defect_type, v.defect_type),
                "크기(mm)": v.size_mm,
                "허용 한계(mm)": "허용 불가" if v.limit_mm is None else v.limit_mm,
                "근거 조항": v.clause,
                "합부": "✅ 합격" if v.passed else "❌ 불합격",
                "판정 근거": v.detail,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 탭 1 — 판독 워크벤치
# ---------------------------------------------------------------------------


def _render_image_input() -> None:
    """이미지 입력(업로드/샘플) + 새 이미지 감지 시 상태 리셋."""
    st.subheader("1️⃣ 이미지 입력")
    src_mode = st.radio(
        "이미지 소스", ["파일 업로드", "샘플 이미지"], horizontal=True, key="wb_src_mode"
    )

    new_key = ""
    loader = None  # () -> (gray, name, meta)

    if src_mode == "파일 업로드":
        up = st.file_uploader(
            "필름 스캔 이미지 (PNG/JPG)", type=["png", "jpg", "jpeg"], key="wb_uploader"
        )
        if up is not None:
            new_key = f"upload:{up.name}:{up.size}"

            def loader(up=up):
                return preprocess.load_grayscale(up.getvalue()), up.name, None

    else:
        try:
            files = sorted(
                p for p in config.SAMPLES_DIR.glob("*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
        except Exception:
            files = []
        if not files:
            st.info(f"샘플 이미지가 없습니다: {config.SAMPLES_DIR}")
        else:
            sel = st.selectbox(
                "샘플 선택", files, format_func=lambda p: p.name, key="wb_sample_sel"
            )
            if sel is not None:
                new_key = f"sample:{sel}"

                def loader(sel=sel):
                    meta = None
                    meta_path = sel.with_suffix(".json")
                    if meta_path.is_file():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        except Exception:
                            meta = None  # 메타 파손은 무시 — 이미지는 정상 로드
                    return preprocess.load_grayscale(sel), sel.name, meta

    # 새 이미지 로드 → 하위 상태 리셋
    if new_key and new_key != st.session_state["wb_image_key"] and loader is not None:
        try:
            gray, name, meta = loader()
        except Exception as exc:
            st.error(f"이미지를 불러오지 못했습니다: {exc}")
            return
        reset_for_new_image()
        st.session_state["wb_image"] = gray
        st.session_state["wb_image_name"] = name
        st.session_state["wb_image_key"] = new_key
        st.session_state["wb_sample_meta"] = meta

    # 샘플 메타 스케일 자동 적용
    meta = st.session_state["wb_sample_meta"]
    if meta and isinstance(meta.get("mm_per_px"), (int, float)):
        cols = st.columns([3, 2])
        cols[0].caption(
            f"샘플 메타 발견: mm_per_px = {meta['mm_per_px']} "
            f"(기준물 {meta.get('marker', {}).get('length_mm', '?')}mm)"
        )
        if cols[1].button("샘플 메타에서 스케일 자동 적용", key="wb_btn_meta_scale"):
            st.session_state["wb_scale_mm_per_px"] = float(meta["mm_per_px"])
            st.session_state["wb_scale_ref"] = "샘플 메타(mm_per_px) 자동 적용"
            refresh_measurement_mm()
            invalidate_judgment()  # 측정 mm가 생기거나 바뀌면 판정 입력이 바뀐 것
            st.rerun()


def _render_detection_button(gray: np.ndarray) -> None:
    """[2차 눈] AI 후보 탐지 실행."""
    st.subheader("2️⃣ 2차 눈 — AI 후보 탐지")
    status = get_detector_status()
    c1, c2 = st.columns([1, 3])
    run = c1.button("AI 후보 탐지 실행", type="primary", key="wb_btn_detect")
    c2.caption(f"백엔드: `{status.get('backend')}` — {status.get('note', '')}")
    if run:
        try:
            detector = detection.get_detector()
            new_cands = detector.detect(gray)
        except Exception as exc:
            st.error(f"후보 탐지에 실패했습니다: {exc}")
            return
        # 판독원이 이미 결론 낸 후보(채택/기각)와 직접 추가분은 보존,
        # 이전 실행의 '제안' 상태 AI 후보만 새 결과로 교체한다.
        kept = [
            c for c in st.session_state["wb_candidates"]
            if c.source == "human" or c.status != "proposed"
        ]
        st.session_state["wb_candidates"] = kept + new_cands
        st.session_state["wb_click_points"] = []  # 진행 중 클릭 점 폐기
        invalidate_judgment()
        # st.rerun() 금지 — 하위 섹션은 같은 실행에서 새 후보를 반영하며,
        # rerun 하면 이 안내문이 사용자에게 표시되기 전에 사라진다.
        st.success(
            f"AI가 결함 후보 {len(new_cands)}건을 제안했습니다 — "
            "recall 우선이므로 오탐이 있을 수 있습니다. 오탐은 [기각] 클릭 한 번."
        )


def _render_click_controls(accepted: list[DefectCandidate]) -> str:
    """클릭 모드 라디오 + 모드별 부속 입력. 현재 모드 문자열 반환."""
    st.subheader("3️⃣ 클릭 도구 — 캘리브레이션 · 측정 · 후보 추가")
    mode = st.radio(
        "클릭 모드",
        CLICK_MODES,
        horizontal=True,
        key="wb_click_mode",
        on_change=on_click_mode_change,
    )

    if mode == "캘리브레이션":
        st.number_input(
            "기준물 실제 길이 (mm) — 예: 납마커 10mm",
            min_value=0.1, value=10.0, step=0.5, key="wb_calib_mm",
        )
        st.caption("이미지에서 기준물 양 끝을 순서대로 2클릭하면 mm/px 스케일이 확정됩니다.")
    elif mode == "결함 측정":
        if not accepted:
            st.info("채택(accepted)된 후보가 없습니다 — 후보를 채택하거나 직접 추가하세요.")
        else:
            st.selectbox(
                "측정 대상 후보 (채택된 후보만)",
                [c.id for c in accepted],
                format_func=lambda cid: next(
                    (f"{c.id} · {c.type_ko}" for c in accepted if c.id == cid), cid
                ),
                key="wb_measure_target",
                on_change=on_click_mode_change,  # 대상 변경 시 진행 중 클릭 점 폐기
            )
            st.caption("결함 지시의 최대 치수 양 끝을 2클릭 — 같은 후보 재측정 시 교체됩니다.")
        if st.session_state["wb_scale_mm_per_px"] is None:
            st.warning("스케일 미확정 — 측정은 px로만 기록되며 mm 환산은 캘리브레이션 후 소급 적용됩니다.")
    elif mode == "후보 추가":
        with st.expander("수동 후보 추가 (AI가 놓친 지시 = 미탐 신호)", expanded=True):
            st.selectbox(
                "결함 유형",
                TYPE_KEYS,
                format_func=lambda k: f"{DEFECT_TYPES[k]} ({k})",
                key="wb_add_type",
            )
            st.caption(
                "이미지에서 bbox 좌상단 → 우하단 순서로 2클릭하면 "
                "source='human', status='accepted', confidence=1.0 후보가 추가됩니다."
            )

    n_pts = len(st.session_state["wb_click_points"])
    if mode != "없음" and n_pts == 1:
        st.info("1번째 점 기록됨 — 2번째 점을 클릭하세요.")
    return mode


def _render_image_and_clicks(gray: np.ndarray, mode: str) -> None:
    """보기 옵션 + 이미지 표시(클릭 컴포넌트) + 클릭 이벤트 처리."""
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        use_clahe = c1.toggle("CLAHE 대비 향상", value=True, key="wb_clahe_on")
        clip = c2.slider(
            "CLAHE clip_limit", min_value=1.0, max_value=8.0, value=3.0, step=0.5,
            key="wb_clahe_clip", disabled=not use_clahe,
        )
        show_cands = c3.toggle("AI 후보 오버레이", value=True, key="wb_overlay_on")

    try:
        rgb = build_overlay_rgb(
            gray, use_clahe=use_clahe, clip_limit=clip, show_candidates=show_cands
        )
    except Exception as exc:
        st.error(f"이미지 렌더링에 실패했습니다: {exc}")
        return

    value = streamlit_image_coordinates(rgb, width=DISPLAY_W, key="wb_imgclick")
    handle_click(value, orig_w=gray.shape[1], orig_h=gray.shape[0], mode=mode)

    # 스케일 상태 표시
    scale = st.session_state["wb_scale_mm_per_px"]
    if scale is not None:
        st.caption(f"📏 스케일 확정: {scale:.4f} mm/px (기준: {st.session_state['wb_scale_ref']})")
    else:
        st.caption("📏 스케일 미확정 — 캘리브레이션 모드에서 기준물을 2클릭하세요.")


def _render_candidate_table() -> None:
    """후보 목록 — 각 행에 유형 수정 selectbox + [채택][기각] 버튼."""
    cands: list[DefectCandidate] = st.session_state["wb_candidates"]
    st.subheader("4️⃣ 결함 후보 목록")
    if not cands:
        st.info("후보가 없습니다 — AI 탐지를 실행하거나 '후보 추가' 모드로 직접 추가하세요.")
        return

    st.caption("오탐은 클릭 한 번 — [기각]. 유형은 selectbox로 바로 수정할 수 있습니다.")
    header = st.columns([2, 3, 1.2, 1.2, 1.2, 1, 1])
    for col, name in zip(header, ["ID", "유형", "신뢰도", "출처", "상태", "", ""]):
        col.markdown(f"**{name}**")

    for c in cands:
        row = st.columns([2, 3, 1.2, 1.2, 1.2, 1, 1])
        row[0].code(c.id, language=None)
        idx = TYPE_KEYS.index(c.defect_type) if c.defect_type in TYPE_KEYS else TYPE_KEYS.index("unknown")
        sel = row[1].selectbox(
            "유형", TYPE_KEYS, index=idx,
            format_func=lambda k: f"{DEFECT_TYPES[k]} ({k})",
            key=f"wb_ctype_{c.id}", label_visibility="collapsed",
        )
        if sel != c.defect_type:
            c.defect_type = sel
            invalidate_judgment()
        row[2].write(f"{c.confidence:.2f}")
        row[3].write(SOURCE_KO.get(c.source, c.source))
        status_ko = STATUS_KO.get(c.status, c.status)
        badge = {"accepted": "🔴", "rejected": "⚪", "proposed": "🟠"}.get(c.status, "")
        row[4].write(f"{badge} {status_ko}")
        if row[5].button("채택", key=f"wb_acc_{c.id}", disabled=(c.status == "accepted")):
            c.status = "accepted"
            st.session_state["wb_click_points"] = []  # 진행 중 클릭 점 폐기
            invalidate_judgment()
            st.rerun()
        if row[6].button("기각", key=f"wb_rej_{c.id}", disabled=(c.status == "rejected")):
            c.status = "rejected"
            st.session_state["wb_click_points"] = []  # 진행 중 클릭 점 폐기
            invalidate_judgment()
            st.rerun()

    # 측정 현황 요약
    meas: dict[str, Measurement] = st.session_state["wb_measurements"]
    accepted = [c for c in cands if c.status == "accepted"]
    if accepted:
        rows = []
        for c in accepted:
            m = meas.get(c.id)
            rows.append(
                {
                    "결함 ID": c.id,
                    "유형": c.type_ko,
                    "측정(px)": round(m.length_px, 1) if m else None,
                    "측정(mm)": round(m.length_mm, 2) if (m and m.length_mm is not None) else None,
                    "측정 상태": "완료" if (m and m.length_mm is not None)
                    else ("px만 (스케일 미확정)" if m else "미측정"),
                }
            )
        st.markdown("**채택 후보 측정 현황**")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_judgment(ctx: InspectionContext) -> None:
    """[룰 판정] 결정론적 룰 엔진 실행 + verdict 표 + 종합 배너."""
    st.subheader("5️⃣ 룰 판정 — 결정론적 룰 엔진")
    st.caption("⚖️ AI는 판정하지 않습니다 — 판정: 룰 엔진 + 판독원")

    refresh_measurement_mm()
    cands: list[DefectCandidate] = st.session_state["wb_candidates"]
    meas: dict[str, Measurement] = st.session_state["wb_measurements"]
    accepted = [c for c in cands if c.status == "accepted"]

    if st.button("판정 실행", type="primary", key="wb_btn_judge"):
        items: list[tuple[str, str, float]] = []
        unmeasured: list[str] = []
        for c in accepted:
            m = meas.get(c.id)
            if m is not None and m.length_mm is not None:
                items.append((c.id, c.defect_type, m.length_mm))
            else:
                unmeasured.append(f"{c.id}({c.type_ko})")
        if unmeasured:
            # fail-safe: 미측정 채택 결함이 있으면 종합 판정을 확정하지 않는다 —
            # 제외하고 진행하면 허용불가 유형(crack 등)조차 룰 엔진에 전달되지 않아
            # 빈 verdicts로 '합격' false PASS가 승인·아카이브될 수 있다.
            invalidate_judgment()
            st.error(
                "측정되지 않은 채택 결함이 있어 판정을 확정할 수 없습니다: "
                + ", ".join(unmeasured)
                + " — '결함 측정' 모드에서 2클릭으로 크기를 확정한 뒤 다시 판정하세요."
            )
            return
        try:
            engine = rules.RuleEngine()
            verdicts = engine.evaluate_all(items, ctx.thickness_mm, ctx.quality_level)
        except Exception as exc:
            st.error(f"룰 판정에 실패했습니다: {exc}")
            return
        st.session_state["wb_verdicts"] = verdicts
        st.session_state["wb_overall"] = rules.overall_pass(verdicts)
        st.session_state["wb_pdf_bytes"] = None  # 판정이 바뀌면 이전 PDF 무효

    verdicts = st.session_state["wb_verdicts"]
    overall = st.session_state["wb_overall"]
    if overall is not None:
        if verdicts:
            st.dataframe(verdicts_to_df(verdicts), width="stretch", hide_index=True)
        else:
            st.caption("판정 대상 결함 없음 (채택+측정된 결함 0건).")
        if overall:
            st.success("✅ 종합 판정: 합격 — 모든 결함 지시가 허용 한계 이내입니다.")
        else:
            st.error("❌ 종합 판정: 불합격 — 허용 한계를 벗어난 결함 지시가 있습니다.")


def _render_report(ctx: InspectionContext) -> None:
    """[소견서] LLM 초안 생성 + 편집."""
    st.subheader("6️⃣ 소견서 초안 — LLM 작문 보조")
    st.caption("🔒 비식별 텍스트만 전송 — 이미지·파일경로·개인식별 정보는 LLM에 보내지 않습니다.")

    if st.session_state["wb_overall"] is None:
        st.info("먼저 [판정 실행]으로 합부를 확정하세요 — 소견서는 확정된 판정 결과만 서술합니다.")
    else:
        if st.button("초안 생성", key="wb_btn_report"):
            try:
                from rtworkbench import report_llm  # 지연 import — 병렬 구현 모듈

                payload = report_llm.build_payload(
                    ctx,
                    st.session_state["wb_verdicts"],
                    list(st.session_state["wb_measurements"].values()),
                )
                text, source = report_llm.ReportWriter().generate(payload)
            except Exception as exc:
                st.error(f"소견서 초안 생성에 실패했습니다: {exc}")
            else:
                st.session_state["wb_report_payload"] = payload
                st.session_state["wb_report_source"] = source
                st.session_state["wb_report_text"] = text
                st.session_state["wb_pdf_bytes"] = None

    source = st.session_state["wb_report_source"]
    if source:
        badge = {"claude": "🟣 Claude", "gemini": "🔵 Gemini",
                 "template": "⚙️ 오프라인 템플릿", "cache": "💾 캐시"}.get(source, source)
        st.markdown(f"초안 생성 경로: **{badge}** (`{source}`)")
    if st.session_state["wb_report_payload"]:
        with st.expander("전송 payload 미리보기 (LLM에 전달된 비식별 텍스트 전문)"):
            st.code(st.session_state["wb_report_payload"], language=None)

    st.text_area(
        "소견서 본문 (판독원이 자유롭게 수정 — 승인 시 이 내용이 저장됩니다)",
        height=260,
        key="wb_report_text",
    )


def _render_approve() -> None:
    """[승인·보관] 아카이브 저장 + PDF."""
    st.subheader("7️⃣ 승인 · 보관")
    judged = st.session_state["wb_overall"] is not None
    # fail-safe: 미측정 채택 결함이 남아 있으면 판정 상태와 무관하게 승인 불가
    meas = st.session_state["wb_measurements"]
    unmeasured_accepted = [
        c for c in st.session_state["wb_candidates"]
        if c.status == "accepted"
        and (meas.get(c.id) is None or meas.get(c.id).length_mm is None)
    ]
    if unmeasured_accepted:
        judged = False
        st.warning(
            "측정되지 않은 채택 결함이 있어 승인할 수 없습니다 — "
            "'결함 측정' 모드에서 크기를 확정한 뒤 재판정하세요."
        )
    elif not judged:
        st.caption("판정 실행 후 승인할 수 있습니다.")

    report_text = st.session_state.get("wb_report_text", "")
    c1, c2 = st.columns([1, 1])

    if c2.button("PDF 미리 생성", key="wb_btn_pdf_preview", disabled=not judged):
        pdf = try_build_pdf(assemble_record(report_text))
        if pdf:
            st.session_state["wb_pdf_bytes"] = pdf

    if c1.button("승인 및 아카이브 저장", type="primary", key="wb_btn_approve", disabled=not judged):
        if not report_text.strip():
            st.warning("소견서 본문이 비어 있습니다 — 그대로 저장합니다 (초안 생성 또는 직접 작성 권장).")
        record = assemble_record(report_text)
        try:
            archive_db.Archive().save(record)
        except Exception as exc:
            st.error(f"아카이브 저장에 실패했습니다: {exc}")
        else:
            st.session_state["wb_archived"] = True
            st.success(
                f"승인 완료 — 아카이브에 저장되었습니다 (record_id: {record.record_id}). "
                "이 승인 기록은 자기개선 루프의 학습 라벨이 됩니다."
            )
            pdf = try_build_pdf(record)
            if pdf:
                st.session_state["wb_pdf_bytes"] = pdf

    pdf_bytes = st.session_state["wb_pdf_bytes"]
    if pdf_bytes:
        st.download_button(
            "PDF 다운로드",
            data=pdf_bytes,
            file_name=f"RT판독소견서_{st.session_state['wb_record_id'] or 'draft'}.pdf",
            mime="application/pdf",
            key="wb_btn_pdf_dl",
        )


def render_tab_workbench(ctx: InspectionContext) -> None:
    _render_image_input()
    gray = st.session_state["wb_image"]
    if gray is None:
        st.info("필름 스캔 이미지를 업로드하거나 샘플을 선택하면 판독을 시작합니다.")
        return

    st.caption(
        f"현재 이미지: `{st.session_state['wb_image_name']}` "
        f"({gray.shape[1]}×{gray.shape[0]} px)"
    )
    _render_detection_button(gray)
    st.divider()
    accepted = [c for c in st.session_state["wb_candidates"] if c.status == "accepted"]
    mode = _render_click_controls(accepted)
    _render_image_and_clicks(gray, mode)
    st.divider()
    _render_candidate_table()
    st.divider()
    _render_judgment(ctx)
    st.divider()
    _render_report(ctx)
    st.divider()
    _render_approve()


# ---------------------------------------------------------------------------
# 탭 2 — 아카이브 검색
# ---------------------------------------------------------------------------


def render_tab_archive() -> None:
    st.subheader("아카이브 검색")
    st.caption("🔎 '3번 블록 기공 이력'을 3초 안에 — 승인 기록은 전부 검색 가능한 자산입니다.")

    with st.form("wb_search_form"):
        c1, c2, c3 = st.columns(3)
        f_block = c1.text_input("블록", key="wb_q_block")
        f_weld = c2.text_input("용접부 ID", key="wb_q_weld")
        f_type = c3.selectbox(
            "결함 유형", ["(전체)"] + TYPE_KEYS,
            format_func=lambda k: k if k == "(전체)" else f"{DEFECT_TYPES[k]} ({k})",
            key="wb_q_type",
        )
        c4, c5, c6 = st.columns(3)
        f_pass = c4.selectbox("합부", ["(전체)", "합격", "불합격"], key="wb_q_pass")
        f_from = c5.date_input("시작일", value=None, key="wb_q_from")
        f_to = c6.date_input("종료일", value=None, key="wb_q_to")
        f_text = st.text_input("자유 텍스트 (소견서/필름ID/용접부/블록)", key="wb_q_text")
        st.form_submit_button("검색")  # 제출 = 재실행 트리거 — 검색은 항상 현재 폼 값으로 수행

    try:
        df = archive_db.Archive().search(
            block=f_block or "",
            weld_id=f_weld or "",
            defect_type="" if f_type == "(전체)" else f_type,
            passed=None if f_pass == "(전체)" else (f_pass == "합격"),
            text=f_text or "",
            date_from=f_from.isoformat() if f_from else "",
            date_to=f_to.isoformat() if f_to else "",
        )
    except Exception as exc:
        st.error(f"아카이브 검색에 실패했습니다: {exc}")
        return

    st.caption(f"검색 결과: {len(df)}건")
    if df.empty:
        st.info("조건에 맞는 기록이 없습니다.")
        return
    st.dataframe(df, width="stretch", hide_index=True)

    st.markdown("**상세 보기**")
    rid = st.selectbox("record_id 선택", list(df["record_id"]), key="wb_detail_rid")
    if not rid:
        return
    try:
        rec = archive_db.Archive().get(rid)
    except Exception as exc:
        st.error(f"기록 조회에 실패했습니다: {exc}")
        return
    if rec is None:
        st.warning("기록을 찾을 수 없습니다.")
        return

    ctx = rec.context
    info = pd.DataFrame(
        {
            "항목": ["필름 ID", "블록", "용접부 ID", "이음 종류", "두께(mm)", "품질등급",
                    "판독원", "기법", "스케일(mm/px)", "이미지", "승인 일시", "초안 경로"],
            "값": [ctx.film_id, ctx.block, ctx.weld_id, ctx.joint_type,
                   ctx.thickness_mm, ctx.quality_level, ctx.inspector, ctx.technique,
                   f"{ctx.scale_mm_per_px:.4f}" if ctx.scale_mm_per_px else "미확정",
                   rec.image_name, rec.created_at, rec.report_source],
        }
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        st.table(info)
        if rec.overall_passed:
            st.success("종합 판정: 합격 ✅")
        else:
            st.error("종합 판정: 불합격 ❌")
    with c2:
        if rec.verdicts:
            st.dataframe(verdicts_to_df(rec.verdicts), width="stretch", hide_index=True)
        else:
            st.caption("판정 결함 없음.")
        st.markdown("**소견서 본문**")
        st.text_area("소견서", value=rec.report_text, height=220,
                     key=f"wb_detail_report_{rid}", disabled=True, label_visibility="collapsed")

    pdf = try_build_pdf(rec)
    if pdf:
        st.download_button(
            "PDF 재다운로드", data=pdf,
            file_name=f"RT판독소견서_{rec.record_id}.pdf", mime="application/pdf",
            key=f"wb_detail_pdf_{rid}",
        )


# ---------------------------------------------------------------------------
# 탭 3 — 자기개선 루프
# ---------------------------------------------------------------------------


def render_tab_loop() -> None:
    st.subheader("자기개선 루프")
    st.caption("♻️ 판독원의 승인 기록이 곧 다음 모델의 학습 데이터가 됩니다.")

    try:
        stats = archive_db.Archive().stats()
    except Exception as exc:
        st.error(f"통계 조회에 실패했습니다: {exc}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("총 검사", stats.get("total", 0))
    m2.metric("합격", stats.get("passed", 0))
    m3.metric("불합격", stats.get("failed", 0))
    m4, m5, m6 = st.columns(3)
    m4.metric("AI 제안", stats.get("ai_proposed", 0))
    rate = stats.get("acceptance_rate")
    m5.metric("AI 채택률", f"{rate * 100:.0f}%" if rate is not None else "—")
    m6.metric("사람 추가 (미탐 신호)", stats.get("human_added", 0),
              help="AI가 놓쳐 판독원이 직접 추가한 결함 수 — 탐지기 개선의 최우선 신호입니다.")

    counts = stats.get("defect_type_counts") or {}
    st.markdown("**결함 유형 분포 (채택 기준)**")
    if counts:
        series = pd.Series(
            {DEFECT_TYPES.get(k, k): v for k, v in counts.items()}, name="건수"
        )
        st.bar_chart(series)
    else:
        st.info("아직 채택된 결함 기록이 없습니다.")

    st.divider()
    st.markdown("**YOLO 학습 라벨 export**")
    st.caption(
        "승인(accepted)된 결함 bbox를 YOLO 형식 라벨(txt)로 내보냅니다 — "
        "채택 = 양성 라벨, 기각 = 오탐 신호, 사람 추가 = 미탐 신호. "
        "이 데이터로 재학습한 가중치를 weights/best.pt에 두면 탐지 부품이 교체됩니다."
    )
    if st.button("YOLO 라벨 export", key="wb_btn_export"):
        out_dir = config.EXPORTS_DIR / f"yolo_labels_{datetime.now():%Y%m%d_%H%M%S}"
        try:
            config.ensure_dirs()
            result = archive_db.Archive().export_yolo_labels(out_dir)
        except Exception as exc:
            st.error(f"라벨 export에 실패했습니다: {exc}")
        else:
            st.success(
                f"export 완료 — 기록 {result.get('records', 0)}건, "
                f"라벨 {result.get('labels', 0)}줄"
            )
            st.code(str(result.get("out_dir", out_dir)), language=None)


# ---------------------------------------------------------------------------
# 탭 4 — 기준표·정보
# ---------------------------------------------------------------------------


def render_tab_info() -> None:
    st.subheader("판정 기준표")
    try:
        engine = rules.RuleEngine()
        meta = engine.meta
        st.warning(f"⚠️ {meta.get('disclaimer', '')}")
        st.caption(f"기준표: {meta.get('name', '')} · 버전 {meta.get('version', '')}")
    except Exception as exc:
        st.error(f"기준표 로드에 실패했습니다: {exc}")

    try:
        criteria = json.loads(Path(config.CRITERIA_PATH).read_text(encoding="utf-8"))
        st.json(criteria, expanded=False)
    except Exception as exc:
        st.error(f"기준표 JSON 표시에 실패했습니다: {exc}")

    st.divider()
    st.subheader("3대 설계 원칙")
    st.markdown(PRINCIPLES_FULL)

    st.divider()
    st.subheader("탐지 백엔드 상태")
    status = get_detector_status()
    st.write(
        {
            "backend": status.get("backend"),
            "yolo_available": status.get("yolo_available"),
            "weights_path": status.get("weights_path"),
            "note": status.get("note"),
        }
    )

    st.divider()
    st.caption(f"RT 판독 워크벤치 v{__version__} — \"판독은 자격자가, 서류는 AI가.\"")


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------


def render_sidebar() -> InspectionContext:
    with st.sidebar:
        st.title("🩻 RT 판독 워크벤치")
        st.caption(PRINCIPLES_SHORT)
        st.divider()

        st.markdown("**검사 컨텍스트**")
        st.text_input("필름 ID", key="wb_ctx_film_id", placeholder="예: F-2026-0830-01")
        st.text_input("블록", key="wb_ctx_block", placeholder="예: 3번 블록")
        st.text_input("용접부 ID", key="wb_ctx_weld_id", placeholder="예: BL3-V-012")
        st.selectbox("이음 종류", JOINT_TYPES, key="wb_ctx_joint")
        # 두께/품질등급은 룰 판정의 직접 입력 — 바뀌면 기존 판정을 무효화한다
        st.number_input(
            "모재 두께 (mm)", min_value=0.1, max_value=500.0, value=12.0, step=0.5,
            key="wb_ctx_thickness", on_change=invalidate_judgment,
        )
        st.selectbox(
            "품질등급 (B: 엄격 > C > D)", list(QUALITY_LEVELS),
            key="wb_ctx_quality", on_change=invalidate_judgment,
        )
        st.text_input("판독원", key="wb_ctx_inspector", placeholder="성명/자격번호")

        st.divider()
        st.markdown("**탐지 백엔드**")
        status = get_detector_status()
        icon = "🟢" if status.get("yolo_available") else "🟡"
        st.caption(f"{icon} `{status.get('backend')}` — {status.get('note', '')}")

        st.divider()
        st.caption("판정 주체: 자격 판독원 + 결정론적 룰 엔진 · AI는 후보 표시와 문서화 보조만.")

    return build_context_from_sidebar()


# ---------------------------------------------------------------------------
# 엔트리 포인트
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="RT 판독 워크벤치", page_icon="🩻", layout="wide")
    init_state()
    try:
        config.ensure_dirs()
    except Exception:
        pass  # 디렉터리 생성 실패는 각 기능 호출부 에러로 드러난다

    ctx = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 판독 워크벤치", "🗂️ 아카이브 검색", "♻️ 자기개선 루프", "📖 기준표·정보"]
    )
    with tab1:
        render_tab_workbench(ctx)
    with tab2:
        render_tab_archive()
    with tab3:
        render_tab_loop()
    with tab4:
        render_tab_info()


main()
