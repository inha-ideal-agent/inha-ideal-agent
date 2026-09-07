"""승인 소견서 PDF 출력 (reportlab, 한글).

한글 폰트는 reportlab 내장 CID 폰트 'HYSMyeongJo-Medium'을 등록해 사용한다
(별도 폰트 파일 불필요). platypus(SimpleDocTemplate/Table/Paragraph)로 A4 문서를
구성하고 bytes 를 반환한다. 실패 시 예외를 전파한다(호출부 UI에서 처리).

구성: 헤더(문서명 'RT 판독 소견서', record_id, 일시)
    / 검사 정보 표 / 결함 판정 표 / 종합 판정(크게) / 소견서 본문(줄바꿈 유지)
    / 선급 보고 항목 표(IACS UR W33 §8.2·8.5 — 입력된 항목만, 없으면 섹션 생략)
    / 푸터(적용 기준표 이름·버전, 초안 생성 경로, 판정 주체 고지, 데모 기준표 디스클레이머).
"""

from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from rtworkbench.models import (
    DEFECT_TYPES,
    REPORT_GROUPS,
    W33_GROUP_COUNTS,
    InspectionRecord,
    w33_coverage,
)

# reportlab 내장 한글 CID 폰트 — 폰트 파일 배포 불필요
_FONT = "HYSMyeongJo-Medium"

# 초안 생성 경로 표기
_SOURCE_KO = {
    "claude": "Claude API",
    "gemini": "Gemini API",
    "local": "로컬 LLM (온프레미스)",
    "template": "오프라인 템플릿",
    "cache": "로컬 캐시(API 사전 생성)",
}

# 판본 분리 전(구버전) 기록 — 적용 기준표가 기록되지 않음
_CRITERIA_UNKNOWN = "미기록 (판본 분리 전 기록)"

_LINE = colors.HexColor("#8a8a8a")
_HEAD_BG = colors.HexColor("#e8e8e8")
_LABEL_BG = colors.HexColor("#f2f2f2")
_PASS = colors.HexColor("#1a7f37")
_FAIL = colors.HexColor("#c62828")


def _register_korean_font() -> None:
    """한글 CID 폰트 등록. 중복 호출해도 안전하다."""
    pdfmetrics.registerFont(UnicodeCIDFont(_FONT))


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle("title", fontName=_FONT, fontSize=17, leading=22, alignment=1),
        "meta": ParagraphStyle(
            "meta", fontName=_FONT, fontSize=8.5, leading=12, alignment=1,
            textColor=colors.HexColor("#555555"),
        ),
        "h2": ParagraphStyle(
            "h2", fontName=_FONT, fontSize=11.5, leading=15, spaceBefore=6, spaceAfter=3,
        ),
        "body": ParagraphStyle("body", fontName=_FONT, fontSize=9.5, leading=14),
        "cell": ParagraphStyle("cell", fontName=_FONT, fontSize=9, leading=12),
        "cell_center": ParagraphStyle(
            "cell_center", fontName=_FONT, fontSize=9, leading=12, alignment=1,
        ),
        "footer": ParagraphStyle(
            "footer", fontName=_FONT, fontSize=8, leading=11,
            textColor=colors.HexColor("#555555"),
        ),
    }


def _glyph_safe(text: str) -> str:
    """내장 CID 폰트에 글리프가 없는 문자를 같은 모양의 문자로 치환.

    가운뎃점 U+00B7('·')은 글리프가 없어 조용히 사라진다('선원 종류·크기' → '선원 종류크기').
    같은 모양으로 렌더링되는 한글 아래아 U+318D('ㆍ')로 바꾼다. 마크업이 섞인 문자열에도 안전하다.
    """
    return text.replace("·", "ㆍ")


def _xml(text: object) -> str:
    """Paragraph 마크업용 이스케이프 + 줄바꿈 유지 + 글리프 치환."""
    return _glyph_safe(escape(str(text)).replace("\n", "<br/>"))


def build_pdf(record: InspectionRecord) -> bytes:
    """InspectionRecord → A4 PDF bytes. 실패 시 예외 전파."""
    _register_korean_font()
    st = _styles()
    ctx = record.context

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="RT 판독 소견서",
        author="RT 판독 워크벤치",
    )

    story: list = []

    # ------------------------------------------------------------ 헤더
    story.append(Paragraph("RT 판독 소견서", st["title"]))
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            f"기록 ID: {_xml(record.record_id)} · 작성 일시: {_xml(record.created_at)}",
            st["meta"],
        )
    )
    story.append(Spacer(1, 5 * mm))

    # ------------------------------------------------------------ 검사 정보 표
    story.append(Paragraph("1. 검사 정보", st["h2"]))
    if ctx.scale_mm_per_px is not None:
        scale_txt = f"{ctx.scale_mm_per_px:.5f} mm/px"
        if ctx.scale_ref:
            scale_txt += f" (기준물: {ctx.scale_ref})"
    else:
        scale_txt = "미확정"
    info_rows = [
        ("필름 ID", ctx.film_id or "-", "블록", ctx.block or "-"),
        ("용접부 ID", ctx.weld_id or "-", "이음 종류", ctx.joint_type or "-"),
        ("모재 두께", f"{ctx.thickness_mm:g} mm", "품질 등급", ctx.quality_level or "-"),
        ("판독원", ctx.inspector or "-", "검사 기법", ctx.technique or "-"),
        ("스케일", scale_txt, "원본 이미지", record.image_name or "-"),
        (
            "평가 길이", f"{ctx.eval_length_mm:g} mm",
            "용접부 폭", f"{ctx.weld_width_mm:g} mm",
        ),
        # 적용 기준표(규격 판본) — 판본 분리 전 기록은 미기록으로 표시
        (
            "적용 기준표", record.criteria_name or _CRITERIA_UNKNOWN,
            "기준표 버전", record.criteria_version or "-",
        ),
    ]
    info_data = [
        [Paragraph(_xml(cell), st["cell"]) for cell in row] for row in info_rows
    ]
    info_table = Table(info_data, colWidths=[26 * mm, 61 * mm, 26 * mm, 61 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, _LINE),
                ("BACKGROUND", (0, 0), (0, -1), _LABEL_BG),
                ("BACKGROUND", (2, 0), (2, -1), _LABEL_BG),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 4 * mm))

    # ------------------------------------------------------------ 결함 판정 표
    story.append(Paragraph("2. 결함 판정 (결정론적 룰 엔진)", st["h2"]))
    # 단위는 셀마다 명시한다 — 단일 판정은 mm, 투영 면적률 그룹 판정은 %
    head = ("ID", "유형", "크기", "허용한계", "근거 조항", "합부")
    rows: list[tuple[str, ...]] = [head]
    if record.verdicts:
        for v in record.verdicts:
            rows.append(
                (
                    v.display_id,  # 그룹 판정은 '합계(기공)' 형태
                    DEFECT_TYPES.get(v.defect_type, v.defect_type),
                    f"{v.size_mm:.2f} {v.unit}",
                    "허용 불가" if v.limit_mm is None else f"{v.limit_mm:.2f} {v.unit}",
                    v.clause,
                    "합격" if v.passed else "불합격",
                )
            )
    else:
        rows.append(("-", "유의미한 결함 지시 없음", "-", "-", "-", "-"))
    verdict_data = [
        [Paragraph(_xml(cell), st["cell_center"]) for cell in row] for row in rows
    ]
    verdict_table = Table(
        verdict_data, colWidths=[30 * mm, 34 * mm, 24 * mm, 30 * mm, 34 * mm, 22 * mm]
    )
    verdict_style = [
        ("GRID", (0, 0), (-1, -1), 0.4, _LINE),
        ("BACKGROUND", (0, 0), (-1, 0), _HEAD_BG),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # 불합격 행 강조 (헤더 다음 행부터 verdicts 순서와 동일)
    for i, v in enumerate(record.verdicts, start=1):
        if not v.passed:
            verdict_style.append(("TEXTCOLOR", (0, i), (-1, i), _FAIL))
    verdict_table.setStyle(TableStyle(verdict_style))
    story.append(verdict_table)
    group_verdicts = [v for v in record.verdicts if v.is_group]
    if group_verdicts:
        # 100 mm 판독 구간 규칙(IACS UR W33 Rev.2) — 어느 구간으로 판정했는지 기록에 남긴다
        windows = [
            f"{DEFECT_TYPES.get(v.defect_type, v.defect_type)} [{v.window_mm[0]:g}~{v.window_mm[1]:g} mm]"
            for v in group_verdicts if v.window_mm is not None
        ]
        whole = [
            DEFECT_TYPES.get(v.defect_type, v.defect_type)
            for v in group_verdicts if v.window_mm is None
        ]
        note = (
            "합계 행 = 유형별 그룹 판정: 누적 길이(평가 길이 내 길이 합) · "
            "투영 면적률(원 근사 면적 합 ÷ 평가 길이 × 용접부 폭, %). "
            f"평가 길이 {ctx.eval_length_mm:g} mm 창을 용접선을 따라 이동시켜 가장 불리한 구간으로 판정"
            "(IACS UR W33 Rev.2 100 mm 판독 구간 규칙)."
        )
        if windows:
            note += " 최악 구간: " + ", ".join(windows) + "."
        if whole:
            note += " 위치 정보 없음 → 전체를 한 구간으로 보수적 합산: " + ", ".join(whole) + "."
        story.append(Paragraph(_glyph_safe(note), st["footer"]))
    story.append(Spacer(1, 5 * mm))

    # ------------------------------------------------------------ 종합 판정 (크게)
    overall_txt = "합격" if record.overall_passed else "불합격"
    overall_style = ParagraphStyle(
        "overall",
        fontName=_FONT,
        fontSize=15,
        leading=19,
        alignment=1,
        textColor=_PASS if record.overall_passed else _FAIL,
    )
    overall_table = Table(
        [[Paragraph(f"3. 종합 판정 : {overall_txt}", overall_style)]],
        colWidths=[174 * mm],
    )
    overall_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1.0, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(overall_table)
    story.append(Spacer(1, 5 * mm))

    # ------------------------------------------------------------ 소견서 본문
    story.append(Paragraph("4. 소견서 본문 (판독원 승인)", st["h2"]))
    story.append(Paragraph(_xml(record.report_text or "-"), st["body"]))
    story.append(Spacer(1, 8 * mm))

    # ------------------------------------------------------------ 선급 보고 항목 (입력된 것만)
    filled = ctx.report.filled_items()
    if filled:
        story.append(Paragraph("5. 선급 보고 항목(IACS UR W33 §8)", st["h2"]))
        # 충족률은 W33 원문 불릿 25개(일반 13 · RT 12) 기준 — 검사 정보 표·판정 표로 충족되는 항목 포함
        n_filled, n_total, missing = w33_coverage(ctx, record)
        header = (
            f"보고 항목 충족 {n_filled}/{n_total} "
            f"(IACS UR W33 §8.2 일반 {W33_GROUP_COUNTS['general']} · §8.5 RT {W33_GROUP_COUNTS['rt']}"
            + (f"; 미충족: {', '.join(missing)}" if missing else "; 모두 충족")
            + ") · 아래 표는 판정에 관여하지 않는 보고서 기재 사항 "
            "(§8.2 일반 항목 중 검사 정보 표에 없는 것 + §8.5 RT 전용 항목, 입력된 것만)."
        )
        story.append(Paragraph(_xml(header), st["footer"]))
        story.append(Spacer(1, 1.5 * mm))
        report_rows: list[list] = [
            [Paragraph("항목", st["cell_center"]), Paragraph("값", st["cell_center"])]
        ]
        report_style = [
            ("GRID", (0, 0), (-1, -1), 0.4, _LINE),
            ("BACKGROUND", (0, 0), (-1, 0), _HEAD_BG),
            ("BACKGROUND", (0, 1), (0, -1), _LABEL_BG),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        current_group = None
        for item, value in filled:
            if item.group != current_group:
                # 그룹 구분 행(§8.2 일반 / §8.5 RT 촬영 조건) — 두 칸 병합
                current_group = item.group
                row_idx = len(report_rows)
                report_rows.append(
                    [Paragraph(_xml(REPORT_GROUPS.get(item.group, item.group)), st["cell"]), ""]
                )
                report_style.append(("SPAN", (0, row_idx), (1, row_idx)))
                report_style.append(("BACKGROUND", (0, row_idx), (1, row_idx), _HEAD_BG))
            report_rows.append(
                [
                    Paragraph(
                        f"{_xml(item.label_ko)}<br/>"
                        f"<font size=7 color='#666666'>{_xml(item.label_en)}</font>",
                        st["cell"],
                    ),
                    Paragraph(_xml(value), st["cell"]),
                ]
            )
        report_table = Table(report_rows, colWidths=[78 * mm, 96 * mm], repeatRows=1)
        report_table.setStyle(TableStyle(report_style))
        story.append(report_table)
        story.append(Spacer(1, 8 * mm))

    # ------------------------------------------------------------ 푸터
    source_ko = _SOURCE_KO.get(record.report_source, record.report_source)
    criteria_txt = record.criteria_label or _CRITERIA_UNKNOWN
    footer = (
        f"적용 기준표: {_xml(criteria_txt)}<br/>"
        f"초안 생성 경로: {_xml(source_ko)} · "
        "판정 주체: 자격 판독원 / AI는 후보 표시·문서화 보조 역할에 한함.<br/>"
        "본 문서의 판정에 적용된 기준표는 ISO 5817 계열의 구조를 모사한 데모 기준이며, "
        "실제 생산 검사 적용 시 선급 NDT 지침에 따라 확정된 기준표로 교체해야 합니다."
    )
    story.append(Paragraph(_glyph_safe(footer), st["footer"]))

    doc.build(story)
    return buf.getvalue()
