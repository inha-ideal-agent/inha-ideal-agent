"""RT 판독 워크벤치 — E2E 데모 스크립트.

전체 파이프라인을 오프라인으로 검증한다:
  샘플 로드 → CLAHE 전처리 → 탐지(후보 채택) → 납마커 캘리브레이션 →
  2클릭 측정 → 룰 판정 → 페이로드 → 소견서(템플릿 경로) → PDF →
  아카이브 저장 → 검색 재조회 → YOLO 라벨 export.

실행: cd <프로젝트 루트> && python3 scripts/demo_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 프로젝트 루트를 sys.path에 보장 (scripts/ 밖에서 실행해도 동작)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rtworkbench import config, measure
from rtworkbench.db import Archive
from rtworkbench.detection import detector_status, get_detector
from rtworkbench.models import (
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    Measurement,
    new_id,
)
from rtworkbench.preprocess import apply_clahe, load_grayscale
from rtworkbench.report_llm import ReportWriter, build_payload
from rtworkbench.report_pdf import build_pdf
from rtworkbench.rules import RuleEngine, overall_pass


def step(msg: str) -> None:
    print(f"[E2E] {msg}")


def main() -> None:
    config.ensure_dirs()

    # ---------------------------------------------------------- 1. 샘플 로드
    sample_png = config.SAMPLES_DIR / "sample_02.png"
    sample_json = config.SAMPLES_DIR / "sample_02.json"
    assert sample_png.exists() and sample_json.exists(), "샘플 파일이 없음 — scripts/generate_samples.py 먼저 실행"
    meta = json.loads(sample_json.read_text(encoding="utf-8"))
    img = load_grayscale(sample_png)
    h, w = img.shape[:2]
    assert [w, h] == meta["image_size"], "이미지 크기와 JSON 메타 불일치"
    step(f"1. 샘플 로드 완료: {sample_png.name} ({w}x{h}), ground truth {len(meta['ground_truth'])}건")

    # ---------------------------------------------------------- 2. CLAHE
    enhanced = apply_clahe(img)
    assert enhanced.shape == img.shape
    step(f"2. CLAHE 전처리 완료 (평균 밝기 {img.mean():.1f} → {enhanced.mean():.1f})")

    # ---------------------------------------------------------- 3. 탐지
    status = detector_status()
    detector = get_detector()
    candidates = detector.detect(img, max_candidates=30)
    assert len(candidates) >= 1, "탐지 후보가 없음 — recall 우선 원칙 위반"
    # 판독원 채택 시뮬레이션: confidence 상위 후보들 채택, 나머지 기각
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    n_accept = min(3, len(candidates))
    for i, c in enumerate(candidates):
        c.status = "accepted" if i < n_accept else "rejected"
    accepted = [c for c in candidates if c.status == "accepted"]
    step(
        f"3. 탐지 완료 (백엔드={status.get('backend', '?')}): 후보 {len(candidates)}건, "
        f"채택 {len(accepted)}건, 기각 {len(candidates) - len(accepted)}건"
    )

    # ---------------------------------------------------------- 4. 캘리브레이션
    mk = meta["marker"]
    mm_per_px = measure.calibrate(tuple(mk["p1"]), tuple(mk["p2"]), mk["length_mm"])
    assert abs(mm_per_px - meta["mm_per_px"]) < 1e-6, "캘리브레이션 스케일 불일치"
    step(f"4. 납마커 캘리브레이션 완료: {mm_per_px:.4f} mm/px (기준 {mk['length_mm']}mm)")

    # ---------------------------------------------------------- 5. 2클릭 측정
    measurements: list[Measurement] = []
    for c in accepted:
        x1, y1, x2, y2 = c.bbox
        p1, p2 = (x1, y1), (x2, y2)  # bbox 대각선을 2클릭으로 간주
        length_px, length_mm = measure.measure(p1, p2, mm_per_px)
        assert length_mm is not None and length_mm > 0
        measurements.append(
            Measurement(defect_id=c.id, p1=p1, p2=p2, length_px=length_px, length_mm=length_mm)
        )
    step("5. 2클릭 측정 완료: " + ", ".join(f"{m.length_mm:.2f}mm" for m in measurements))

    # ---------------------------------------------------------- 6. 룰 판정
    context = InspectionContext(
        film_id="DEMO-E2E-001",
        block="3번 블록",
        weld_id="W-102",
        thickness_mm=12.0,
        quality_level="B",
        inspector="데모 판독원",
        scale_mm_per_px=mm_per_px,
        scale_ref=f"납마커 {mk['length_mm']:g}mm",
    )
    engine = RuleEngine()
    items = [(c.id, c.defect_type, m.length_mm) for c, m in zip(accepted, measurements)]
    verdicts = engine.evaluate_all(items, context.thickness_mm, context.quality_level)
    ok = overall_pass(verdicts)
    step(
        f"6. 룰 판정 완료: {len(verdicts)}건, 종합 {'합격' if ok else '불합격'} "
        f"({sum(v.passed for v in verdicts)}건 합격 / {sum(not v.passed for v in verdicts)}건 불합격)"
    )

    # ---------------------------------------------------------- 7. 페이로드 + 소견서(템플릿)
    payload = build_payload(context, verdicts, measurements)
    assert "데모 판독원" not in payload, "비식별 원칙 위반: 페이로드에 검사원 실명 포함"
    # 오프라인 템플릿 경로 강제: API 키를 비워 폴백 유도
    config.ANTHROPIC_API_KEY = ""
    config.GEMINI_API_KEY = ""
    writer = ReportWriter()
    report_text, source = writer.generate(payload)
    assert source == "template", f"오프라인 템플릿 경로 기대, 실제 source={source}"
    assert report_text.strip(), "소견서 본문이 비어 있음"
    step(f"7. 소견서 초안 생성 완료 (source={source}, {len(report_text)}자)")

    # ---------------------------------------------------------- 8. PDF
    record_id = new_id("rec")
    record = InspectionRecord(
        record_id=record_id,
        context=context,
        candidates=candidates,
        measurements=measurements,
        verdicts=verdicts,
        overall_passed=ok,
        report_text=report_text,
        report_source=source,
        image_name=sample_png.name,
        image_size=(w, h),
    )
    pdf_bytes = build_pdf(record)
    assert isinstance(pdf_bytes, bytes) and pdf_bytes.startswith(b"%PDF"), "PDF 바이트 검증 실패"
    pdf_path = config.EXPORTS_DIR / f"demo_e2e_{record_id}.pdf"
    pdf_path.write_bytes(pdf_bytes)
    step(f"8. PDF 생성 완료: {pdf_path} ({len(pdf_bytes)} bytes)")

    # ---------------------------------------------------------- 9. 아카이브 저장
    archive = Archive(config.DB_PATH)
    archive.save(record)
    loaded = archive.get(record_id)
    assert loaded is not None and loaded.record_id == record_id, "저장 기록 재조회 실패"
    step(f"9. 아카이브 저장 완료: {config.DB_PATH} (record_id={record_id})")

    # ---------------------------------------------------------- 10. 검색 재조회
    df = archive.search(block="3번 블록")
    assert record_id in df["record_id"].tolist(), "검색 결과에 저장 기록이 없음"
    step(f"10. 검색 재조회 확인: '3번 블록' 필터 → {len(df)}건 (저장 기록 포함)")

    # ---------------------------------------------------------- 11. YOLO 라벨 export
    out_dir = config.EXPORTS_DIR / f"yolo_labels_demo_{time.strftime('%Y%m%d_%H%M%S')}"
    result = archive.export_yolo_labels(out_dir)
    assert result["records"] >= 1, "export된 record가 없음"
    assert (Path(result["out_dir"]) / "classes.txt").exists(), "classes.txt 미생성"
    step(
        f"11. YOLO 라벨 export 완료: records={result['records']}, "
        f"labels={result['labels']}, out_dir={result['out_dir']}"
    )

    step("전체 E2E 파이프라인 통과.")


if __name__ == "__main__":
    main()
