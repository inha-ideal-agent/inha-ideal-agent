"""CV 폴백 탐지기 측정 스크립트 — 합성 필름 6장에서 재현율(recall)과 장당 오탐을 잰다.

detection.py 의 CVFallbackDetector 주석("합성 필름 6장 실측: 임계 0.5 에서 재현율 14/14, 장당 오탐
3.8건")과 앱·계획서가 인용하는 탐지 수치는 이 스크립트의 출력이다. 운영점(MIN_CONFIDENCE)이나
탐지 파라미터를 바꾸면 다시 실행해 두 곳의 숫자를 갱신한다(tests/test_detection.py 가 대조한다).

정의
- 정답(GT): scripts/generate_samples.py 의 SAMPLE_SPECS(seed·결함 구성)로 만든 필름의 심은 결함 bbox.
- 매칭: IoU > 0.1 이거나 어느 한쪽 bbox 중심이 상대 bbox 안에 있으면 '후보로 제시됨'.
- 재현율 = 매칭된 GT 수 / 전체 GT 수.  장당 오탐 = (제시 후보 수 − 매칭된 GT 수) / 필름 수
  — 한 GT 에 후보가 여럿 겹쳐도 그 GT 는 한 번만 '검출'로 세고, 나머지는 보수적으로 오탐에 넣는다.

실행: cd <프로젝트 루트> && python3 scripts/eval_detector.py [--min-confidence 0.5] [--max-candidates 30]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 보장 (scripts/ 밖에서 실행해도 동작)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rtworkbench.detection import CVFallbackDetector, DefectCandidate  # noqa: E402

_GEN_PATH = PROJECT_ROOT / "scripts" / "generate_samples.py"


def load_generator():
    """scripts/generate_samples.py 를 모듈로 로드 (scripts 는 패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location("generate_samples_for_eval", _GEN_PATH)
    assert spec is not None and spec.loader is not None
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


# --- 매칭 유틸 (tests/test_detection.py 와 같은 정의) -------------------------


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def _center(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _contains(bbox, pt) -> bool:
    return bbox[0] <= pt[0] <= bbox[2] and bbox[1] <= pt[1] <= bbox[3]


def matched(gt_bbox, cand_bbox) -> bool:
    """IoU > 0.1 이거나, 어느 한쪽 bbox 중심이 상대 bbox 안이면 '후보 제시됨'."""
    return (
        iou(gt_bbox, cand_bbox) > 0.1
        or _contains(gt_bbox, _center(cand_bbox))
        or _contains(cand_bbox, _center(gt_bbox))
    )


# --- 측정 ---------------------------------------------------------------------


def evaluate(
    min_confidence: float | None = None,
    max_candidates: int = 30,
    *,
    detector: CVFallbackDetector | None = None,
    specs=None,
) -> dict:
    """SAMPLE_SPECS 전체에 대해 탐지기를 돌려 재현율·장당 오탐을 계산한다.

    Returns: {
      "films": [{name, seed, n_gt, n_hit, n_cands, n_fp, misses:[gt dict], per_gt:[(gt, best_cand|None)]}],
      "n_films", "gt_total", "hit_total", "recall", "fp_total", "fp_per_film",
      "min_confidence", "max_candidates",
    }
    """
    gen = load_generator()
    det = detector or CVFallbackDetector()
    specs = list(specs if specs is not None else gen.SAMPLE_SPECS)
    effective_conf = det.MIN_CONFIDENCE if min_confidence is None else float(min_confidence)

    films: list[dict] = []
    gt_total = hit_total = fp_total = 0
    for name, seed, spec_list in specs:
        img, gt = gen.generate_film(seed=seed, defects_spec=spec_list)
        cands: list[DefectCandidate] = det.detect(
            img, max_candidates=max_candidates, min_confidence=min_confidence
        )
        per_gt: list[tuple[dict, DefectCandidate | None]] = []
        misses: list[dict] = []
        for g in gt:
            best: DefectCandidate | None = None
            for c in cands:
                if matched(g["bbox"], c.bbox) and (best is None or c.confidence > best.confidence):
                    best = c
            per_gt.append((g, best))
            if best is None:
                misses.append(g)
        n_hit = len(gt) - len(misses)
        n_fp = max(0, len(cands) - n_hit)  # 보수적: 검출로 세지 않은 후보는 전부 오탐
        films.append({
            "name": name, "seed": seed, "n_gt": len(gt), "n_hit": n_hit,
            "n_cands": len(cands), "n_fp": n_fp, "misses": misses, "per_gt": per_gt,
        })
        gt_total += len(gt)
        hit_total += n_hit
        fp_total += n_fp

    n_films = len(films)
    return {
        "films": films,
        "n_films": n_films,
        "gt_total": gt_total,
        "hit_total": hit_total,
        "recall": hit_total / gt_total if gt_total else 0.0,
        "fp_total": fp_total,
        "fp_per_film": fp_total / n_films if n_films else 0.0,
        "min_confidence": effective_conf,
        "max_candidates": max_candidates,
    }


def format_summary(stats: dict) -> str:
    """detection.py 주석·앱 캡션에 옮겨 적는 한 줄 요약."""
    return (
        f"합성 필름 {stats['n_films']}장 실측: 임계 {stats['min_confidence']:.1f} 에서 "
        f"재현율 {stats['hit_total']}/{stats['gt_total']}, 장당 오탐 {stats['fp_per_film']:.1f}건"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="후보 최소 신뢰도 (기본: CVFallbackDetector.MIN_CONFIDENCE, 0.0 이면 필터 없음)")
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--verbose", "-v", action="store_true", help="GT 별 매칭 후보를 출력")
    args = ap.parse_args(argv)

    stats = evaluate(args.min_confidence, args.max_candidates)
    for f in stats["films"]:
        types: dict[str, int] = {}
        for _g, best in f["per_gt"]:
            if best is not None:
                types[best.defect_type] = types.get(best.defect_type, 0) + 1
        print(f"{f['name']}: gt={f['n_gt']} hit={f['n_hit']} cands={f['n_cands']} fp={f['n_fp']}")
        for m in f["misses"]:
            print(f"   MISS: {m}")
        if args.verbose:
            for g, best in f["per_gt"]:
                tag = f"→ {best.defect_type} conf={best.confidence:.2f}" if best else "→ (미탐)"
                print(f"   gt {g['defect_type']:>18} {tag}")
    print()
    print(f"RECALL = {stats['hit_total']}/{stats['gt_total']} = {stats['recall']:.2f}")
    print(f"FP/FILM = {stats['fp_total']}/{stats['n_films']} = {stats['fp_per_film']:.2f}")
    print(format_summary(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
