"""합성 RT 샘플 필름 생성기.

실제 조선소 필름은 반출 불가(방산 보안)이므로, 데모/테스트용 합성 방사선 사진을
생성한다. AI Hub/RIAWELC 실데이터 학습 전까지의 개발용 안전망.

실행: cd rt-workbench && python3 scripts/generate_samples.py
출력: data/samples/sample_01.png ~ sample_06.png
      + 각 sample_XX.json (ground truth, mm_per_px, 납마커 캘리브레이션 정보)

스케일 계약: 정확히 5.0 px/mm (mm_per_px = 0.2).
납마커 바 양 끝점 거리 = 정확히 50 px = 10 mm (2클릭 캘리브레이션 데모용).

필름 구성:
  - 배경: 어두운 모재 톤 + 감광 그라데이션 + 저주파 모틀 + 필름 그레인
  - 중앙 수평 용접 비드 밴드(밝은 띠 + 물결 리플 텍스처, 중심선 요동)
  - 좌측 IQI 와이어 세트(수직 얇은 선 7개, 간격 일정, 점점 가늘어짐)
  - 좌하단 납마커 바(밝음 — 납이 방사선을 차폐)
  - 결함 시딩(비드 밴드 안): porosity / cluster_porosity / crack /
    lack_of_fusion / slag_inclusion — bbox ground truth 기록

numpy.random.default_rng(seed) 사용 — 같은 seed 는 같은 이미지(테스트 재현성).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# scripts/ 를 직접 실행해도 rtworkbench 패키지를 찾도록 프로젝트 루트를 경로에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rtworkbench import config  # noqa: E402

# --- 스케일 계약 (동결) -----------------------------------------------------
PX_PER_MM = 5.0
MM_PER_PX = 1.0 / PX_PER_MM  # = 0.2
MARKER_LEN_PX = 50  # 납마커 양 끝점 거리 = 50 px
MARKER_LEN_MM = 10.0  # = 10 mm


def marker_geometry(width: int = 1400, height: int = 520) -> dict:
    """납마커 바의 캘리브레이션 끝점 2개(p1, p2)와 실제 길이(mm)를 반환한다.

    generate_film 이 그리는 위치와 동일한 결정론적 기하 — |p2 - p1| = 정확히 50 px.
    """
    mx, my = 80, height - 58  # 바 좌상단
    y_mid = float(my + 5)  # 바 두께 10px 의 세로 중앙
    return {
        "p1": [float(mx), y_mid],
        "p2": [float(mx + MARKER_LEN_PX), y_mid],
        "length_mm": MARKER_LEN_MM,
    }


# --- 결함 드로잉 (float 이미지에 어두운 지시를 새김) -------------------------


def _subtract_gaussian(
    img: np.ndarray, cx: float, cy: float, sx: float, sy: float, depth: float
) -> None:
    """(cx,cy) 중심의 이방성 가우시안 만큼 밝기를 깎는다(어두운 지시)."""
    h, w = img.shape
    ext = int(max(sx, sy) * 4 + 3)
    x1, x2 = max(0, int(cx) - ext), min(w, int(cx) + ext)
    y1, y2 = max(0, int(cy) - ext), min(h, int(cy) + ext)
    if x2 <= x1 or y2 <= y1:
        return
    yy, xx = np.mgrid[y1:y2, x1:x2]
    g = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))
    img[y1:y2, x1:x2] -= depth * g


def _draw_porosity(img: np.ndarray, rng: np.random.Generator, cx: float, cy: float) -> list[float]:
    """기공 — 어두운 원형 반점."""
    r = float(rng.uniform(3.5, 7.0))
    depth = float(rng.uniform(48, 75))
    _subtract_gaussian(img, cx, cy, r / 1.6, r / 1.6, depth)
    pad = r + 2.0
    return [cx - pad, cy - pad, cx + pad, cy + pad]


def _draw_cluster_porosity(
    img: np.ndarray, rng: np.random.Generator, cx: float, cy: float
) -> list[float]:
    """군집 기공 — 소형 기공 여러 개가 모인 군집 (포락 bbox 1건으로 기록)."""
    n = int(rng.integers(4, 7))
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(n):
        ox = float(rng.uniform(-18, 18))
        oy = float(rng.uniform(-10, 10))
        r = float(rng.uniform(2.4, 4.2))
        depth = float(rng.uniform(42, 62))
        _subtract_gaussian(img, cx + ox, cy + oy, r / 1.6, r / 1.6, depth)
        xs += [cx + ox - r - 2, cx + ox + r + 2]
        ys += [cy + oy - r - 2, cy + oy + r + 2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _draw_crack(img: np.ndarray, rng: np.random.Generator, cx: float, cy: float) -> list[float]:
    """균열 — 가늘고 어두운 지그재그 선 (대체로 수평, 약한 기울기)."""
    total = float(rng.uniform(75, 130))
    depth = float(rng.uniform(58, 85))
    tilt = float(rng.uniform(-0.12, 0.12))
    n_seg = int(rng.integers(6, 10))
    step = total / n_seg

    pts = [(cx - total / 2.0, cy)]
    x, y = pts[0]
    for _ in range(n_seg):
        x += step
        jag = float(rng.uniform(-4.0, 4.0))
        y = float(np.clip(cy + tilt * (x - cx) + jag, cy - 10, cy + 10))
        pts.append((x, y))

    h, w = img.shape
    line_mask = np.zeros((h, w), dtype=np.float32)
    ipts = np.array([[int(round(px)), int(round(py))] for px, py in pts], dtype=np.int32)
    cv2.polylines(line_mask, [ipts], isClosed=False, color=1.0, thickness=2)
    line_mask = cv2.GaussianBlur(line_mask, (0, 0), 0.7)
    if line_mask.max() > 0:
        line_mask /= line_mask.max()
    img -= depth * line_mask

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3]


def _draw_lack_of_fusion(
    img: np.ndarray, rng: np.random.Generator, cx: float, cy: float
) -> list[float]:
    """융합불량 — 비드 경계의 길쭉하고 어두운 저대비 띠."""
    length = float(rng.uniform(90, 150))
    hh = float(rng.uniform(5.0, 8.0))
    depth = float(rng.uniform(24, 34))  # 저대비 — 느슨한 임계가 아니면 놓치기 쉬움
    _subtract_gaussian(img, cx, cy, length / 2.6, hh / 1.8, depth)
    return [cx - length / 2, cy - hh * 1.6, cx + length / 2, cy + hh * 1.6]


def _draw_slag_inclusion(
    img: np.ndarray, rng: np.random.Generator, cx: float, cy: float
) -> list[float]:
    """슬래그 개재물 — 불규칙한 어두운 덩어리(타원 여러 개의 합성)."""
    n = int(rng.integers(3, 6))
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(n):
        ox = float(rng.uniform(-12, 12))
        oy = float(rng.uniform(-4, 4))
        sx = float(rng.uniform(2.5, 5.0))
        sy = float(rng.uniform(2.0, 4.0))
        depth = float(rng.uniform(38, 58))
        _subtract_gaussian(img, cx + ox, cy + oy, sx, sy, depth)
        xs += [cx + ox - 2 * sx, cx + ox + 2 * sx]
        ys += [cy + oy - 2 * sy, cy + oy + 2 * sy]
    return [min(xs), min(ys), max(xs), max(ys)]


_DRAWERS = {
    "porosity": _draw_porosity,
    "cluster_porosity": _draw_cluster_porosity,
    "crack": _draw_crack,
    "lack_of_fusion": _draw_lack_of_fusion,
    "slag_inclusion": _draw_slag_inclusion,
}


# --- 필름 1장 생성 -----------------------------------------------------------


def generate_film(
    width: int = 1400,
    height: int = 520,
    seed: int = 0,
    defects_spec: list[str] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """합성 RT 필름 1장 생성 → (8-bit 그레이스케일 이미지, ground_truth 목록).

    defects_spec: 시딩할 결함 유형 문자열 목록(_DRAWERS 의 키).
      None 이면 seed 기반 무작위 구성(2~4개), [] 이면 무결함 필름.
    ground_truth: [{'defect_type': str, 'bbox': [x1, y1, x2, y2]}, ...] (원본 px 좌표).
    같은 seed 는 같은 이미지 — numpy.random.default_rng(seed) 만 사용한다.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)

    # 1) 배경: 모재 톤 + 감광 그라데이션 + 비네팅
    img = np.full((height, width), 62.0, dtype=np.float64)
    tilt = float(rng.uniform(-22, 22))
    img += tilt * (xx / width - 0.5)
    img += -10.0 * (((yy - height / 2.0) / (height / 2.0)) ** 2)

    # 2) 용접 비드 밴드: 요동하는 중심선 + 평탄한 상부(슈퍼가우시안) + 리플 텍스처
    y_c = height / 2.0
    wobble = 4.0 * np.sin(2 * np.pi * xx / 300.0 + float(rng.uniform(0, 2 * np.pi)))
    band_half = 40.0
    profile = np.exp(-(((yy - (y_c + wobble)) / band_half) ** 4))
    img += 62.0 * profile
    img += 5.0 * np.sin(2 * np.pi * xx / 55.0 + float(rng.uniform(0, 2 * np.pi))) * profile

    # 3) IQI 와이어 7개 (좌측, 일정 간격, 점점 가늘어지는 대비)
    wy1, wy2 = int(y_c - 58), int(y_c + 58)
    for i in range(7):
        wx = 70 + i * 14
        img[wy1:wy2, wx : wx + 2] += 26.0 - 2.5 * i

    # 4) 납마커 바 (밝음 — 납이 방사선을 차폐). 끝점 거리 = 정확히 50 px = 10 mm
    marker = marker_geometry(width, height)
    mx, my = int(marker["p1"][0]), int(marker["p1"][1] - 5)
    img[my : my + 10, mx : mx + MARKER_LEN_PX + 1] += 150.0

    # 5) 결함 시딩 (비드 밴드 안, 와이어/마커 회피 영역)
    if defects_spec is None:
        n = int(rng.integers(2, 5))
        pool = ["porosity", "porosity", "slag_inclusion", "lack_of_fusion", "crack"]
        defects_spec = [str(t) for t in rng.choice(pool, size=n)]

    ground_truth: list[dict] = []
    placed: list[tuple[float, float]] = []
    for dtype in defects_spec:
        if dtype not in _DRAWERS:
            raise ValueError(f"지원하지 않는 결함 유형: {dtype} (가능: {sorted(_DRAWERS)})")
        cx, cy = 0.0, y_c
        for _ in range(60):  # 겹침 회피 재시도
            cx = float(rng.uniform(240, width - 90))
            if dtype == "lack_of_fusion":
                edge = float(rng.choice([-1.0, 1.0]))
                cy = float(y_c + edge * rng.uniform(26, 33))
            else:
                cy = float(y_c + rng.uniform(-24, 24))
            if all(abs(cx - px) > 170 or abs(cy - py) > 70 for px, py in placed):
                break
        placed.append((cx, cy))
        bbox = _DRAWERS[dtype](img, rng, cx, cy)
        ground_truth.append(
            {"defect_type": dtype, "bbox": [round(float(v), 1) for v in bbox]}
        )

    # 6) 광학 블러 + 저주파 모틀 + 필름 그레인
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.9).astype(np.float64)
    coarse = rng.normal(0.0, 6.0, (height // 4, width // 4)).astype(np.float32)
    coarse = cv2.GaussianBlur(coarse, (0, 0), 1.2)
    img += cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    img += rng.normal(0.0, 3.0, (height, width))

    return np.clip(img, 0, 255).astype(np.uint8), ground_truth


# --- 샘플 세트 저장 ----------------------------------------------------------

# (파일명, seed, 결함 구성) — sample_01 은 무결함(합격 데모), sample_03/06 은 균열 포함(불합격 데모)
SAMPLE_SPECS: list[tuple[str, int, list[str]]] = [
    ("sample_01", 101, []),
    ("sample_02", 102, ["porosity", "porosity", "slag_inclusion"]),
    ("sample_03", 103, ["crack", "porosity"]),
    ("sample_04", 104, ["lack_of_fusion", "porosity", "porosity"]),
    ("sample_05", 105, ["cluster_porosity", "porosity"]),
    ("sample_06", 106, ["slag_inclusion", "lack_of_fusion", "crack", "porosity"]),
]


def main() -> None:
    """샘플 6장을 config.SAMPLES_DIR 에 저장한다 (PNG + ground truth JSON)."""
    config.ensure_dirs()
    for name, seed, spec in SAMPLE_SPECS:
        img, gt = generate_film(seed=seed, defects_spec=spec)
        png_path = config.SAMPLES_DIR / f"{name}.png"
        if not cv2.imwrite(str(png_path), img):
            raise RuntimeError(f"PNG 저장 실패: {png_path}")

        meta = {
            "film": f"{name}.png",
            "seed": seed,
            "image_size": [int(img.shape[1]), int(img.shape[0])],
            "mm_per_px": MM_PER_PX,
            "marker": marker_geometry(img.shape[1], img.shape[0]),
            "ground_truth": gt,
        }
        json_path = config.SAMPLES_DIR / f"{name}.json"
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[생성] {png_path.name}  결함 {len(gt)}건  → {json_path.name}")

    print(f"완료: {config.SAMPLES_DIR} (mm_per_px={MM_PER_PX}, 마커 50px=10mm)")


if __name__ == "__main__":
    main()
