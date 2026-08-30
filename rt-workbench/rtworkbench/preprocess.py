"""필름 스캔 이미지 전처리 — 로드 / CLAHE 대비 향상 / 표시용 변환.

원칙 2(새 행동 요구 0): 입력은 기존 필름 스캔 이미지 파일 그대로다.
전처리는 화면 표시와 탐지 전단(2차 눈)을 돕는 보조일 뿐, 원본 픽셀은 보존한다.

구현 계약:
  load_grayscale(source)  — 경로(str|Path) 또는 bytes → 8-bit 그레이 ndarray(H,W)
  apply_clahe(img, ...)   — CLAHE 적용본 반환(입력 불변)
  to_display_rgb(img)     — 그레이(H,W) → RGB(H,W,3) uint8 (오버레이 그리기 용)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np

__all__ = ["load_grayscale", "apply_clahe", "to_display_rgb"]


def load_grayscale(source: Union[str, Path, bytes, bytearray]) -> np.ndarray:
    """파일 경로(str|Path) 또는 bytes 를 받아 8-bit 그레이스케일 ndarray(H,W)로 로드한다.

    컬러 입력은 그레이스케일로 변환된다(IMREAD_GRAYSCALE 디코딩으로 통일).
    16-bit 스캔 입력도 8-bit 로 축소된다.

    Raises:
        ValueError: 파일이 없거나, 타입이 지원되지 않거나, 디코딩에 실패했을 때.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {path}")
        # np.fromfile 경유 — 한글 등 비ASCII 경로에서도 안전하게 읽는다.
        raw = np.fromfile(str(path), dtype=np.uint8)
    elif isinstance(source, (bytes, bytearray)):
        raw = np.frombuffer(bytes(source), dtype=np.uint8)
    else:
        raise ValueError(
            f"지원하지 않는 입력 타입입니다: {type(source).__name__} (파일 경로 또는 bytes 필요)"
        )

    if raw.size == 0:
        raise ValueError("빈 입력입니다 — 이미지 데이터가 없습니다.")

    img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 디코딩할 수 없습니다 (지원 포맷: PNG/JPG/TIFF/BMP 등).")
    return img


def apply_clahe(img: np.ndarray, clip_limit: float = 3.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE(국소 적응 히스토그램 평활화)를 적용한 8-bit 그레이스케일을 반환한다.

    입력 배열은 수정하지 않는다(새 배열 반환). uint8 이 아닌 입력은 0~255 로
    정규화 후 처리한다(계약: 출력은 항상 8-bit).

    Raises:
        ValueError: 입력이 크기 0이 아닌 2차원 그레이스케일 ndarray 가 아닐 때.
            (크기 0 배열은 OpenCV CLAHE 내부에서 정지/오동작하므로 여기서 거부한다.)
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.size == 0:
        raise ValueError("apply_clahe 입력은 크기 0이 아닌 2차원 그레이스케일 ndarray 여야 합니다.")

    src = img
    if src.dtype != np.uint8:
        src = cv2.normalize(src.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid))
    )
    return clahe.apply(src)


def to_display_rgb(img: np.ndarray) -> np.ndarray:
    """그레이스케일(H,W) → RGB(H,W,3) uint8 변환 (bbox 오버레이 등 화면 표시용).

    이미 (H,W,3)인 입력은 uint8 복사본을 반환한다.

    Raises:
        ValueError: ndarray 가 아니거나 지원하지 않는 형상일 때.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("to_display_rgb 입력은 ndarray 여야 합니다.")

    src = img
    if src.dtype != np.uint8:
        src = np.clip(src, 0, 255).astype(np.uint8)

    if src.ndim == 2:
        return cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
    if src.ndim == 3 and src.shape[2] == 3:
        return src.copy()
    raise ValueError(f"지원하지 않는 이미지 형상입니다: {src.shape}")
