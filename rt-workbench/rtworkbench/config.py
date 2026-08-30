"""RT 판독 워크벤치 — 경로/환경 설정."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("RTWB_DATA_DIR", PROJECT_DIR / "data"))
SAMPLES_DIR = DATA_DIR / "samples"
EXPORTS_DIR = DATA_DIR / "exports"
DB_PATH = Path(os.environ.get("RTWB_DB_PATH", DATA_DIR / "rt_workbench.db"))
LLM_CACHE_PATH = DATA_DIR / "llm_cache.json"

CRITERIA_PATH = PACKAGE_DIR / "criteria" / "demo_iso5817_like.json"

# YOLO 가중치가 이 경로에 있으면 YoloDetector가 사용된다. 없으면 CV 폴백 탐지기.
YOLO_WEIGHTS_PATH = Path(os.environ.get("RTWB_YOLO_WEIGHTS", PROJECT_DIR / "weights" / "best.pt"))

# LLM API 키 (없으면 오프라인 템플릿 백엔드로 자동 폴백)
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

ANTHROPIC_MODEL = os.environ.get("RTWB_ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_MODEL = os.environ.get("RTWB_GEMINI_MODEL", "gemini-2.0-flash")


def ensure_dirs() -> None:
    for p in (DATA_DIR, SAMPLES_DIR, EXPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
