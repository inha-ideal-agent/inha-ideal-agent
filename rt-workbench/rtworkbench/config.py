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

# ---------------------------------------------------------------------------
# LLM 보안 정책 — 로컬 우선 (local-first)
#
# 조선소/NDT 현장은 검사 텍스트의 외부 반출이 기본적으로 허용되지 않는다.
# 따라서 기본 정책은 "로컬 LLM(OpenAI 호환 API: Ollama/vLLM 등) 또는 오프라인 템플릿"이며,
# 외부 클라우드 API(Anthropic/Gemini)는 RTWB_ALLOW_CLOUD_LLM=1 로 명시 허용해야만 호출된다.
# API 키가 설정되어 있어도 허용 플래그가 없으면 절대 호출하지 않는다.
# ---------------------------------------------------------------------------

# OpenAI 호환 base URL (…/v1). Ollama 기본값: http://localhost:11434/v1
LOCAL_LLM_URL = os.environ.get("RTWB_LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.environ.get("RTWB_LOCAL_LLM_MODEL", "exaone3.5:7.8b")
# 외부 클라우드 LLM 허용 여부 — 기본 False(차단)
ALLOW_CLOUD_LLM = os.environ.get("RTWB_ALLOW_CLOUD_LLM", "0") == "1"
# LLM 호출 타임아웃(초) — 스톨 시 UI 동결 방지
LLM_TIMEOUT_S = 20


def ensure_dirs() -> None:
    for p in (DATA_DIR, SAMPLES_DIR, EXPORTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
