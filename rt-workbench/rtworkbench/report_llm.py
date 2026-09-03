"""소견서 초안 생성 + LLM 고유 기능 — LLM 어댑터 (로컬 LLM → [클라우드] → 오프라인 템플릿).

보안 설계(계획서 §2 기능 3): LLM에는 이미지가 아닌 **비식별 판정 결과 텍스트만** 전송한다.
build_payload()/build_history_payload()가 그 통로이며, 이미지/파일 경로/검사원 실명(inspector)을
포함하지 않는다. film_id/weld_id/block 은 업무 식별자이므로 포함한다.

LLM 보안 정책 — 로컬 우선 (local-first, config 참조):
  조선소/NDT 현장은 검사 텍스트의 외부 반출이 기본 금지다. 기본 정책은
  "로컬 LLM(OpenAI 호환 API — Ollama/vLLM) 또는 오프라인 템플릿"이며,
  외부 클라우드 API(Anthropic/Gemini)는 RTWB_ALLOW_CLOUD_LLM=1 로 명시 허용 시에만 호출한다.
  API 키가 있어도 허용 플래그가 없으면 절대 호출하지 않는다.

폴백 순서 (ReportWriter.generate):
  ① 캐시(payload sha256 키) → ② 로컬 LLM(OpenAI 호환 /chat/completions)
  → ③ Anthropic(클라우드 허용 + 키) → ④ Gemini REST(클라우드 허용 + 키)
  → ⑤ 오프라인 템플릿(항상 성공, 결정론적).
네트워크/API 예외는 절대 전파하지 않고 다음 백엔드로 폴백한다.
LLM 성공 시 결과를 캐시에 저장해 오프라인 데모에 대비한다(캐시 히트 시 source='cache').

LLM 고유 기능 (템플릿이 대체할 수 없는 것):
  - parse_search_query(): 한국어 자연어 질의 → Archive.search 필터 추출.
    LLM은 "필터 추출"만 하고, 검색 자체는 결정론적 Archive.search 가 수행한다.
    LLM 실패/차단 시 규칙 기반 파서(source='rule')로 폴백.
  - summarize_history(): 과거 검사 기록(비식별 요약)을 판독원이 읽기 좋은 한국어로 요약.
    LLM 실패/차단 시 통계 전용 템플릿(source='template')으로 폴백.

원칙 1: AI는 판정하지 않는다 — 모든 시스템 프롬프트에 판정 변경·수치 창작을 금지하고,
주어진 판정 결과를 정리/설명하는 역할만 맡긴다.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import tempfile
import threading
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

from rtworkbench import config
from rtworkbench.models import (
    DEFECT_TYPES,
    InspectionContext,
    InspectionRecord,
    Measurement,
    RuleVerdict,
)

# ------------------------------------------------------------------ 상수

# LLM 시스템 프롬프트 — 판정 권한이 없음을 명시한다 (원칙 1).
SYSTEM_PROMPT = (
    "너는 RT(방사선투과검사) 검사 소견서 작성 보조다. "
    "합부 판정을 바꾸거나 새로운 수치를 만들어내서는 절대 안 된다. "
    "입력으로 주어진 판정 결과(결함 유형, 크기, 허용 한계, 근거 조항, 합부)만을 사용하여 "
    "공식 RT 검사 소견서 문체의 한국어 문서로 정리하라. "
    "섹션 구성: 1. 검사 개요 / 2. 결함 소견 / 3. 판정 / 4. 비고. "
    "비고에는 '본 소견서 초안은 확정된 판정 결과로부터 자동 생성되었으며, "
    "합부 판정의 주체는 자격 판독원'이라는 취지의 문장을 반드시 포함하라."
)

# 자연어 검색 질의 → 필터 추출 프롬프트 (JSON only). {today}/{keys} 는 호출 시 채운다.
SEARCH_SYSTEM_PROMPT = (
    "너는 RT 검사 아카이브 검색 필터 추출기다. 사용자의 한국어 질의에서 검색 조건만 뽑아 "
    "JSON 객체 하나만 출력한다. 설명·코드블록·기타 텍스트는 절대 출력하지 않는다. "
    "판정(합부)을 바꾸거나 질의에 없는 조건·수치를 만들어내서는 절대 안 된다. "
    "스키마: {{\"block\": str, \"weld_id\": str, \"defect_type\": str, \"passed\": bool|null, "
    "\"text\": str, \"date_from\": \"YYYY-MM-DD\", \"date_to\": \"YYYY-MM-DD\"}}. "
    "해당 없는 항목은 빈 문자열(passed 는 null). "
    "block 은 'N번 블록' 형식. defect_type 은 다음 키 중 하나만 허용: {keys}. "
    "passed 는 합격→true, 불합격→false, 언급 없음→null. "
    "상대 날짜(어제/지난주/지난달/최근 N일 등)는 오늘 {today} 기준으로 ISO 날짜로 환산한다. "
    "text 는 필름 ID처럼 다른 항목에 넣을 수 없는 식별자/구절만 넣는다."
)

# 이력 요약 프롬프트 — 판정 변경·수치 창작 금지.
HISTORY_SYSTEM_PROMPT = (
    "너는 RT(방사선투과검사) 검사 이력 요약 보조다. "
    "합부 판정을 바꾸거나 새로운 수치를 만들어내서는 절대 안 된다. "
    "입력으로 주어진 집계와 기록 목록에 있는 숫자만 인용하고, 없는 수치는 언급하지 않는다. "
    "판독원이 빠르게 파악할 수 있도록 한국어로 간결하게(10줄 이내) 요약하라: "
    "결함 유형별 건수, 합격/불합격 비율, 반복되는 블록·용접부, "
    "주목할 불합격 항목(근거 조항 포함). 질문이 주어지면 그 관점에서 요약하되 "
    "주어진 데이터로 답할 수 없으면 '데이터 부족'이라고 명시하라."
)

# build_payload ↔ template_report 가 공유하는 페이로드 섹션 마커
_SEC_OVERVIEW = "검사 개요"
_SEC_DEFECTS = "결함별 판정"
_SEC_OVERALL = "종합 판정"

# 결함이 없을 때의 표준 서술 문구
NO_DEFECT_PHRASE = "유의미한 결함 지시 없음"

_GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)

# 로컬 LLM은 온프레미스 연결 — 환경변수 프록시(HTTP_PROXY 등)를 타지 않고 직접 연결한다.
_NO_PROXY: dict[str, None] = {"http": None, "https": None}

# 이력 요약 입력 상한 (프롬프트 길이/지연 제한)
HISTORY_MAX_RECORDS = 60

# 검색 필터 키 (Archive.search 시그니처와 동일)
SEARCH_FILTER_KEYS: tuple[str, ...] = (
    "block", "weld_id", "defect_type", "passed", "text", "date_from", "date_to",
)


# ------------------------------------------------------------------ 페이로드

def build_payload(
    context: InspectionContext,
    verdicts: list[RuleVerdict],
    measurements: list[Measurement],
) -> str:
    """비식별 판정 요약 텍스트 생성 (한국어).

    포함: 업무 식별자(film_id/weld_id/block), 이음 종류, 모재 두께, 품질 등급,
          결함별(유형/크기mm/허용한계mm/합부/근거 조항), 전체 합부.
    제외: 검사원 실명(inspector), 이미지 데이터, 파일 경로.
          LLM 전송의 유일한 통로인 이 함수에서 비식별 원칙을 강제한다.
    """
    measured = {m.defect_id: m for m in measurements}

    lines: list[str] = []
    lines.append(f"[{_SEC_OVERVIEW}]")
    lines.append(f"필름 ID: {context.film_id or '-'}")
    lines.append(f"블록: {context.block or '-'}")
    lines.append(f"용접부 ID: {context.weld_id or '-'}")
    lines.append(f"이음 종류: {context.joint_type or '-'}")
    lines.append(f"모재 두께: {context.thickness_mm:g} mm")
    lines.append(f"품질 등급: {context.quality_level or '-'}")
    lines.append(f"검사 기법: {context.technique or '-'}")

    lines.append("")
    lines.append(f"[{_SEC_DEFECTS}]")
    if verdicts:
        for i, v in enumerate(verdicts, 1):
            type_ko = DEFECT_TYPES.get(v.defect_type, v.defect_type)
            unit = getattr(v, "unit", "mm") or "mm"
            limit_txt = (
                "허용 불가 유형" if v.limit_mm is None else f"허용한계 {v.limit_mm:.2f} {unit}"
            )
            # 그룹 판정(누적 길이·투영 면적률)은 '합계(기공)' 라벨 + 해당 단위(mm/%)로 표기
            label = f"{i}. {v.display_id}" if getattr(v, "is_group", False) else f"{i}. {type_ko}"
            entry = (
                f"{label} — 크기 {v.size_mm:.2f} {unit}, {limit_txt}, "
                f"판정 {'합격' if v.passed else '불합격'}, 근거 조항 {v.clause}"
            )
            m = measured.get(v.defect_id)
            if m is not None and m.length_mm is not None:
                entry += f", 판독원 2클릭 측정 {m.length_mm:.2f} mm"
            lines.append(entry)
            if v.detail:
                lines.append(f"   근거: {v.detail}")
    else:
        lines.append(NO_DEFECT_PHRASE)

    n_fail = sum(1 for v in verdicts if not v.passed)
    lines.append("")
    lines.append(f"[{_SEC_OVERALL}]")
    if verdicts:
        lines.append(
            f"{'합격' if n_fail == 0 else '불합격'} "
            f"(판정 대상 결함 {len(verdicts)}건 중 불합격 {n_fail}건)"
        )
    else:
        lines.append(f"합격 ({NO_DEFECT_PHRASE})")

    return "\n".join(lines)


# ------------------------------------------------------------------ 정책/백엔드 (모듈 수준)

def _cloud_allowed() -> bool:
    """외부 클라우드 LLM 허용 여부 — 호출 시점에 config 를 읽는다 (테스트 monkeypatch 대응)."""
    return bool(getattr(config, "ALLOW_CLOUD_LLM", False))


def _local_base_url() -> str:
    return str(getattr(config, "LOCAL_LLM_URL", "") or "").strip().rstrip("/")


def _call_local(
    system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 1500
) -> str | None:
    """로컬 LLM(OpenAI 호환 /chat/completions) 호출. URL 없으면 스킵, 모든 예외는 삼키고 None."""
    base = _local_base_url()
    if not base:
        return None
    try:
        body = {
            "model": config.LOCAL_LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        r = requests.post(
            f"{base}/chat/completions",
            json=body,
            timeout=config.LLM_TIMEOUT_S,
            proxies=_NO_PROXY,
        )
        r.raise_for_status()
        data = r.json()
        text = str(data["choices"][0]["message"]["content"]).strip()
        return text or None
    except Exception:
        return None


def _call_anthropic(system: str, user: str, *, max_tokens: int = 1500) -> str | None:
    """Anthropic SDK 호출. 키 없으면 스킵(None), 모든 예외는 삼키고 None.

    주의: 클라우드 허용 게이트(_cloud_allowed)는 호출자(_call_llm/ReportWriter.generate)가
    적용한다. 이 함수는 게이트 없는 원시 백엔드다.
    """
    api_key = config.ANTHROPIC_API_KEY  # 런타임 조회 (테스트 monkeypatch 대응)
    if not api_key:
        return None
    try:
        import anthropic  # 지연 import — SDK 부재/문제 시에도 폴백 가능

        # 짧은 타임아웃/재시도 제한 — SDK 기본값(read 600s × 재시도 2회)이면
        # 스톨 시 UI가 수십 분 동결된다.
        client = anthropic.Anthropic(
            api_key=api_key, timeout=float(config.LLM_TIMEOUT_S), max_retries=0
        )
        resp = client.messages.create(
            model=config.ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        return text or None
    except Exception:
        return None


def _call_gemini(system: str, user: str) -> str | None:
    """Gemini REST(generateContent) 호출. 키 없으면 스킵, 모든 예외는 삼키고 None.

    주의: 클라우드 허용 게이트는 호출자가 적용한다 (_call_anthropic 과 동일).
    """
    api_key = config.GEMINI_API_KEY  # 런타임 조회 (테스트 monkeypatch 대응)
    if not api_key:
        return None
    try:
        url = _GEMINI_URL_TMPL.format(model=config.GEMINI_MODEL, key=api_key)
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
        }
        r = requests.post(url, json=body, timeout=config.LLM_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        text = str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
        return text or None
    except Exception:
        return None


def _call_llm(
    system: str, user: str, *, temperature: float = 0.2, max_tokens: int = 1500
) -> tuple[str, str] | tuple[None, None]:
    """정책 순서대로 LLM 호출 → (text, source) 또는 (None, None).

    로컬 → (클라우드 허용 시에만) Anthropic → Gemini. 예외는 전파하지 않는다.
    """
    text = _call_local(system, user, temperature=temperature, max_tokens=max_tokens)
    if text:
        return text, "local"
    if _cloud_allowed():
        text = _call_anthropic(system, user, max_tokens=max_tokens)
        if text:
            return text, "claude"
        text = _call_gemini(system, user)
        if text:
            return text, "gemini"
    return None, None


def llm_policy_status() -> dict[str, Any]:
    """현재 LLM 정책/백엔드 상태 (UI 표시용). 네트워크 예외는 전파하지 않는다.

    local_reachable: GET {LOCAL_LLM_URL}/models (1.5s) 성공 여부. URL 미설정이면 None.
    effective_order: 실제로 시도되는 백엔드 순서 (클라우드는 허용 + 키가 있을 때만 포함).
    """
    cloud_allowed = _cloud_allowed()
    local_url = _local_base_url()
    local_model = str(getattr(config, "LOCAL_LLM_MODEL", "") or "")
    anthropic_key = bool(config.ANTHROPIC_API_KEY)
    gemini_key = bool(config.GEMINI_API_KEY)

    local_reachable: bool | None
    if not local_url:
        local_reachable = None
    else:
        try:
            r = requests.get(f"{local_url}/models", timeout=1.5, proxies=_NO_PROXY)
            local_reachable = bool(r.ok)
        except Exception:
            local_reachable = False

    order: list[str] = ["cache"]
    if local_url:
        order.append("local")
    if cloud_allowed and anthropic_key:
        order.append("claude")
    if cloud_allowed and gemini_key:
        order.append("gemini")
    order.append("template")

    return {
        "cloud_allowed": cloud_allowed,
        "local_url": local_url,
        "local_model": local_model,
        "local_reachable": local_reachable,
        "anthropic_key": anthropic_key,
        "gemini_key": gemini_key,
        "effective_order": order,
    }


# ------------------------------------------------------------------ 생성기

class ReportWriter:
    """소견서 초안 생성기 — 캐시 → 로컬 LLM → [Claude → Gemini] → 오프라인 템플릿 폴백."""

    def __init__(self, cache_path: Path = config.LLM_CACHE_PATH):
        self.cache_path = Path(cache_path)

    # ---------------------------------------------------------- 공개 API

    def generate(self, payload: str) -> tuple[str, str]:
        """소견서 초안 생성 → (report_text, source).

        source ∈ {'cache', 'local', 'claude', 'gemini', 'template'}.
        어떤 경우에도 예외를 전파하지 않는다 — 템플릿 백엔드는 절대 실패하지 않는다.
        클라우드(Claude/Gemini)는 config.ALLOW_CLOUD_LLM 이 True 일 때만 시도한다.
        """
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # ① 캐시 히트 (사전 캐시로 오프라인 데모에서도 LLM 문체 재현 가능)
        entry = self._load_cache().get(key)
        if isinstance(entry, dict) and entry.get("text"):
            return str(entry["text"]), "cache"

        # ② 로컬 LLM (온프레미스 — 기본 경로)
        text = self._try_local(payload)
        if text:
            self._save_cache(key, text, "local")
            return text, "local"

        # ③④ 외부 클라우드 — 명시 허용(RTWB_ALLOW_CLOUD_LLM=1) 시에만
        if _cloud_allowed():
            text = self._try_anthropic(payload)
            if text:
                self._save_cache(key, text, "claude")
                return text, "claude"

            text = self._try_gemini(payload)
            if text:
                self._save_cache(key, text, "gemini")
                return text, "gemini"

        # ⑤ 오프라인 템플릿 — 항상 성공 (LLM 성공이 아니므로 캐시에 저장하지 않음)
        return template_report(payload), "template"

    # ---------------------------------------------------------- 캐시

    def _load_cache(self) -> dict[str, Any]:
        """캐시 파일(JSON dict: sha256 → {text, source}) 로드. 실패 시 빈 dict."""
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # read-modify-write 구간 보호 (같은 프로세스의 동시 세션 lost-update 방지)
    _cache_lock = threading.Lock()

    def _save_cache(self, key: str, text: str, source: str) -> None:
        """LLM 성공 결과를 캐시에 저장. 저장 실패는 치명적이지 않으므로 삼킨다.

        임시 파일에 기록 후 os.replace 로 원자적 교체 — 도중 중단되어도
        기존 캐시 파일이 절단된 불완전 JSON으로 남지 않는다.
        """
        try:
            with self._cache_lock:
                cache = self._load_cache()
                cache[key] = {"text": text, "source": source}
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(
                    dir=self.cache_path.parent, prefix=self.cache_path.name, suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(cache, f, ensure_ascii=False, indent=2)
                    os.replace(tmp_path, self.cache_path)
                except BaseException:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
        except Exception:
            pass

    # ---------------------------------------------------------- 백엔드 (원시 — 게이트는 generate 가 적용)

    def _try_local(self, payload: str) -> str | None:
        """로컬 LLM(OpenAI 호환) 호출. URL 없으면 스킵, 모든 예외는 삼키고 None."""
        return _call_local(SYSTEM_PROMPT, payload, temperature=0.2, max_tokens=1500)

    def _try_anthropic(self, payload: str) -> str | None:
        """Anthropic SDK 호출. 키 없으면 스킵(None), 모든 예외는 삼키고 None."""
        return _call_anthropic(SYSTEM_PROMPT, payload, max_tokens=1500)

    def _try_gemini(self, payload: str) -> str | None:
        """Gemini REST(generateContent) 호출. 키 없으면 스킵, 모든 예외는 삼키고 None."""
        return _call_gemini(SYSTEM_PROMPT, payload)


# ------------------------------------------------------------------ 오프라인 템플릿

def _split_sections(payload: str) -> dict[str, list[str]]:
    """'[섹션명]' 마커 기준으로 페이로드를 섹션별 줄 목록으로 분해."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in payload.splitlines():
        line = raw.strip()
        m = re.fullmatch(r"\[(.+)\]", line)
        if m:
            current = m.group(1)
            sections.setdefault(current, [])
        elif current is not None and line:
            sections[current].append(line)
    return sections


def template_report(payload: str) -> str:
    """결정론적 오프라인 템플릿 — 검사 리포트 문체의 한국어 소견서.

    LLM 없이도 데모가 100% 동작하도록 절대 실패하지 않는다.
    섹션: 1. 검사 개요 / 2. 결함 소견 / 3. 판정 / 4. 비고.
    """
    sections = _split_sections(payload)
    overview = sections.get(_SEC_OVERVIEW, [])
    defects = sections.get(_SEC_DEFECTS, [])
    overall = sections.get(_SEC_OVERALL, [])
    has_defects = bool(defects) and not any(NO_DEFECT_PHRASE in line for line in defects)

    lines: list[str] = []
    lines.append("RT 판독 소견서 (초안)")
    lines.append("=" * 40)
    lines.append("")

    lines.append("1. 검사 개요")
    if overview:
        lines.extend(f"  - {line}" for line in overview)
    else:
        # 섹션 마커가 없는 임의 페이로드도 소견서 형태로 감싼다 (절대 실패 금지)
        lines.append("  - 아래 판정 요약 원문 참조.")
        lines.extend(f"    {line.strip()}" for line in payload.splitlines() if line.strip())
    lines.append("")

    lines.append("2. 결함 소견")
    if has_defects:
        lines.append(
            "  본 검사에서 관찰된 결함 지시 및 결정론적 룰 엔진의 항목별 판정 결과는 다음과 같다."
        )
        lines.extend(f"  {line}" for line in defects)
    else:
        lines.append(f"  본 검사 범위 내에서 {NO_DEFECT_PHRASE}이 확인되었다.")
    lines.append("")

    lines.append("3. 판정")
    if overall:
        lines.extend(f"  종합 판정: {line}" for line in overall)
    else:
        lines.append("  종합 판정 정보가 요약에 포함되지 않았다. 판독원의 확인이 필요하다.")
    lines.append(
        "  상기 합부 판정은 적용 기준표에 따른 결정론적 룰 엔진 산출 결과이며, "
        "자격 판독원의 검토·확정을 전제로 한다."
    )
    lines.append("")

    lines.append("4. 비고")
    lines.append(
        "  - 본 소견서 초안은 확정된 판정 결과로부터 자동 생성되었으며, "
        "합부 판정의 주체는 자격 판독원임."
    )
    lines.append(
        "  - 본 판정에 적용된 기준표는 ISO 5817 계열의 구조를 모사한 데모 기준으로, "
        "실제 생산 검사 적용 전 선급 NDT 지침에 따른 기준표 교체·검증이 필요함."
    )
    return "\n".join(lines)


# ==================================================================
# LLM 고유 기능 1 — 자연어 검색 질의 → Archive.search 필터
# ==================================================================

# 결함 유형 별칭 (긴 것부터 매칭 — '군집 기공' 이 '기공' 보다 먼저)
_DEFECT_ALIASES: list[tuple[str, str]] = [
    ("군집 기공", "cluster_porosity"),
    ("군집기공", "cluster_porosity"),
    ("클러스터 기공", "cluster_porosity"),
    ("슬래그 개재물", "slag_inclusion"),
    ("슬래그개재물", "slag_inclusion"),
    ("슬래그", "slag_inclusion"),
    ("융합불량", "lack_of_fusion"),
    ("융합 불량", "lack_of_fusion"),
    ("용입부족", "incomplete_penetration"),
    ("용입 부족", "incomplete_penetration"),
    ("언더컷", "undercut"),
    ("균열", "crack"),
    ("크랙", "crack"),
    ("기공", "porosity"),
    ("미분류 지시", "unknown"),
    ("미분류", "unknown"),
]

_KO_TO_KEY: dict[str, str] = {v: k for k, v in DEFECT_TYPES.items()}

_RE_BLOCK_A = re.compile(r"(\d{1,4})\s*번\s*블[록럭]")  # "3번 블록"
_RE_BLOCK_B = re.compile(r"블[록럭]\s*(\d{1,4})")  # "블록 3"
# 식별자 토큰: BL3-V-012, W-104, F-2026-0830-01 (앞뒤가 영숫자/하이픈이 아닐 것)
_RE_ID_TOKEN = re.compile(r"(?<![\w-])[A-Za-z]{1,4}\d{0,4}(?:-[A-Za-z0-9]{1,6})+(?![\w-])")
_RE_ISO_DATE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_RE_YM_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})(?![\d-])")
_RE_YM_KO = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
_RE_Y_KO = re.compile(r"(\d{4})\s*년(?!\s*\d{1,2}\s*월)")
_RE_M_KO = re.compile(r"(?<![\d년\s])\s*(?<!\d)(\d{1,2})\s*월")
_RE_RECENT = re.compile(r"(?:최근|지난)\s*(\d{1,3})\s*(일|주|개월|달)")
_RE_QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{1,60})[\"'“”‘’]")


def _empty_filters() -> dict[str, Any]:
    return {
        "block": "",
        "weld_id": "",
        "defect_type": "",
        "passed": None,
        "text": "",
        "date_from": "",
        "date_to": "",
    }


def _month_range(y: int, m: int) -> tuple[str, str]:
    last = calendar.monthrange(y, m)[1]
    return date(y, m, 1).isoformat(), date(y, m, last).isoformat()


def _rule_dates(q: str, today: date) -> tuple[str, str]:
    """상대/절대 날짜 표현 → (date_from, date_to) ISO. 인식 실패 시 ('', '')."""
    compact = q.replace(" ", "")

    def _plausible_year(y: int) -> bool:
        return 2000 <= y <= 2099

    # 절대 ISO 날짜 (1개: 당일, 2개 이상: min~max)
    iso = _RE_ISO_DATE.findall(q)
    if iso:
        ds: list[date] = []
        for y, m, d in iso:
            try:
                if _plausible_year(int(y)):
                    ds.append(date(int(y), int(m), int(d)))
            except ValueError:
                continue
        if ds:
            return min(ds).isoformat(), max(ds).isoformat()

    # YYYY-MM / YYYY년 M월
    m_ = _RE_YM_ISO.search(q) or _RE_YM_KO.search(q)
    if m_:
        y, mo = int(m_.group(1)), int(m_.group(2))
        if _plausible_year(y) and 1 <= mo <= 12:
            return _month_range(y, mo)

    # YYYY년 (월 없음)
    m_ = _RE_Y_KO.search(q)
    if m_:
        y = int(m_.group(1))
        if _plausible_year(y):
            return date(y, 1, 1).isoformat(), date(y, 12, 31).isoformat()

    # 최근/지난 N일|주|개월
    m_ = _RE_RECENT.search(q)
    if m_:
        n, unit = int(m_.group(1)), m_.group(2)
        days = n if unit == "일" else n * 7 if unit == "주" else n * 30
        return (today - timedelta(days=days)).isoformat(), today.isoformat()

    if "오늘" in compact or "금일" in compact:
        return today.isoformat(), today.isoformat()
    if "어제" in compact or "전일" in compact:
        y = today - timedelta(days=1)
        return y.isoformat(), y.isoformat()
    if "지난주" in compact or "전주" in compact or "저번주" in compact:
        monday = today - timedelta(days=today.weekday() + 7)
        return monday.isoformat(), (monday + timedelta(days=6)).isoformat()
    if "이번주" in compact or "금주" in compact:
        monday = today - timedelta(days=today.weekday())
        return monday.isoformat(), (monday + timedelta(days=6)).isoformat()
    if "지난달" in compact or "전월" in compact or "저번달" in compact or "지난월" in compact:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        return _month_range(last_prev.year, last_prev.month)
    if "이번달" in compact or "금월" in compact or "이달" in compact:
        return _month_range(today.year, today.month)
    if "작년" in compact or "전년" in compact:
        return date(today.year - 1, 1, 1).isoformat(), date(today.year - 1, 12, 31).isoformat()
    if "올해" in compact or "금년" in compact:
        return date(today.year, 1, 1).isoformat(), today.isoformat()

    # "8월" 단독 → 올해의 해당 월
    m_ = _RE_M_KO.search(q)
    if m_:
        mo = int(m_.group(1))
        if 1 <= mo <= 12:
            return _month_range(today.year, mo)

    return "", ""


def _rule_parse_query(q: str, today: date) -> dict[str, Any]:
    """결정론적 규칙 기반 파서 — LLM 없이도 항상 동작한다."""
    f = _empty_filters()
    q = (q or "").strip()
    if not q:
        return f

    # 블록
    m = _RE_BLOCK_A.search(q) or _RE_BLOCK_B.search(q)
    if m:
        f["block"] = f"{int(m.group(1))}번 블록"

    # 식별자 토큰: 2~3세그먼트 + 마지막이 숫자 → 용접부 ID, 그 외(필름 ID 등) → text
    text_tokens: list[str] = []
    for tok in _RE_ID_TOKEN.findall(q):
        segs = tok.split("-")
        if 2 <= len(segs) <= 3 and segs[-1].isdigit() and not f["weld_id"]:
            f["weld_id"] = tok.upper()
        else:
            text_tokens.append(tok)

    # 결함 유형 (긴 별칭 우선)
    for alias, key in _DEFECT_ALIASES:
        if alias in q:
            f["defect_type"] = key
            break

    # 합부 ('불합격' → False 를 먼저 검사)
    if any(w in q for w in ("불합격", "부적합", "리젝", "reject", "REJECT", "불량 판정")):
        f["passed"] = False
    elif any(w in q for w in ("합격", "적합", "패스", "pass", "PASS")):
        f["passed"] = True

    # 날짜 — 식별자 토큰(F-2026-0830-01 등)의 숫자가 날짜로 오인되지 않도록 제거 후 파싱
    f["date_from"], f["date_to"] = _rule_dates(_RE_ID_TOKEN.sub(" ", q), today)

    # 자유 텍스트: 따옴표 구절 > 필름 ID 류 토큰
    quoted = _RE_QUOTED.search(q)
    if quoted:
        f["text"] = quoted.group(1).strip()
    elif text_tokens:
        f["text"] = text_tokens[0]

    return f


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """LLM 응답에서 첫 '{' ~ 마지막 '}' 구간을 JSON 객체로 파싱. 실패 시 None."""
    if not text:
        return None
    s = text.strip()
    # ```json ... ``` 코드블록 제거
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.MULTILINE)
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _valid_iso_date(v: Any) -> str:
    if not isinstance(v, str):
        return ""
    v = v.strip()[:10]
    try:
        return date.fromisoformat(v).isoformat()
    except ValueError:
        return ""


def _sanitize_filters(obj: dict[str, Any]) -> dict[str, Any]:
    """LLM 출력 → Archive.search 필터로 정규화. 허용 키/값만 남긴다 (LLM 은 필터 추출만)."""
    f = _empty_filters()

    def _s(v: Any, limit: int = 60) -> str:
        if v is None or isinstance(v, bool):
            return ""
        return str(v).strip()[:limit]

    f["block"] = _s(obj.get("block"))
    m = _RE_BLOCK_A.search(f["block"]) or _RE_BLOCK_B.search(f["block"])
    if m:  # 'N번 블록' 형식으로 정규화
        f["block"] = f"{int(m.group(1))}번 블록"
    elif f["block"].isdigit():
        f["block"] = f"{int(f['block'])}번 블록"

    f["weld_id"] = _s(obj.get("weld_id"))
    f["text"] = _s(obj.get("text"), 120)

    dt = _s(obj.get("defect_type"))
    if dt in DEFECT_TYPES:
        f["defect_type"] = dt
    elif dt in _KO_TO_KEY:
        f["defect_type"] = _KO_TO_KEY[dt]
    else:
        for alias, key in _DEFECT_ALIASES:
            if dt and alias in dt:
                f["defect_type"] = key
                break

    p = obj.get("passed")
    if isinstance(p, bool):
        f["passed"] = p
    elif isinstance(p, str):
        pl = p.strip().lower()
        if pl in ("true", "합격", "pass"):
            f["passed"] = True
        elif pl in ("false", "불합격", "fail"):
            f["passed"] = False

    f["date_from"] = _valid_iso_date(obj.get("date_from"))
    f["date_to"] = _valid_iso_date(obj.get("date_to"))
    if f["date_from"] and f["date_to"] and f["date_from"] > f["date_to"]:
        f["date_from"], f["date_to"] = f["date_to"], f["date_from"]
    return f


def _has_any_filter(f: dict[str, Any]) -> bool:
    return any(f.get(k) not in ("", None) for k in SEARCH_FILTER_KEYS)


def parse_search_query(
    nl_query: str, *, today: date | None = None
) -> tuple[dict[str, Any], str]:
    """한국어 자연어 질의 → Archive.search kwargs 추출 → (filters, source).

    LLM(로컬 → 클라우드 허용 시)에 JSON 필터 추출만 맡기고, 실패/차단/무응답 시
    결정론적 규칙 파서로 폴백한다(source='rule'). 검색 자체는 이 함수 밖에서
    Archive.search 가 결정론적으로 수행한다 — LLM 은 필터 추출 이상을 결정하지 않는다.

    today: 상대 날짜(어제/지난달/최근 N일) 환산 기준일 (테스트 결정성용, 기본 오늘).
    """
    today = today or date.today()
    q = (nl_query or "").strip()
    if not q:
        return _empty_filters(), "rule"

    try:
        system = SEARCH_SYSTEM_PROMPT.format(
            today=today.isoformat(), keys=", ".join(DEFECT_TYPES.keys())
        )
        text, source = _call_llm(system, q, temperature=0.0, max_tokens=400)
        if text and source:
            obj = _extract_json_object(text)
            if obj is not None:
                filters = _sanitize_filters(obj)
                if _has_any_filter(filters):
                    return filters, source
    except Exception:
        pass

    return _rule_parse_query(q, today), "rule"


# ==================================================================
# LLM 고유 기능 2 — 검사 이력 요약
# ==================================================================

def _history_stats(records: list[InspectionRecord]) -> dict[str, Any]:
    """결정론적 집계 — 유형별 건수, 합부, 반복 블록/용접부, 불합격 항목."""
    n = len(records)
    n_pass = sum(1 for r in records if r.overall_passed)
    type_counts: Counter[str] = Counter()
    fail_type_counts: Counter[str] = Counter()
    blocks: Counter[str] = Counter()
    welds: Counter[str] = Counter()
    failed_items: list[dict[str, Any]] = []
    dates: list[str] = []

    for r in records:
        ctx = r.context
        if ctx.block:
            blocks[ctx.block] += 1
        if ctx.weld_id:
            welds[ctx.weld_id] += 1
        if r.created_at:
            dates.append(r.created_at[:10])
        for c in r.candidates:
            if c.status == "accepted":
                type_counts[DEFECT_TYPES.get(c.defect_type, c.defect_type)] += 1
        for v in r.verdicts:
            if not v.passed:
                ko = DEFECT_TYPES.get(v.defect_type, v.defect_type)
                fail_type_counts[ko] += 1
                failed_items.append(
                    {
                        "date": r.created_at[:10],
                        "block": ctx.block or "-",
                        "weld_id": ctx.weld_id or "-",
                        "type": ko,
                        "size_mm": v.size_mm,
                        "limit_mm": v.limit_mm,
                        "clause": v.clause,
                    }
                )

    return {
        "total": n,
        "passed": n_pass,
        "failed": n - n_pass,
        "type_counts": dict(type_counts.most_common()),
        "fail_type_counts": dict(fail_type_counts.most_common()),
        "repeated_blocks": {b: c for b, c in blocks.most_common() if c >= 2},
        "repeated_welds": {w: c for w, c in welds.most_common() if c >= 2},
        "failed_items": failed_items,
        "date_min": min(dates) if dates else "",
        "date_max": max(dates) if dates else "",
    }


def build_history_payload(records: list[InspectionRecord]) -> str:
    """이력 요약용 비식별 입력 텍스트 — 검사원 실명·이미지 파일명·소견서 본문 제외.

    소견서 본문은 판독원이 자유 편집한 것이므로(실명 포함 가능) 전송하지 않는다.
    """
    recs = list(records)[:HISTORY_MAX_RECORDS]
    st_ = _history_stats(recs)
    lines: list[str] = []
    lines.append("[집계]")
    lines.append(f"총 {st_['total']}건, 합격 {st_['passed']}건, 불합격 {st_['failed']}건")
    if st_["date_min"]:
        lines.append(f"기간: {st_['date_min']} ~ {st_['date_max']}")
    if st_["type_counts"]:
        lines.append(
            "결함 유형별(채택): " + ", ".join(f"{k} {v}건" for k, v in st_["type_counts"].items())
        )
    if st_["fail_type_counts"]:
        lines.append(
            "불합격 결함 유형별: "
            + ", ".join(f"{k} {v}건" for k, v in st_["fail_type_counts"].items())
        )
    if st_["repeated_blocks"]:
        lines.append(
            "반복 블록: " + ", ".join(f"{k} {v}회" for k, v in st_["repeated_blocks"].items())
        )
    if st_["repeated_welds"]:
        lines.append(
            "반복 용접부: " + ", ".join(f"{k} {v}회" for k, v in st_["repeated_welds"].items())
        )

    lines.append("")
    lines.append("[기록 목록]")
    for r in recs:
        ctx = r.context
        parts = [
            r.created_at[:10] or "-",
            f"필름 {ctx.film_id or '-'}",
            f"블록 {ctx.block or '-'}",
            f"용접부 {ctx.weld_id or '-'}",
            f"등급 {ctx.quality_level or '-'}",
            "합격" if r.overall_passed else "불합격",
        ]
        if r.verdicts:
            vs = []
            for v in r.verdicts:
                ko = DEFECT_TYPES.get(v.defect_type, v.defect_type)
                lim = "허용불가" if v.limit_mm is None else f"한계 {v.limit_mm:.2f}mm"
                vs.append(
                    f"{ko} {v.size_mm:.2f}mm/{lim}/{'합' if v.passed else '불'}/{v.clause}"
                )
            parts.append("결함: " + "; ".join(vs))
        else:
            accepted = [
                DEFECT_TYPES.get(c.defect_type, c.defect_type)
                for c in r.candidates if c.status == "accepted"
            ]
            parts.append("결함: " + (", ".join(accepted) if accepted else "없음"))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def template_history_summary(records: list[InspectionRecord], question: str = "") -> str:
    """결정론적 통계 전용 요약 (오프라인 폴백) — 절대 실패하지 않는다."""
    recs = list(records)[:HISTORY_MAX_RECORDS]
    if not recs:
        return "요약할 기록이 없습니다."
    st_ = _history_stats(recs)
    n = st_["total"]
    lines: list[str] = []
    lines.append("검사 이력 요약 (통계 템플릿)")
    if st_["date_min"]:
        lines.append(f"- 기간: {st_['date_min']} ~ {st_['date_max']}")
    rate = (st_["passed"] / n * 100.0) if n else 0.0
    lines.append(
        f"- 총 {n}건 — 합격 {st_['passed']}건, 불합격 {st_['failed']}건 (합격률 {rate:.0f}%)"
    )
    if st_["type_counts"]:
        lines.append(
            "- 결함 유형별(채택): "
            + ", ".join(f"{k} {v}건" for k, v in st_["type_counts"].items())
        )
    else:
        lines.append("- 채택된 결함 없음")
    if st_["fail_type_counts"]:
        lines.append(
            "- 불합격 결함 유형별: "
            + ", ".join(f"{k} {v}건" for k, v in st_["fail_type_counts"].items())
        )
    if st_["repeated_blocks"]:
        lines.append(
            "- 반복 블록: " + ", ".join(f"{k} {v}회" for k, v in st_["repeated_blocks"].items())
        )
    if st_["repeated_welds"]:
        lines.append(
            "- 반복 용접부: " + ", ".join(f"{k} {v}회" for k, v in st_["repeated_welds"].items())
        )
    if st_["failed_items"]:
        lines.append("- 주요 불합격 항목:")
        for it in st_["failed_items"][:10]:
            lim = "허용 불가" if it["limit_mm"] is None else f"한계 {it['limit_mm']:.2f}mm"
            lines.append(
                f"  · {it['date']} {it['block']} / {it['weld_id']} — "
                f"{it['type']} {it['size_mm']:.2f}mm ({lim}), 조항 {it['clause']}"
            )
    if question.strip():
        lines.append(f"- 질문 '{question.strip()}' 에 대한 서술형 답변은 LLM 미사용으로 생략 (통계만 제공).")
    lines.append("- 본 요약은 확정된 판정 기록의 집계이며 합부를 변경하지 않는다.")
    return "\n".join(lines)


def summarize_history(
    records: list[InspectionRecord], question: str = ""
) -> tuple[str, str]:
    """과거 검사 기록 요약 → (summary_text, source).

    source ∈ {'local', 'claude', 'gemini', 'template'}.
    LLM 에는 비식별 입력(build_history_payload)만 전송하며 최대 HISTORY_MAX_RECORDS 건.
    LLM 실패/차단 시 통계 전용 템플릿으로 폴백 — 어떤 경우에도 예외를 전파하지 않는다.
    """
    recs = [r for r in list(records) if r is not None][:HISTORY_MAX_RECORDS]
    if not recs:
        return template_history_summary(recs, question), "template"

    try:
        user = build_history_payload(recs)
        if question.strip():
            user += f"\n\n[질문]\n{question.strip()[:300]}"
        text, source = _call_llm(HISTORY_SYSTEM_PROMPT, user, temperature=0.2, max_tokens=900)
        if text and source:
            return text, source
    except Exception:
        pass

    return template_history_summary(recs, question), "template"
