"""소견서 초안 생성 — LLM 어댑터 (Claude → Gemini → 오프라인 템플릿).

보안 설계(계획서 §2 기능 3): LLM에는 이미지가 아닌 **비식별 판정 결과 텍스트만** 전송한다.
build_payload()가 그 유일한 통로이며, 이미지/파일 경로/검사원 실명(inspector)을 포함하지 않는다.
film_id/weld_id/block 은 업무 식별자이므로 포함한다.

폴백 순서 (ReportWriter.generate):
  ① 캐시(payload sha256 키) → ② Anthropic(키 있으면) → ③ Gemini REST(키 있으면)
  → ④ 오프라인 템플릿(항상 성공, 결정론적).
네트워크/API 예외는 절대 전파하지 않고 다음 백엔드로 폴백한다.
API 성공 시 결과를 캐시에 저장해 오프라인 데모에 대비한다(캐시 히트 시 source='cache').

원칙 1: AI는 판정하지 않는다 — 시스템 프롬프트로 판정 변경·수치 창작을 금지하고,
주어진 판정 결과를 공식 RT 검사 소견서 문체(한국어)로 정리하는 역할만 맡긴다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

import requests

from rtworkbench import config
from rtworkbench.models import DEFECT_TYPES, InspectionContext, Measurement, RuleVerdict

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

# build_payload ↔ template_report 가 공유하는 페이로드 섹션 마커
_SEC_OVERVIEW = "검사 개요"
_SEC_DEFECTS = "결함별 판정"
_SEC_OVERALL = "종합 판정"

# 결함이 없을 때의 표준 서술 문구
NO_DEFECT_PHRASE = "유의미한 결함 지시 없음"

_GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
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
            limit_txt = (
                "허용 불가 유형" if v.limit_mm is None else f"허용한계 {v.limit_mm:.2f} mm"
            )
            entry = (
                f"{i}. {type_ko} — 크기 {v.size_mm:.2f} mm, {limit_txt}, "
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


# ------------------------------------------------------------------ 생성기

class ReportWriter:
    """소견서 초안 생성기 — 캐시 → Claude → Gemini → 오프라인 템플릿 폴백."""

    def __init__(self, cache_path: Path = config.LLM_CACHE_PATH):
        self.cache_path = Path(cache_path)

    # ---------------------------------------------------------- 공개 API

    def generate(self, payload: str) -> tuple[str, str]:
        """소견서 초안 생성 → (report_text, source).

        source ∈ {'claude', 'gemini', 'template', 'cache'}.
        어떤 경우에도 예외를 전파하지 않는다 — 템플릿 백엔드는 절대 실패하지 않는다.
        """
        key = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        # ① 캐시 히트 (사전 캐시로 오프라인 데모에서도 LLM 문체 재현 가능)
        entry = self._load_cache().get(key)
        if isinstance(entry, dict) and entry.get("text"):
            return str(entry["text"]), "cache"

        # ② Anthropic (키 없으면 스킵, 예외는 폴백)
        text = self._try_anthropic(payload)
        if text:
            self._save_cache(key, text, "claude")
            return text, "claude"

        # ③ Gemini REST (키 없으면 스킵, 예외는 폴백)
        text = self._try_gemini(payload)
        if text:
            self._save_cache(key, text, "gemini")
            return text, "gemini"

        # ④ 오프라인 템플릿 — 항상 성공 (API 성공이 아니므로 캐시에 저장하지 않음)
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
        """API 성공 결과를 캐시에 저장. 저장 실패는 치명적이지 않으므로 삼킨다.

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

    # ---------------------------------------------------------- 백엔드

    def _try_anthropic(self, payload: str) -> str | None:
        """Anthropic SDK 호출. 키 없으면 스킵(None), 모든 예외는 삼키고 None."""
        api_key = config.ANTHROPIC_API_KEY  # 런타임 조회 (테스트 monkeypatch 대응)
        if not api_key:
            return None
        try:
            import anthropic  # 지연 import — SDK 부재/문제 시에도 폴백 가능

            # Gemini 경로(timeout=20)와 동급의 짧은 타임아웃/재시도 제한 —
            # 기본값(read 600s × 재시도 2회)이면 스톨 시 UI가 수십 분 동결된다.
            client = anthropic.Anthropic(api_key=api_key, timeout=20.0, max_retries=0)
            resp = client.messages.create(
                model=config.ANTHROPIC_MODEL,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": payload}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()
            return text or None
        except Exception:
            return None

    def _try_gemini(self, payload: str) -> str | None:
        """Gemini REST(generateContent) 호출. 키 없으면 스킵, 모든 예외는 삼키고 None."""
        api_key = config.GEMINI_API_KEY  # 런타임 조회 (테스트 monkeypatch 대응)
        if not api_key:
            return None
        try:
            url = _GEMINI_URL_TMPL.format(model=config.GEMINI_MODEL, key=api_key)
            body = {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "contents": [{"role": "user", "parts": [{"text": payload}]}],
            }
            r = requests.post(url, json=body, timeout=20)
            r.raise_for_status()
            data = r.json()
            text = str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
            return text or None
        except Exception:
            return None


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
