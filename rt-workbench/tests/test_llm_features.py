"""LLM 정책(로컬 우선·클라우드 차단) + LLM 고유 기능(자연어 검색 필터·이력 요약) 테스트.

네트워크 호출 없이 검증한다: requests.post/get 을 monkeypatch 로 가로채고(기본 = 연결 거부),
anthropic SDK 는 생성자 호출을 플래그로 기록하는 가짜 모듈로 대체한다.
"""

from __future__ import annotations

import importlib
import sys
import types
from datetime import date, timedelta
from pathlib import Path

import pytest
import requests

from rtworkbench import config, report_llm
from rtworkbench.models import (
    DefectCandidate,
    InspectionContext,
    InspectionRecord,
    RuleVerdict,
)
from rtworkbench.report_llm import (
    SEARCH_FILTER_KEYS,
    ReportWriter,
    build_history_payload,
    llm_policy_status,
    parse_search_query,
    summarize_history,
)

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

TODAY = date(2026, 9, 3)  # 목요일 (weekday()==3) — 상대 날짜 테스트 기준일
LOCAL_URL = "http://localhost:11434/v1"
INSPECTOR_NAME = "홍길동"
IMAGE_NAME = "필름_스캔_777.png"
PAYLOAD = "[검사 개요]\n필름 ID: F-1\n블록: 3번 블록\n\n[결함별 판정]\n유의미한 결함 지시 없음\n\n[종합 판정]\n합격"


# ---------------------------------------------------------------- 픽스처


class _FakeResp:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.ok = status < 400

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._payload


def _openai_reply(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def net(monkeypatch):
    """모든 HTTP 를 가로챈다. 기본: 연결 거부(로컬 LLM 다운). calls 에 요청을 기록한다.

    state['post_reply'] — None: ConnectionError / _FakeResp: 그대로 반환.
    정책 기본값도 고정한다: 클라우드 차단, 키 없음, 로컬 URL/모델 고정.
    """
    calls: dict[str, list] = {"post": [], "get": []}
    state: dict = {"post_reply": None, "get_reply": None}

    def fake_post(url, *args, **kwargs):
        calls["post"].append(
            {"url": url, "json": kwargs.get("json"), "timeout": kwargs.get("timeout")}
        )
        reply = state["post_reply"]
        if reply is None:
            raise requests.ConnectionError("blocked in test")
        return reply

    def fake_get(url, *args, **kwargs):
        calls["get"].append({"url": url, "timeout": kwargs.get("timeout")})
        reply = state["get_reply"]
        if reply is None:
            raise requests.ConnectionError("blocked in test")
        return reply

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(config, "ALLOW_CLOUD_LLM", False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(config, "LOCAL_LLM_URL", LOCAL_URL)
    monkeypatch.setattr(config, "LOCAL_LLM_MODEL", "test-model")
    return types.SimpleNamespace(calls=calls, state=state)


@pytest.fixture
def fake_anthropic(monkeypatch):
    """anthropic SDK 가짜 모듈 — 생성자 호출 횟수를 기록하고 즉시 실패시킨다(폴백 유도)."""
    flag = {"constructed": 0}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            flag["constructed"] += 1
            raise RuntimeError("cloud call attempted (fake)")

    mod = types.ModuleType("anthropic")
    mod.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", mod)
    return flag


def _cloud_keys(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "g-test")


def _make_record(
    rid: str, *, block: str, weld: str, created_at: str,
    defects: list[tuple[str, float, float | None, bool, str]],
) -> InspectionRecord:
    """defects: (defect_type, size_mm, limit_mm, passed, clause)."""
    ctx = InspectionContext(
        film_id=f"F-{rid}", block=block, weld_id=weld, inspector=INSPECTOR_NAME,
    )
    candidates, verdicts = [], []
    for i, (dtype, size, limit, passed, clause) in enumerate(defects):
        did = f"{rid}-d{i}"
        candidates.append(
            DefectCandidate(id=did, defect_type=dtype, bbox=(0, 0, 10, 10),
                            confidence=0.9, status="accepted")
        )
        verdicts.append(
            RuleVerdict(defect_id=did, defect_type=dtype, size_mm=size, quality_level="B",
                        thickness_mm=12.0, limit_mm=limit, passed=passed, clause=clause,
                        detail="")
        )
    overall = all(v.passed for v in verdicts)
    return InspectionRecord(
        record_id=rid, context=ctx, candidates=candidates, measurements=[],
        verdicts=verdicts, overall_passed=overall,
        report_text=f"소견서 본문 (판독원 {INSPECTOR_NAME})", report_source="template",
        image_name=IMAGE_NAME, image_size=(1000, 800), created_at=f"{created_at}T10:00:00",
    )


@pytest.fixture
def records() -> list[InspectionRecord]:
    return [
        _make_record("r1", block="3번 블록", weld="BL3-V-012", created_at="2026-08-05",
                     defects=[("porosity", 3.2, 2.4, False, "DEMO-2011")]),
        _make_record("r2", block="3번 블록", weld="BL3-V-013", created_at="2026-08-12",
                     defects=[("porosity", 1.0, 2.4, True, "DEMO-2011"),
                              ("crack", 4.0, None, False, "DEMO-100")]),
        _make_record("r3", block="5번 블록", weld="BL5-H-001", created_at="2026-08-20",
                     defects=[("slag_inclusion", 1.1, 2.0, True, "DEMO-301")]),
    ]


# ---------------------------------------------------------------- 정책: 클라우드 차단


def test_allow_cloud_flag_parsing(monkeypatch):
    """RTWB_ALLOW_CLOUD_LLM 은 정확히 '1' 일 때만 허용, 미설정/기타 값은 차단."""
    monkeypatch.setenv("RTWB_ALLOW_CLOUD_LLM", "1")
    importlib.reload(config)
    assert config.ALLOW_CLOUD_LLM is True
    monkeypatch.setenv("RTWB_ALLOW_CLOUD_LLM", "true")
    importlib.reload(config)
    assert config.ALLOW_CLOUD_LLM is False
    monkeypatch.delenv("RTWB_ALLOW_CLOUD_LLM", raising=False)
    importlib.reload(config)
    assert config.ALLOW_CLOUD_LLM is False
    assert config.LLM_TIMEOUT_S == 20
    assert config.LOCAL_LLM_URL.endswith("/v1")
    monkeypatch.undo()
    importlib.reload(config)  # 원래 환경 기준으로 복원


def test_cloud_blocked_by_default_even_with_keys(net, fake_anthropic, monkeypatch, tmp_path):
    """키가 있어도 ALLOW_CLOUD_LLM=False(기본)면 Anthropic/Gemini 를 절대 호출하지 않는다."""
    _cloud_keys(monkeypatch)
    writer = ReportWriter(cache_path=tmp_path / "cache.json")
    text, source = writer.generate(PAYLOAD)

    assert source == "template"
    assert "RT 판독 소견서" in text
    assert fake_anthropic["constructed"] == 0  # Anthropic 클라이언트 생성조차 안 됨
    urls = [c["url"] for c in net.calls["post"]]
    assert urls == [f"{LOCAL_URL}/chat/completions"]  # 로컬만 시도
    assert not any("googleapis" in u for u in urls)


def test_cloud_used_only_when_explicitly_allowed(net, fake_anthropic, monkeypatch, tmp_path):
    """ALLOW_CLOUD_LLM=True + 키 → 로컬 실패 후 Anthropic → Gemini 순으로 시도(실패 시 템플릿)."""
    _cloud_keys(monkeypatch)
    monkeypatch.setattr(config, "ALLOW_CLOUD_LLM", True)
    writer = ReportWriter(cache_path=tmp_path / "cache.json")
    text, source = writer.generate(PAYLOAD)

    assert source == "template"  # 가짜 백엔드는 모두 실패 → 템플릿
    assert fake_anthropic["constructed"] == 1
    urls = [c["url"] for c in net.calls["post"]]
    assert urls[0] == f"{LOCAL_URL}/chat/completions"  # 로컬이 항상 1순위
    assert any("googleapis" in u for u in urls[1:])


# ---------------------------------------------------------------- 정책: 로컬 백엔드


def test_local_backend_used_when_reachable(net, tmp_path):
    net.state["post_reply"] = _FakeResp(_openai_reply("  로컬 LLM 소견서 본문  "))
    writer = ReportWriter(cache_path=tmp_path / "cache.json")

    text, source = writer.generate(PAYLOAD)
    assert (text, source) == ("로컬 LLM 소견서 본문", "local")

    call = net.calls["post"][0]
    assert call["url"] == f"{LOCAL_URL}/chat/completions"
    assert call["timeout"] == config.LLM_TIMEOUT_S
    body = call["json"]
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.2
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][1]["content"] == PAYLOAD
    system = body["messages"][0]["content"]
    assert "판정" in system and "수치" in system  # 판정 변경·수치 창작 금지

    # 로컬 성공 결과는 캐시 → 이후 로컬이 죽어도 캐시 히트
    net.state["post_reply"] = None
    assert writer.generate(PAYLOAD) == ("로컬 LLM 소견서 본문", "cache")


@pytest.mark.parametrize(
    "reply",
    [
        None,  # 연결 거부
        _FakeResp({"error": "boom"}, status=500),  # HTTP 오류
        _FakeResp({"unexpected": []}),  # 스키마 불일치
        _FakeResp(_openai_reply("   ")),  # 빈 응답
    ],
)
def test_local_failure_falls_back_to_template(net, tmp_path, reply):
    net.state["post_reply"] = reply
    writer = ReportWriter(cache_path=tmp_path / "cache.json")
    text, source = writer.generate(PAYLOAD)
    assert source == "template"
    assert "RT 판독 소견서" in text
    assert not (tmp_path / "cache.json").exists()  # 템플릿 결과는 캐시하지 않음


def test_local_url_unset_skips_local(net, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_LLM_URL", "")
    text, source = ReportWriter(cache_path=tmp_path / "c.json").generate(PAYLOAD)
    assert source == "template"
    assert net.calls["post"] == []
    status = llm_policy_status()
    assert status["local_reachable"] is None
    assert status["effective_order"] == ["cache", "template"]


# ---------------------------------------------------------------- llm_policy_status


def test_llm_policy_status_default(net):
    s = llm_policy_status()
    assert s["cloud_allowed"] is False
    assert s["local_url"] == LOCAL_URL
    assert s["local_model"] == "test-model"
    assert s["local_reachable"] is False
    assert s["anthropic_key"] is False and s["gemini_key"] is False
    assert s["effective_order"] == ["cache", "local", "template"]
    probe = net.calls["get"][0]
    assert probe["url"] == f"{LOCAL_URL}/models"
    assert probe["timeout"] <= 1.5


def test_llm_policy_status_keys_without_permission(net, monkeypatch):
    _cloud_keys(monkeypatch)
    s = llm_policy_status()
    assert s["anthropic_key"] is True and s["gemini_key"] is True
    assert s["effective_order"] == ["cache", "local", "template"]  # 키가 있어도 차단

    monkeypatch.setattr(config, "ALLOW_CLOUD_LLM", True)
    s = llm_policy_status()
    assert s["cloud_allowed"] is True
    assert s["effective_order"] == ["cache", "local", "claude", "gemini", "template"]


def test_llm_policy_status_local_reachable(net):
    net.state["get_reply"] = _FakeResp({"data": []})
    assert llm_policy_status()["local_reachable"] is True


# ---------------------------------------------------------------- parse_search_query (규칙 폴백)

_MONDAY = TODAY - timedelta(days=TODAY.weekday())


@pytest.mark.parametrize(
    "query, expected",
    [
        (
            "3번 블록에서 지난달 기공으로 불합격한 건",
            {"block": "3번 블록", "defect_type": "porosity", "passed": False,
             "date_from": "2026-08-01", "date_to": "2026-08-31"},
        ),
        ("블록 12 균열 합격", {"block": "12번 블록", "defect_type": "crack", "passed": True}),
        ("BL3-V-012 용접부 슬래그 이력", {"weld_id": "BL3-V-012", "defect_type": "slag_inclusion"}),
        (
            "군집 기공 이번 주",
            {"defect_type": "cluster_porosity", "date_from": _MONDAY.isoformat(),
             "date_to": (_MONDAY + timedelta(days=6)).isoformat()},
        ),
        (
            "지난주 W-104",
            {"weld_id": "W-104", "date_from": (_MONDAY - timedelta(days=7)).isoformat(),
             "date_to": (_MONDAY - timedelta(days=1)).isoformat()},
        ),
        (
            "최근 7일 융합불량",
            {"defect_type": "lack_of_fusion",
             "date_from": (TODAY - timedelta(days=7)).isoformat(), "date_to": TODAY.isoformat()},
        ),
        ("2026년 7월 용입부족", {"defect_type": "incomplete_penetration",
                              "date_from": "2026-07-01", "date_to": "2026-07-31"}),
        ("2026-05 언더컷", {"defect_type": "undercut", "date_from": "2026-05-01",
                          "date_to": "2026-05-31"}),
        ("어제 불합격", {"passed": False, "date_from": (TODAY - timedelta(days=1)).isoformat(),
                     "date_to": (TODAY - timedelta(days=1)).isoformat()}),
        ("오늘 부적합", {"passed": False, "date_from": TODAY.isoformat(),
                     "date_to": TODAY.isoformat()}),
        (
            "이번 달 F-2026-0830-01",  # 필름 ID 의 숫자를 날짜로 오인하면 안 된다
            {"text": "F-2026-0830-01", "date_from": "2026-09-01", "date_to": "2026-09-30"},
        ),
        ("8월 크랙", {"defect_type": "crack", "date_from": "2026-08-01", "date_to": "2026-08-31"}),
        ("올해 '수직 이음'", {"text": "수직 이음", "date_from": "2026-01-01",
                          "date_to": TODAY.isoformat()}),
        ("", {}),
    ],
)
def test_parse_search_query_rule_fallback(net, query, expected):
    filters, source = parse_search_query(query, today=TODAY)
    assert source == "rule"  # 로컬 LLM 다운 + 클라우드 차단 → 규칙 파서
    assert set(filters) == set(SEARCH_FILTER_KEYS)
    for k, v in expected.items():
        assert filters[k] == v, (k, filters)
    for k in SEARCH_FILTER_KEYS:
        if k not in expected:
            assert filters[k] in ("", None), (k, filters)


def test_parse_search_query_rule_result_is_valid_search_kwargs(net, tmp_path):
    """규칙 파서 출력은 그대로 Archive.search(**filters) 에 넣을 수 있어야 한다."""
    from rtworkbench.db import Archive

    filters, _ = parse_search_query("3번 블록 지난달 기공 불합격", today=TODAY)
    df = Archive(tmp_path / "t.db").search(**filters)
    assert list(df.columns)[:2] == ["record_id", "created_at"]


# ---------------------------------------------------------------- parse_search_query (LLM 경로)


def test_parse_search_query_uses_local_llm_json(net):
    net.state["post_reply"] = _FakeResp(_openai_reply(
        '```json\n{"block": "3", "weld_id": "", "defect_type": "기공", "passed": false, '
        '"text": "", "date_from": "2026-08-01", "date_to": "2026-08-31", "extra": "ignored"}\n```'
    ))
    q = "3번 블록 지난달 기공 불합격"
    filters, source = parse_search_query(q, today=TODAY)
    assert source == "local"
    assert filters == {
        "block": "3번 블록", "weld_id": "", "defect_type": "porosity", "passed": False,
        "text": "", "date_from": "2026-08-01", "date_to": "2026-08-31",
    }
    body = net.calls["post"][0]["json"]
    system = body["messages"][0]["content"]
    assert "JSON" in system and "porosity" in system and TODAY.isoformat() in system
    assert "판정" in system  # 판정 변경 금지 명시
    assert body["messages"][1]["content"] == q


def test_parse_search_query_llm_garbage_falls_back_to_rule(net):
    net.state["post_reply"] = _FakeResp(_openai_reply("죄송합니다, 질문을 이해하지 못했습니다."))
    filters, source = parse_search_query("3번 블록 균열", today=TODAY)
    assert source == "rule"
    assert filters["block"] == "3번 블록" and filters["defect_type"] == "crack"


def test_parse_search_query_llm_invalid_values_sanitized(net):
    net.state["post_reply"] = _FakeResp(_openai_reply(
        '{"block": "3번 블록", "defect_type": "외계결함", "passed": "maybe", '
        '"date_from": "어제", "date_to": "2026-13-45", "danger": "DROP TABLE"}'
    ))
    filters, source = parse_search_query("3번 블록", today=TODAY)
    assert source == "local"
    assert filters["block"] == "3번 블록"
    assert filters["defect_type"] == "" and filters["passed"] is None
    assert filters["date_from"] == "" and filters["date_to"] == ""
    assert "danger" not in filters


def test_parse_search_query_cloud_not_used_when_blocked(net, fake_anthropic, monkeypatch):
    _cloud_keys(monkeypatch)
    filters, source = parse_search_query("3번 블록 기공", today=TODAY)
    assert source == "rule"
    assert fake_anthropic["constructed"] == 0
    assert not any("googleapis" in c["url"] for c in net.calls["post"])


# ---------------------------------------------------------------- summarize_history


def test_summarize_history_template_fallback(net, records):
    text, source = summarize_history(records, question="3번 블록 경향은?")
    assert source == "template"
    assert "총 3건" in text
    assert "합격 1건" in text and "불합격 2건" in text
    assert "기공 2건" in text and "균열 1건" in text
    assert "3번 블록 2회" in text  # 반복 블록
    assert "DEMO-2011" in text and "DEMO-100" in text  # 불합격 항목 근거 조항
    # 비식별: 판독원 실명·이미지 파일명 제외
    assert INSPECTOR_NAME not in text and IMAGE_NAME not in text


def test_history_payload_is_deidentified(records):
    payload = build_history_payload(records)
    assert INSPECTOR_NAME not in payload
    assert IMAGE_NAME not in payload and ".png" not in payload
    assert "소견서 본문" not in payload  # 판독원 자유 편집 본문은 전송하지 않음
    assert "총 3건" in payload and "BL3-V-012" in payload and "DEMO-2011" in payload


def test_summarize_history_local_llm(net, records):
    net.state["post_reply"] = _FakeResp(_openai_reply("3번 블록 기공 불합격이 반복됨."))
    text, source = summarize_history(records, question="경향?")
    assert (text, source) == ("3번 블록 기공 불합격이 반복됨.", "local")
    body = net.calls["post"][0]["json"]
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]
    assert "판정" in system and "수치" in system
    assert INSPECTOR_NAME not in user and IMAGE_NAME not in user
    assert "총 3건" in user and "[질문]" in user and "경향?" in user


def test_summarize_history_caps_records(net):
    many = [
        _make_record(f"r{i}", block="1번 블록", weld=f"W-{i:03d}", created_at="2026-08-01",
                     defects=[("porosity", 1.0, 2.4, True, "DEMO-2011")])
        for i in range(80)
    ]
    payload = build_history_payload(many)
    listed = [ln for ln in payload.splitlines() if ln.startswith("2026-08-01 |")]
    assert len(listed) == report_llm.HISTORY_MAX_RECORDS == 60
    text, source = summarize_history(many)
    assert source == "template" and "총 60건" in text


def test_summarize_history_empty(net):
    assert summarize_history([]) == ("요약할 기록이 없습니다.", "template")
    assert net.calls["post"] == []  # 빈 입력은 LLM 호출 없음


# ---------------------------------------------------------------- 앱 통합 (AppTest, 오프라인)


def test_app_nl_search_offline_uses_rule_parser(net):
    """아카이브 탭 '자연어로 찾기' — LLM 불가 환경에서 규칙 파서로 해석·검색된다."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)
    assert not at.exception
    at.text_input(key="wb_nl_query").set_value("3번 블록 지난달 기공 불합격")
    at.run(timeout=60)
    next(b for b in at.button if b.key == "wb_btn_nl_parse").click()
    at.run(timeout=60)
    assert not at.exception, at.exception
    f = at.session_state["wb_nl_filters"]
    assert f["block"] == "3번 블록" and f["defect_type"] == "porosity" and f["passed"] is False
    assert at.session_state["wb_nl_source"] == "rule"
    # 클라우드/외부 호출 없음 — 로컬 프로브/시도만
    assert all("localhost" in c["url"] for c in net.calls["post"] + net.calls["get"])
