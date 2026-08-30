# 🩻 RT 판독 워크벤치 (RT Reading Workbench)

> **"판독은 자격자가, 서류는 AI가."**
> RT(방사선투과검사) 판독원 1인의 하루에서 **문서 노동**과 **이력 검색 노동**을 지우는 책상 위 소프트웨어.
> 2026 K-조선 해커톤(생산 분야) 프로토타입 — v3 컨셉 구현체 (연습용).

---

## 3대 설계 원칙

1. **AI는 판정하지 않는다. 판정을 보조하고 문서화한다.**
   합부는 결정론적 룰 엔진 + 판독원. LLM은 설명·작문 전담 (환각 차단 + 자격자 권한 존중).
2. **장비 교체 없이, 새 행동 요구 없이.**
   입력은 지금 있는 필름 스캔 이미지. 판독원의 기존 업무 사이에 꽂힌다.
3. **탐지는 부품이다.**
   신규성은 판독 워크플로우 전체(2차 눈 + 판정 보조 + 소견서 + 아카이브 + 자기개선)의 통합.

## 4대 기능

| # | 기능 | 구현 |
|---|---|---|
| 1 | **2차 눈 (Second Reader)** — 판독원이 먼저 보고, AI가 놓침 방지 후보 표시. recall 우선(오탐은 클릭 한 번, 미탐은 배에 남는다) | `rtworkbench/detection.py` — YOLO 백엔드(가중치 있을 때) + OpenCV 폴백 탐지기(항상 동작) |
| 2 | **2클릭 자 + 룰 판정** — 판독원이 IQI/납마커 기준 2클릭으로 스케일 확정, 결함 크기도 2클릭 확정 → 룰 엔진이 기준표 대조 합부+조항 제시 | `rtworkbench/measure.py` + `rtworkbench/rules.py` + `criteria/demo_iso5817_like.json` |
| 3 | **소견서 초안 자동 생성** — 확정된 판정 결과(비식별 텍스트만)를 LLM이 리포트 문체로 작성 → 판독원 수정·승인 → PDF | `rtworkbench/report_llm.py` (Claude→Gemini→오프라인 템플릿 폴백 + 응답 캐시) + `rtworkbench/report_pdf.py` |
| 4 | **검색 아카이브 + 자기개선 루프** — 승인 기록 DB 축적, "3번 블록 기공 이력" 3초 검색, 승인 기록 = YOLO 라벨 데이터 export | `rtworkbench/db.py` (SQLite) |

## 실행

### 가장 쉬운 방법 — 파일 하나로 실행 (터미널 명령 불필요)

| 환경 | 방법 |
|---|---|
| Windows | **`실행.bat` 더블클릭** (Python만 설치되어 있으면 됨) |
| macOS/Linux | `./run_workbench.sh` 또는 `python3 run_workbench.py` |

런처가 알아서 처리합니다: 필수 패키지 확인(없으면 설치 제안) → 샘플 필름 자동 생성(최초 1회)
→ 빈 포트에 서버 기동 → **기본 브라우저 자동 오픈**. 종료는 창에서 Ctrl+C.
(화면은 브라우저에 뜹니다 — Streamlit 기반 데스크톱 웹앱의 동작 방식입니다.)

### 수동 실행

```bash
cd rt-workbench
pip install -r requirements.txt

# 합성 샘플 필름 생성 (실데이터 반출 불가 환경용 데모 이미지)
python3 scripts/generate_samples.py

# 앱 실행
streamlit run app.py
```

LLM API 키는 **없어도 전부 동작**합니다(오프라인 템플릿 백엔드).
있으면 자동 사용: `ANTHROPIC_API_KEY`(1순위), `GEMINI_API_KEY`(2순위 폴백).

```bash
# 헤드리스 E2E 데모 (탐지→캘리브레이션→측정→판정→소견서→PDF→DB→검색→라벨 export)
python3 scripts/demo_e2e.py

# 테스트
python3 -m pytest tests/ -q
```

## 데모 시나리오 (판독원의 5분)

1. **필름 로드**: 샘플 선택 또는 스캔 이미지 업로드 → CLAHE 대비 향상 토글
2. **2차 눈**: "AI 후보 탐지" → 주황 박스 후보 → 오탐은 [기각] 클릭 한 번, 실결함은 [채택]
3. **캘리브레이션**: 클릭 모드 '캘리브레이션' → 납마커 양 끝 2클릭 + 실길이 10mm 입력 → mm/px 확정
4. **측정**: 클릭 모드 '결함 측정' → 후보 선택 → 결함 양 끝 2클릭 → mm 확정
5. **판정**: "판정 실행" → 결함별 허용 한계·조항·합부 + 종합 판정 배너 (룰 엔진, AI 아님)
6. **소견서**: "초안 생성" → 편집 → **승인** → PDF 다운로드 + 아카이브 저장
7. **아카이브 탭**: "3번 블록" 검색 → 과거 이력 즉시 조회
8. **자기개선 탭**: AI 채택률·미탐(사람 추가) 지표 확인 → YOLO 라벨 export

## 아키텍처

```
app.py (Streamlit UI — 탭: 판독 워크벤치 / 아카이브 검색 / 자기개선 루프 / 기준표·정보)
 ├─ preprocess.py   CLAHE 전처리
 ├─ detection.py    2차 눈: YoloDetector(옵션) / CVFallbackDetector(기본)
 ├─ measure.py      2클릭 자: 캘리브레이션(mm/px)·거리 측정   ← 사람이 확정
 ├─ rules.py        결정론적 룰 엔진 ← criteria/*.json (데이터 주도, 교체 가능)
 ├─ report_llm.py   소견서 초안: 비식별 payload → Claude→Gemini→템플릿 폴백 (+캐시)
 ├─ report_pdf.py   승인 소견서 PDF (reportlab, 한글 CID 폰트)
 ├─ db.py           SQLite 아카이브: 저장/검색/통계/YOLO 라벨 export
 └─ models.py       공용 데이터 모델 (전 모듈의 계약)
```

## 보안 설계

- LLM에는 **이미지를 보내지 않는다**. `build_payload()`가 만든 비식별 판정 결과 텍스트만 전송
  (검사원 실명·파일 경로 제외). 전송 내용은 UI에서 미리보기 가능.
- 오프라인 모드: API 응답 사전 캐시 + 결정론적 템플릿 백엔드로 네트워크 없이 시연 가능.
- 로드맵: 온프레미스/로컬 모델 전환.

## 주의 (데모 한계)

- `criteria/demo_iso5817_like.json`의 수치는 **ISO 5817/10675 계열 구조를 모사한 데모 값**이며
  규격 원문이 아닙니다. 실제 생산 검사 합부 기준은 선급 NDT 지침 확인·도메인 멘토링으로 확정 후
  JSON만 교체하면 됩니다 (룰 엔진 코드는 무변경 — 데이터 주도 설계).
- CV 폴백 탐지기는 모델 가중치 없이 데모가 항상 동작하게 하는 안전망입니다.
  실 성능은 AI Hub 용접 AI(71761)·RIAWELC 학습 YOLOv8s 가중치를 `weights/best.pt`에 두면
  자동으로 YOLO 백엔드로 전환됩니다 (`RTWB_YOLO_WEIGHTS` 환경변수로 경로 변경 가능).
- 합성 샘플 필름은 개발·시연용이며 실제 방사선 사진이 아닙니다.
