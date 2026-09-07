# 🩻 RT 판독 워크벤치 (RT Reading Workbench)

> **"판독은 자격자가, 서류는 AI가."**
> RT(방사선투과검사) 판독원 1인의 하루에서 **문서 노동**과 **이력 검색 노동**을 지우는 책상 위 소프트웨어.
> 2026 K-조선 해커톤(생산 분야) 프로토타입 — v3 컨셉 구현체 (연습용).

---

## 3대 설계 원칙

1. **AI는 판정하지 않는다. 판정을 보조하고 문서화한다.**
   합부는 결정론적 룰 엔진 + 판독원. LLM은 설명·작문 전담 (환각 차단 + 자격자 권한 존중).
2. **기존 사진·기존 절차 위에.**
   입력은 스캔한 필름 사진 또는 디지털 RT(CR/DR) 사진. 판독원 개인에게 새 장비·새 촬영 절차를
   요구하지 않는다(필름 디지털화는 ISO 14096-2 등급 디지타이저·스캔 서비스 등 조직 단위 전제조건).
3. **탐지는 부품이다.**
   신규성은 판독 워크플로우 전체(2차 눈 + 판정 보조 + 소견서 + 아카이브 + 자기개선)의 통합.

## 4대 기능

| # | 기능 | 구현 |
|---|---|---|
| 1 | **2차 눈 (Second Reader)** — 판독원이 먼저 보고, AI가 놓침 방지 후보 표시. recall 우선(오탐은 클릭 한 번, 미탐은 배에 남는다) | `rtworkbench/detection.py` — YOLO 백엔드(가중치 있을 때) + OpenCV 폴백 탐지기(항상 동작) |
| 2 | **2클릭 자 + 룰 판정** — 판독원이 IQI/납마커 기준 2클릭으로 스케일 확정, 결함 크기도 2클릭 확정 → 룰 엔진이 기준표 대조 합부+조항 제시. 판정식 3종: **단일 치수** `min(계수×t, cap)` / **누적 길이** (평가 길이 내 같은 유형의 길이 합) / **투영 면적률** (Σ원 근사 면적 ÷ (평가 길이 × 용접부 폭), %) — 뒤의 둘은 유형별 `합계(...)` 행으로 추가되며, 사이드바의 **평가 길이·용접부 폭** 입력이 기준 구간. 스케일이 확정되면 평가 길이(기본 **100 mm**) 창을 용접선을 따라 이동시켜 **가장 불리한 구간**으로 판정(IACS UR W33 Rev.2 100 mm 판독 구간 지침 — 아래 절 참조) | `rtworkbench/measure.py` + `rtworkbench/rules.py` + `criteria/*.json` (규격 판본별 데모 기준표 2종 — 사이드바 **기준표(규격 판본)**에서 선택) |
| 3 | **소견서 초안 자동 생성** — 확정된 판정 결과(비식별 텍스트만)를 LLM이 리포트 문체로 작성 → 판독원 수정·승인 → PDF | `rtworkbench/report_llm.py` (로컬 LLM→[명시 허용 시 Claude→Gemini]→오프라인 템플릿 폴백 + 응답 캐시) + `rtworkbench/report_pdf.py` |
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

### LLM 정책 — 로컬 우선 (local-first)

조선소/NDT 현장은 검사 텍스트의 외부 반출이 기본 금지입니다. 따라서 **기본 정책은
로컬 LLM 또는 오프라인 템플릿**이며, 외부 클라우드 API는 명시 허용 시에만 사용합니다.
LLM이 하나도 없어도 **전부 동작**합니다(오프라인 템플릿 백엔드).

폴백 순서: 캐시 → 로컬 LLM → *(RTWB_ALLOW_CLOUD_LLM=1 일 때만)* Claude → Gemini → 오프라인 템플릿

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `RTWB_LOCAL_LLM_URL` | `http://localhost:11434/v1` | OpenAI 호환 base URL (Ollama/vLLM 등) |
| `RTWB_LOCAL_LLM_MODEL` | `exaone3.5:7.8b` | 로컬 모델명 — 기본값은 **EXAONE 3.5**(LG AI연구원 한국어 모델). EXAONE 4.x·HyperCLOVA X SEED 등 Ollama/vLLM이 서빙하는 모델명으로 교체 가능 |
| `RTWB_ALLOW_CLOUD_LLM` | `0` (차단) | `1` 로 설정할 때만 외부 클라우드 API 사용 — 키가 있어도 이 값이 없으면 호출하지 않음 |
| `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | (없음) | 클라우드 허용 시 사용 (Claude 1순위, Gemini 2순위) |

```bash
# 로컬 LLM 예시 (Ollama)
ollama pull exaone3.5:7.8b && ollama serve   # 기본 URL http://localhost:11434/v1
```

현재 정책·연결 상태는 앱의 **📖 기준표·정보 → LLM 정책** 표에서 확인할 수 있습니다.

**LLM 고유 기능** (템플릿이 대체할 수 없는 것 — 단, LLM은 어느 경우에도 판정하지 않음):

- **🗣️ 자연어로 찾기** (아카이브 탭): "3번 블록에서 지난달 기공으로 불합격한 건" → LLM이
  검색 **필터만** JSON으로 추출(블록/용접부/결함 유형/합부/기간) → 검색은 결정론적 DB 조회.
  LLM 불가 시 규칙 파서(`rule`)로 폴백 — 상대 날짜(어제/이번 주/지난달/최근 N일/YYYY년 M월) 인식.
- **AI 이력 요약** (아카이브 탭): 검색 결과(최대 60건)의 비식별 집계를 판독원용 한국어 요약으로
  (유형별 건수, 합부 비율, 반복 블록·용접부, 주요 불합격 항목+조항). LLM 불가 시 통계 템플릿.

```bash
# 헤드리스 E2E 데모 (탐지→캘리브레이션→측정→판정→소견서→PDF→DB→검색→라벨 export)
python3 scripts/demo_e2e.py

# 테스트
python3 -m pytest tests/ -q
```

## 웹 배포 (아이패드·외부 접속용) — 데모 목적은 전부 무료

| 방법 | 비용 | 특징 |
|---|---|---|
| 같은 와이파이 접속 | 무료 | PC에서 실행 → 태블릿/폰이 `http://<PC IP>:8501` 접속 (런처가 주소 표시) |
| [Streamlit Community Cloud](https://share.streamlit.io) | 무료 | GitHub 연결 → 브랜치·`rt-workbench/app.py` 지정하면 공개 URL 발급. 일정 시간 미사용 시 슬립(첫 접속 시 깨어남), 재시작 시 저장소 초기화 |
| Hugging Face Spaces | 무료 티어 | 동봉된 `Dockerfile`로 배포 가능 |
| 소형 VPS (Lightsail 등) | 월 ~5천 원 | 항상 켜짐 + 데이터 영속 필요할 때만 |

무료 클라우드는 재시작 시 SQLite DB가 초기화되므로, **자기개선 루프 탭의
"JSON 백업 다운로드 / 복원"**으로 아카이브를 보존·복원하세요.

## 데모 시나리오 (판독원의 5분)

1. **필름 로드**: 샘플 선택 또는 스캔 이미지 업로드 → CLAHE 대비 향상 토글
2. **2차 눈**: "AI 후보 탐지" → 주황 박스 후보 → 오탐은 [기각] 클릭 한 번, 실결함은 [채택]
3. **캘리브레이션**: 클릭 모드 '캘리브레이션' → 납마커 양 끝 2클릭 + 실길이 10mm 입력 → mm/px 확정
4. **측정**: 클릭 모드 '결함 측정' → 후보 선택 → 결함 양 끝 2클릭 → mm 확정
5. **판정**: 사이드바에서 **기준표(규격 판본)** 확인(건조계약일 기준 2023/2021 또는 2014/2016 구조) → "판정 실행" → 결함별 허용 한계·조항·합부 + 유형별 합계 행(누적 길이·투영 면적률) + 종합 판정 배너 (룰 엔진, AI 아님)
6. **소견서**: "초안 생성" → 편집 → **승인** → PDF 다운로드 + 아카이브 저장
7. **아카이브 탭**: "3번 블록" 검색 → 과거 이력 즉시 조회
8. **자기개선 탭**: AI 채택률·미탐(사람 추가) 지표 확인 → YOLO 라벨 export

## 아키텍처

```
app.py (Streamlit UI — 탭: 판독 워크벤치 / 아카이브 검색 / 자기개선 루프 / 기준표·정보)
 ├─ preprocess.py   CLAHE 전처리
 ├─ detection.py    2차 눈: YoloDetector(옵션) / CVFallbackDetector(기본)
 ├─ measure.py      2클릭 자: 캘리브레이션(mm/px)·거리 측정   ← 사람이 확정
 ├─ rules.py        결정론적 룰 엔진 ← criteria/*.json (단일 치수·누적 길이·투영 면적률, 데이터 주도)
 │                    그룹 판정은 100 mm 창 이동(최악 구간) — worst_window(); 위치 없으면 전체 구간 폴백
 │                    규격 판본별 기준표 JSON 분리 — 파일 하나가 판본 하나, 엔진은 JSON 교체만으로 대응:
 │                    ├ demo_iso5817_like.json       ISO 5817:2023/ISO 10675-1:2021 구조 모사
 │                    │                                (IACS UR W33 Rev.2 — 2028.01.01 이후 건조계약분, 기본)
 │                    └ demo_iso5817_2014_like.json  ISO 5817:2014/ISO 10675-1:2016 구조 모사
 │                                                     (2028.01.01 이전 건조계약분)
 ├─ report_llm.py   소견서 초안·자연어 검색 필터·이력 요약: 비식별 payload → 로컬 LLM→[클라우드]→템플릿 (+캐시)
 ├─ report_pdf.py   승인 소견서 PDF (reportlab, 한글 CID 폰트)
 ├─ db.py           SQLite 아카이브: 저장/검색/통계/YOLO 라벨 export
 └─ models.py       공용 데이터 모델 (전 모듈의 계약)
```

## 보안 설계

- LLM에는 **이미지를 보내지 않는다**. `build_payload()`가 만든 비식별 판정 결과 텍스트만 전송
  (검사원 실명·파일 경로 제외). 전송 내용은 UI에서 미리보기 가능.
- 오프라인 모드: API 응답 사전 캐시 + 결정론적 템플릿 백엔드로 네트워크 없이 시연 가능.
- **로컬 우선 LLM 정책**: 기본은 온프레미스 로컬 LLM(OpenAI 호환) 또는 템플릿.
  외부 클라우드 API는 `RTWB_ALLOW_CLOUD_LLM=1` 로 명시 허용해야만 호출 — 키가 있어도 기본 차단.
- 자연어 검색에서 LLM은 **필터 추출만** 담당 — 검색·판정은 결정론적 코드가 수행.

## 기준표(규격 판본) 선택 — 판본별 JSON 분리

건조계약일에 따라 적용 규격 판본이 다릅니다(IACS UR W33 Rev.2: **2028.01.01 이후 건조계약분**은
ISO 5817:2023/ISO 10675-1:2021, 그 이전은 ISO 5817:2014/ISO 10675-1:2016). 워크벤치는 판본을
**기준표 JSON 파일 단위**로 분리하며, 룰 엔진 코드는 어느 판본이든 동일합니다.

| 파일 (`rtworkbench/criteria/`) | 구조 모사 대상 | 적용 대상 |
|---|---|---|
| `demo_iso5817_like.json` (기본) | ISO 5817:2023 / ISO 10675-1:2021 | 2028.01.01 이후 건조계약분 (IACS UR W33 Rev.2) |
| `demo_iso5817_2014_like.json` | ISO 5817:2014 / ISO 10675-1:2016 | 2028.01.01 이전 건조계약분 |

- 사이드바 **기준표(규격 판본)** selectbox에서 선택 → 판정 실행 시 그 JSON으로 `RuleEngine`이 구성됩니다.
  선택을 바꾸면 기존 판정은 무효화되어 재판정이 필요합니다(두께·품질등급 변경과 같은 취급).
- 적용된 기준표의 **이름·버전**은 판정 표(`기준표` 열), 소견서 PDF(검사 정보 표·푸터),
  아카이브 기록(`criteria_name`/`criteria_version`, 상세 보기)에 남습니다. 판본 분리 전 기록은 "미기록"으로 표시됩니다.
- 새 판본/실제 기준표 추가 = `criteria/` 에 같은 스키마의 JSON 파일을 하나 더 두는 것뿐입니다
  (`rules.list_criteria()`가 자동 인식, `RuleEngine("<파일명 stem>")`으로 로드). 두 파일의 `meta.edition_note`가
  판본을, `meta.disclaimer`가 데모 여부를 명시합니다.

## 100 mm 판독 구간 — 창 이동(최악 구간) 그룹 판정

IACS UR W33 Rev.2 지침: *"판독 구간 100 mm 를 필름 전체에 걸쳐 적용한다(연속된 여러 100 mm 구간) …
기공은 가장 심한 분포를 담은 100 mm 구간을 택한다."* 워크벤치의 그룹 판정(누적 길이·투영 면적률)은
이 지침을 다음과 같이 구현합니다(`rules.worst_window`, `RuleEngine.evaluate_group(..., positions=)`).

1. **위치**: 결함 중심의 용접선 축 위치(mm) = 2클릭 측정선(`Measurement.p1/p2`)의 중점 픽셀 좌표 × 스케일.
   용접선은 이미지 **x축**을 따라 놓였다고 가정합니다(`measure.weld_axis_position_mm(..., axis='x'|'y')`,
   앱은 `app.WELD_AXIS = 'x'`). 스케일이 확정된 뒤에만 위치가 생깁니다.
2. **창 이동**: 사이드바 평가 길이(기본 100 mm) 창을 용접선 축을 따라 이동시킵니다 — 각 결함 위치에서
   시작하는 창과 끝나는 창을 후보로 삼으면 최악 창을 반드시 찾습니다(어떤 창이든 왼쪽 끝을 첫 결함까지
   밀어도 구성원을 잃지 않으므로). 창 안 여부는 결함 **중심** 기준(양끝 포함)이며 창 안 결함은 크기 전체를 셉니다.
3. **최악 구간 선택**: 같은 유형의 창 안 결함으로 누적 길이 또는 투영 면적률을 계산해 **값이 가장 큰 창**
   하나로 판정합니다(동률이면 시작 좌표가 작은 창 — 결정론적). 판정 근거에
   `최악 100mm 구간 [x0~x1 mm] 내 기공 3건(필름 전체 5건 중) …` 로 구간을 명시하고
   `RuleVerdict.window_start_mm/window_end_mm` 에 기록합니다(판정 표 캡션·PDF 각주에도 표시).
4. **폴백**: 유형 내 결함 하나라도 위치가 없으면(스케일 미확정 등) 종전처럼 필름 전체를 한 구간으로
   합산하고 근거에 `위치 정보 없음 → 전체를 한 구간으로 보수적 합산` 을 덧붙입니다(과소 판정 방지).

## 선급 보고 항목 (IACS UR W33 §8.2·8.5) — 보고서 기재 사항 저장

IACS UR W33 §8은 선급에 제출하는 NDT 보고서에 **일반 항목(§8.2, 13개)** 과 **RT 전용 항목(§8.5, 12개)** 을
요구합니다. 워크벤치는 이 항목들을 검사 기록과 함께 저장해 소견서 PDF·아카이브에 남깁니다.
§8.2 항목 중 이미 검사 컨텍스트/기록에 있는 것(검사일 = 승인 일시, 용접부 위치 = 블록/용접부 ID,
이음 종류, 모재 두께, 합격 기준 = 품질등급 + 적용 기준표, 결과 = 판정 표, 합부 = 종합 판정,
평가자 = 판독원)은 중복 입력받지 않고, 나머지를 사이드바 **선급 보고 항목(IACS UR W33 §8.2·8.5)**
expander(기본 접힘)에서 입력합니다. 사이드바 캡션 **"📋 선급 보고 항목(IACS UR W33 §8) n/25 충족 — 미입력: …"**
이 **보고 항목 25개(일반 13·RT 12) 충족률**을 보여줍니다 — 분모는 입력 필드 수가 아니라 W33 **원문 불릿 수**이며,
검사 컨텍스트·판정·승인 기록으로 충족되는 불릿(검사일, 합격 기준, 검사 결과 …)도 셉니다(`models.w33_coverage`).

| 구분 | 항목 (`InspectionContext.report` = `ReportDetails` 필드) |
|---|---|
| 일반 (§8.2) | 선체 번호 `hull_number` · 검사 용접 길이 `weld_length_mm` · 검사자 자격 등급 `personnel_qualification` · 강재 등급 `steel_grade` · 용접 방법 `welding_process` · 검사 규격 `testing_standard` · 검사 장비·배치 `equipment` · 제한사항·관찰 조건·온도 `limitations_viewing` · 보수 횟수(2회 초과 시) `repairs_count` |
| RT 촬영 조건 (§8.5) | 선원 종류·크기 `source_type_size` · X선 관전압 `xray_kv` · 필름 종류·카세트당 매수 `film_type` · 촬영 매수 `exposures_count` · 증감지 `screens` · 촬영 기법 `exposure_technique` · 노출 시간 `exposure_time_s` · SFD `sfd_mm` · 선원-용접부 거리 `source_to_weld_mm` · 용접부-필름 거리 `weld_to_film_mm` · 빔 입사각 `beam_angle_deg` · IQI 감도 `iqi_sensitivity` · IQI 종류·위치 `iqi_type_position` · 농도 `density` · 기하학적 불선명도 `geometric_unsharpness` · RT 합격 등급 `rt_acceptance_class` |

- **판정에 관여하지 않습니다.** 룰 엔진 입력은 두께·품질등급·기준표·평가 구간뿐이며, 보고 항목을
  바꿔도 기존 판정은 무효화되지 않습니다. 대신 그 항목이 실리는 산출물은 낡은 것으로 봅니다 — 미리 생성한
  PDF와 초안 payload를 버리고, 이미 만든 초안 본문은 남긴 채 "초안을 다시 생성하세요" 경고를 띄웁니다
  (소견서 본문의 촬영 조건과 PDF 5절 표가 어긋나지 않도록).
- 어디에 실리는가: 소견서 PDF **"5. 선급 보고 항목(IACS UR W33 §8)"** 표(입력된 항목만, 한국어 라벨 +
  W33 원문 병기, 하나도 없으면 섹션 생략) · LLM 초안 payload의 **`[촬영 조건]`** 섹션(입력된 항목만 —
  검사원 실명은 여전히 제외, 자격 등급만 포함) · 아카이브 기록(`context.report`, 상세 보기 expander).
- 항목 목록은 `models.REPORT_ITEMS` 한 곳에서 관리합니다(한국어 라벨·W33 원문·단위). `ReportDetails.coverage()`
  는 입력 **필드** 수 `(입력 수, 25)` 를 돌려주고, UI·PDF·아카이브가 보여주는 충족률은
  `models.w33_coverage(context, record=None, judged=…) → (충족 수, 25, 미충족 라벨)` 입니다.
  보고 항목이 없는 구버전 기록/백업 JSON도 그대로 로드됩니다.
- **충족 규칙** (`models.W33_ITEMS[*].rule_ko` 와 동일) — 원칙: 불릿의 하위 정보 중 워크벤치가 담을 수 있는
  것이 **모두** 있을 때 충족, 컨텍스트/판정/승인 기록에 이미 있는 정보는 자동 충족, 조건부 항목은 보수적으로 처리.

  | § | W33 불릿 | 충족 조건 |
  |---|---|---|
  | 8.2 | Date of testing | 항상(승인 시각 자동 기록) |
  | 8.2 | Hull number, location and length of weld inspected | 선체 번호 + 위치(블록 또는 용접부 ID) + 검사 용접 길이 |
  | 8.2 | Names, qualification level and signature of personnel | 판독원(성명) + 검사자 자격 등급 (서명은 출력물에 수기) |
  | 8.2 | Identification of the component examined | 블록 |
  | 8.2 | Identification of the welds examined | 용접부 ID |
  | 8.2 | Steel grade, type of joint, thickness, welding process | 강재 등급 + 용접 방법 (이음 종류·두께는 컨텍스트 필수값) |
  | 8.2 | Acceptance criteria | 항상(품질등급 + 적용 기준표 이름·버전) |
  | 8.2 | Testing standards used | 검사 규격 (기법명 'RT (필름 스캔)'은 규격이 아니므로 제외) |
  | 8.2 | Testing equipment and arrangement used | 검사 장비·배치 |
  | 8.2 | Any test limitations, viewing conditions and temperature | 제한사항·관찰 조건·온도 |
  | 8.2 | Results of testing … location and size of reportable indications | 판정 실행(verdict 표; 결함 없음도 결과) |
  | 8.2 | Statement of acceptance / non-acceptance, evaluation date, evaluator | 판정 실행 + 판독원(성명) (평가일 자동) |
  | 8.2 | Number of repairs if specific area repaired more than twice | 보수 횟수 ≥ 1 (0 = 해당 없음은 기본값과 구분 불가 → 미충족) |
  | 8.5 | Type and size of radiation source, X-ray voltage | 선원 종류·크기 (관전압은 X선 장비에만 해당 → 필수 아님) |
  | 8.5 | Type of film/designation and number of film per cassette | 필름 종류·카세트당 매수 |
  | 8.5 | Number of radiographs (exposures) | 촬영 매수 |
  | 8.5 | Type of intensifying screens | 증감지 종류 |
  | 8.5 | Exposure technique, time of exposure and source-to-film distance | 촬영 기법 + 노출 시간 + SFD |
  | 8.5 | Distance from radiation source to weld | 선원-용접부 거리 |
  | 8.5 | Distance from source side of the weld to radiographic film | 용접부-필름 거리 |
  | 8.5 | Angle of radiation beam through the weld | 빔 입사각 |
  | 8.5 | Sensitivity, type and position of IQI | IQI 감도 + IQI 종류·위치 |
  | 8.5 | Density | 농도 |
  | 8.5 | Geometric un-sharpness | 기하학적 불선명도 |
  | 8.5 | Specific acceptance class criteria for RT | RT 합격 등급 |
- 수치 항목도 자유 서식 문자열로 저장합니다(보고서에 옮겨 적는 용도, 계산에 쓰지 않음).

## 주의 (데모 한계)

- `criteria/` 의 두 JSON(`demo_iso5817_like.json`, `demo_iso5817_2014_like.json`)의 수치는 각 판본의
  **구조를 모사한 데모 값**이며 규격 원문의 수치가 아닙니다 — 두 파일 사이의 한계값 차이도 판본 전환이
  판정에 반영됨을 보이기 위한 데모 차이입니다. 실제 생산 검사 합부 기준은 선급 NDT 지침 확인·도메인
  멘토링으로 확정 후 JSON만 교체하면 됩니다 (룰 엔진 코드는 무변경 — 데이터 주도 설계).
- CV 폴백 탐지기는 모델 가중치 없이 데모가 항상 동작하게 하는 안전망입니다.
  실 성능은 AI Hub 용접 AI(71761)·RIAWELC 학습 **YOLO26s**(Ultralytics YOLO, 2026.01 공개 — YOLOv8/11 가중치도
  같은 API) 가중치를 `weights/best.pt`에 두면
  자동으로 YOLO 백엔드로 전환됩니다 (`RTWB_YOLO_WEIGHTS` 환경변수로 경로 변경 가능).
  YOLO26 가중치 로드에는 **`ultralytics>=8.4`** 가 필요합니다(`requirements.txt`의 주석 해제 후 설치;
  구버전 ultralytics는 YOLO26 아키텍처를 인식하지 못합니다).
- 합성 샘플 필름은 개발·시연용이며 실제 방사선 사진이 아닙니다.

## 참고 자료

제안서 참고 문헌 중 규격·데이터셋·논문·도구 항목(뉴스·보도자료 제외)입니다(앱의 **📖 기준표·정보 → 참고 자료**와 동일하게 유지 — 테스트가 두 곳을 대조합니다).

**규칙·규격**
1. IACS UR W33 Rev.1/Corr.1 (2021) §3.3·§8·§9.2 — 선급 NDT 규칙 (§8: 보고서 필수 항목 25개)
2. IACS UR W33 Rev.2 · W34 Rev.1 — 2026.07.15 승인, 2028.01.01 이후 건조계약분 적용, 100 mm 평가 길이
3. ISO 10675-1:2021 — 용접부 RT 허용 레벨(1·2·3)
4. ISO 5817:2023 — 강 용접부 품질 등급(B·C·D)
5. ISO 17636-1/-2:2022 — RT 촬영 기법(필름 / 디지털 검출기)
6. ISO 14096-2 — 필름 디지타이저 등급(DS/DB/DA) — 필름 스캔의 조직 단위 전제조건
7. ISO 9712:2021 — NDT 인력 자격(판정은 Level 2 이상)

**데이터셋**
8. GDXray — Mery et al., J Nondestruct Eval 34:42, 2015 (Welds 시리즈, 연구·교육 목적만)
9. RIAWELC — Totino·Spagnolo·Perri 2022, GitHub (조각 사진 4종 분류, 위치 박스 없음)
10. AI Hub 71761 용접 AI 학습 데이터 — NIA 2023 (RT·VT 용접 사진, 폴리곤 라벨; 수치는 활용 신청 시 데이터셋 페이지에서 확인)

**논문**
11. WeldVGG — Sensors 25(19):6183, 2025 (데이터셋 간 제로샷 전이 한계)
12. Palma-Ramírez et al. — Heliyon 2024, e30590
13. Lu et al. — J Nondestruct Eval 44:91, 2025 (GDXray mAP@0.5 97.2%)

**도구**
14. Ultralytics YOLO26 — 2026.01 공개, AGPL-3.0 (사내 배포 시 Enterprise 라이선스; YOLO26s 권장)
15. Streamlit — 브라우저 UI(내 PC에서 실행, 같은 와이파이의 태블릿 접속)
16. Ollama / vLLM — OpenAI 호환 로컬 LLM API
17. EXAONE 3.5 / 4.x — LG AI연구원 한국어 모델 (로컬 기본값 exaone3.5:7.8b)
18. HyperCLOVA X SEED — NAVER 한국어 모델 후보 (검색 요약 기준)

"(검색 요약 기준)" 표시는 원문이 아닌 검색 스니펫으로만 확인한 항목 — 제출 전 원문 확인 필요.
