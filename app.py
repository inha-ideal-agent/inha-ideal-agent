import streamlit as st
from src.workflow import Workflow
import time

# 웹 페이지 설정 (탭 제목, 아이콘)
st.set_page_config(page_title="iDEAL AI Research Agent", page_icon="🚢", layout="wide")

# 사이드바 설정
with st.sidebar:
    st.image("https://www.inha.ac.kr/sites/inha/images/common/logo.png", width=200)  # 인하대 로고
    st.title("설정")
    st.info("이 에이전트는 실시간 웹 스크래핑을 통해 석박사급 기술 보고서를 생성합니다.")

# 메인 화면
st.title("🚢 인하대학교 iDEAL 랩실 전용 AI 연구 에이전트")
st.subheader("주제를 입력하면 에이전트가 전 세계 웹을 뒤져 리포트를 작성합니다.")

# 검색창 (인터넷 검색 기능의 시작점)
query = st.text_input("연구 주제 또는 기술명을 입력하세요:", placeholder="예: YOLOv11의 조선소 안전 관리 적용 방안")

if st.button("🔍 연구 시작"):
    if not query:
        st.warning("질문을 입력해 주세요!")
    else:
        # 에이전트 가동
        with st.status("에이전트가 인터넷에서 자료를 수집하고 분석 중입니다...", expanded=True) as status:
            st.write("🌐 실시간 웹 스캐닝 중...")
            start_time = time.time()

            try:
                # 우리가 만든 Workflow 클래스 호출
                agent = Workflow()
                result = agent.run(query)

                end_time = time.time()
                status.update(label=f"✅ 리포트 생성 완료! (소요 시간: {int(end_time - start_time)}초)", state="complete",
                              expanded=False)

                # 결과 출력
                st.divider()
                st.markdown(f"### 📊 '{query}' 에 대한 기술 분석 리포트")
                st.markdown(result.analysis)  # 한국어 최종 리포트 출력

                # 다운로드 버튼 추가
                st.download_button(
                    label="📄 리포트 다운로드 (Markdown)",
                    data=result.analysis,
                    file_name=f"{query}_analysis.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")