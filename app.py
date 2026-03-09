import streamlit as st
from src.workflow import Workflow
import time
import sqlite3
import pandas as pd
from datetime import datetime


# 1. 데이터베이스 설정 (연구 기록 저장용)
def init_db():
    conn = sqlite3.connect('ideal_research.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (
                     id
                     INTEGER
                     PRIMARY
                     KEY
                     AUTOINCREMENT,
                     query
                     TEXT,
                     content
                     TEXT,
                     date
                     TEXT
                 )''')
    conn.commit()
    conn.close()


def save_report(query, content):
    conn = sqlite3.connect('ideal_research.db')
    c = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO reports (query, content, date) VALUES (?, ?, ?)", (query, content, date))
    conn.commit()
    conn.close()


def delete_report(report_id):
    conn = sqlite3.connect('ideal_research.db')
    c = conn.cursor()
    c.execute("DELETE FROM reports WHERE id=?", (report_id,))
    conn.commit()
    conn.close()


def update_report(report_id, new_content):
    conn = sqlite3.connect('ideal_research.db')
    c = conn.cursor()
    c.execute("UPDATE reports SET content=? WHERE id=?", (new_content, report_id))
    conn.commit()
    conn.close()


# 초기화
init_db()
st.set_page_config(page_title="iDEAL Research Platform", page_icon="🚢", layout="wide")

# 사이드바: 랩실 로고 및 히스토리 관리
with st.sidebar:
    st.image("https://www.inha.ac.kr/sites/inha/images/common/logo.png", width=200)
    st.title("📂 연구 히스토리")

    conn = sqlite3.connect('ideal_research.db')
    history_df = pd.read_sql_query("SELECT id, query, date FROM reports ORDER BY date DESC", conn)
    conn.close()

    if not history_df.empty:
        for index, row in history_df.iterrows():
            if st.button(f"📄 {row['query'][:15]}... ({row['date'][5:10]})", key=f"hist_{row['id']}"):
                st.session_state['view_id'] = row['id']
    else:
        st.write("저장된 리포트가 없습니다.")

    st.divider()
    st.info("💡 **Deep Dive Mode**: Firecrawl 검색 강도를 높여 석박사급 리포트를 생성합니다.")

# 메인 화면 UI
st.title("🚢 iDEAL Lab Intelligence Platform")
st.caption("Inha University - Intelligent Digital Engineering & Architecture Lab")

# 탭 구성: 새로운 연구 vs 저장된 연구
tab1, tab2 = st.tabs(["🔍 신규 기술 분석", "📝 리포트 관리 및 수정"])

with tab1:
    query = st.text_input("분석하고자 하는 연구 주제를 입력하세요:", placeholder="예: 탄소중립 선박을 위한 수소 연료전지 기술 동향")

    col1, col2 = st.columns([1, 4])
    with col1:
        deep_dive = st.toggle("Deep Dive Mode (심층 분석)", value=True)

    if st.button("🚀 분석 엔진 가동"):
        if not query:
            st.warning("주제를 입력해 주세요.")
        else:
            with st.status("iDEAL AI가 전 세계 웹 데이터를 스캐닝 중입니다...", expanded=True) as status:
                start_time = time.time()
                try:
                    agent = Workflow()
                    # Deep Dive 모드일 때 프롬프트를 더 강력하게 수정하여 전달
                    enhanced_query = f"{query}에 대해 석박사 수준의 깊이 있는 기술 아키텍처와 한계점, 향후 연구 방향을 포함해서 작성해줘." if deep_dive else query

                    result = agent.run(enhanced_query)
                    end_time = time.time()

                    # DB 저장
                    save_report(query, result.analysis)

                    status.update(label=f"✅ 리포트 생성 및 DB 저장 완료! ({int(end_time - start_time)}초)", state="complete")
                    st.markdown(result.analysis)
                    st.rerun()  # 히스토리 업데이트를 위해 리프레시
                except Exception as e:
                    st.error(f"엔진 오류: {str(e)}")

with tab2:
    if 'view_id' in st.session_state:
        conn = sqlite3.connect('ideal_research.db')
        curr_report = pd.read_sql_query(f"SELECT * FROM reports WHERE id={st.session_state['view_id']}", conn)
        conn.close()

        if not curr_report.empty:
            st.subheader(f"📌 리포트 제목: {curr_report['query'].values[0]}")
            st.caption(f"생성 일시: {curr_report['date'].values[0]}")

            # 수정 기능: text_area에 내용 담기
            edited_content = st.text_area("리포트 내용 수정", value=curr_report['content'].values[0], height=500)

            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("💾 수정사항 저장", use_container_width=True):
                    update_report(st.session_state['view_id'], edited_content)
                    st.success("리포트가 업데이트되었습니다!")
                    st.rerun()
            with c2:
                st.download_button("📥 Markdown 다운로드", data=edited_content, file_name="research_report.md",
                                   use_container_width=True)
            with c3:
                if st.button("🗑️ 리포트 삭제", type="primary", use_container_width=True):
                    delete_report(st.session_state['view_id'])
                    del st.session_state['view_id']
                    st.warning("리포트가 삭제되었습니다.")
                    st.rerun()
        else:
            st.info("왼쪽 사이드바에서 리포트를 선택해 주세요.")
    else:
        st.info("관리할 리포트를 사이드바 히스토리에서 선택하거나 새로 생성하세요.")