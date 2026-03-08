from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import ResearchState, CompanyInfo, CompanyAnalysis
from .firecrawl import FirecrawlService
from .prompts import DeveloperToolsPrompts


class Workflow:
    def __init__(self):
        self.firecrawl = FirecrawlService()
        # gpt-4o-mini의 추론 능력을 극대화하기 위해 온도를 낮게 유지합니다.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.prompts = DeveloperToolsPrompts()
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ResearchState)
        graph.add_node("extract_tools", self._extract_tools_step)
        graph.add_node("research", self._research_step)
        graph.add_node("analyze", self._analyze_step)
        graph.set_entry_point("extract_tools")
        graph.add_edge("extract_tools", "research")
        graph.add_edge("research", "analyze")
        graph.add_edge("analyze", END)
        return graph.compile()

    def _extract_tools_step(self, state: ResearchState) -> Dict[str, Any]:
        print(f"🔍 [1단계] 고밀도 데이터 스캐닝 시작: {state.query}")
        # 검색 소스를 5개로 확장하여 정보의 다양성을 확보합니다.
        article_query = f"{state.query} technical architecture review 2026"
        search_results = self.firecrawl.search_companies(article_query, num_results=5)

        all_content = ""
        # SearchData 객체 대응 로직
        data = getattr(search_results, 'data', []) if search_results else []

        for result in data:
            url = result.get("url")
            if url:
                scraped = self.firecrawl.scrape_company_pages(url)
                if scraped and hasattr(scraped, 'markdown'):
                    # 컨텍스트 수집량을 8,000자로 대폭 늘려 디테일을 확보합니다.
                    all_content += (scraped.markdown[:8000] + "\n\n")

        messages = [
            SystemMessage(content="당신은 기술 분석 전문가입니다. 제공된 자료에서 가장 혁신적이고 실무적인 도구 6개 이상을 추출하세요."),
            HumanMessage(content=self.prompts.tool_extraction_user(state.query, all_content))
        ]

        try:
            response = self.llm.invoke(messages)
            tool_names = [n.strip() for n in response.content.split("\n") if n.strip()]
            print(f"✅ 추출된 핵심 기술군: {', '.join(tool_names[:5])}")
            return {"extracted_tools": tool_names}
        except Exception as e:
            print(f"Extraction Error: {e}")
            return {"extracted_tools": []}

    def _research_step(self, state: ResearchState) -> Dict[str, Any]:
        # 비교 분석 대상을 6개로 늘려 리포트의 풍성함을 더합니다.
        tool_names = state.extracted_tools[:6] if state.extracted_tools else [state.query]
        print(f"🔬 [2단계] {len(tool_names)}개 기술 스택 심층 조사 중...")

        companies = []
        for tool_name in tool_names:
            # 단순 사이트가 아닌 '기술 사양'과 '아키텍처'를 위주로 검색합니다.
            search_results = self.firecrawl.search_companies(f"{tool_name} documentation architecture specs",
                                                             num_results=2)
            data = getattr(search_results, 'data', []) if search_results else []

            if data:
                res = data[0]
                company = CompanyInfo(
                    name=tool_name,
                    description=res.get("markdown", "")[:2000],  # 설명 길이를 대폭 확장
                    website=res.get("url", "")
                )

                scraped = self.firecrawl.scrape_company_pages(company.website)
                if scraped and hasattr(scraped, 'markdown'):
                    analysis = self._analyze_company_content(tool_name, scraped.markdown)
                    # models.py 필드에 맞게 자동 매핑
                    for field in ["pricing_model", "is_open_source", "tech_stack", "description",
                                  "api_available", "language_support", "integration_capabilities"]:
                        setattr(company, field, getattr(analysis, field, None))
                companies.append(company)
        return {"companies": companies}

    def _analyze_company_content(self, company_name: str, content: str) -> CompanyAnalysis:
        structured_llm = self.llm.with_structured_output(CompanyAnalysis)
        messages = [
            SystemMessage(content="전문 연구원으로서 기술적 아키텍처와 한계점을 정밀하게 분석하세요."),
            HumanMessage(content=self.prompts.tool_analysis_user(company_name, content))
        ]
        try:
            return structured_llm.invoke(messages)
        except Exception:
            return CompanyAnalysis(pricing_model="Unknown")

    def _analyze_step(self, state: ResearchState) -> Dict[str, Any]:
        print("📊 [3단계] 석박사급 한국어 기술 분석 보고서 생성 중...")
        company_data = ", ".join([c.model_dump_json() for c in state.companies])

        # 한국어 출력과 학술적 형식을 강제하는 프롬프트 지침
        detailed_prompt = f"""
        당신은 인하대학교 iDEAL 랩실의 수석 연구원입니다.
        주제: '{state.query}'에 대한 심층 기술 분석 및 전략 제언

        작성 지침:
        1. **모든 내용은 반드시 한국어로 작성하세요.**
        2. 서론, 본론(도구별 심층 분석), 결론(연구 및 도입 제언) 형식을 갖추세요.
        3. 각 기술의 아키텍처, 확장성, 보안성을 비판적으로 분석하세요.
        4. 도구 간의 주요 스펙 차이를 보여주는 비교 표(Markdown Table)를 포함하세요.
        5. 분량은 최소 2,000자 이상의 풍부한 내용을 담으세요.
        6. 기술 용어는 한국어와 영어를 병기하세요. (예: 분산 시스템(Distributed System))

        수집 데이터: {company_data}
        """

        messages = [
            SystemMessage(content="당신은 기술 전문 컨설턴트이자 연구 보고서 작성 전문가입니다. 모든 응답은 한국어로 생성합니다."),
            HumanMessage(content=detailed_prompt)
        ]

        try:
            response = self.llm.invoke(messages)
            return {"analysis": response.content}
        except Exception as e:
            return {"analysis": f"리포트 생성 실패: {e}"}

    def run(self, query: str) -> ResearchState:
        initial_state = ResearchState(query=query)
        final_state = self.workflow.invoke(initial_state)
        return ResearchState(**final_state)