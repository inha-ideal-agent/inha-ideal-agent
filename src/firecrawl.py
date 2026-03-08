import os
from firecrawl import FirecrawlApp  # ScrapeOptions 임포트를 제거하여 ImportError를 방지합니다.
from dotenv import load_dotenv

load_dotenv()


class FirecrawlService:
    def __init__(self):
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("Missing FIRECRAWL_API_KEY environment variable")
        self.app = FirecrawlApp(api_key=api_key)

    def search_companies(self, query: str, num_results: int = 5):
        """
        웹 검색을 수행하고 결과를 반환합니다.
        최신 SDK 버전과의 호환성을 위해 딕셔너리 형태의 옵션을 사용합니다.
        """
        try:
            # ScrapeOptions 클래스 대신 딕셔너리를 사용하여 Pydantic 검증 에러를 해결했습니다.
            result = self.app.search(
                query=f"{query} company pricing",
                limit=num_results,
                scrape_options={"formats": ["markdown"]}
            )
            return result
        except Exception as e:
            print(f"Firecrawl Search Error: {e}")
            return []

    def scrape_company_pages(self, url: str):
        """
        특정 URL의 내용을 스크래핑합니다.
        """
        try:
            result = self.app.scrape_url(
                url,
                params={"formats": ["markdown"]} # 최신 버전 규격에 맞게 인자명을 조정했습니다.
            )
            return result
        except Exception as e:
            print(f"Firecrawl Scrape Error: {e}")
            return None