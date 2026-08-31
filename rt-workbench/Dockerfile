# RT 판독 워크벤치 — 서버/클라우드 배포용 (Hugging Face Spaces, VPS, 사내 서버 등)
# 빌드:  docker build -t rt-workbench .
# 실행:  docker run -p 8501:8501 rt-workbench
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 샘플 필름을 이미지에 포함 (없으면 생성)
RUN python scripts/generate_samples.py

EXPOSE 8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
