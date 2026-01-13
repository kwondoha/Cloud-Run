FROM python:3.13-slim

# Copy uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN uv sync

# Expose port
EXPOSE 8080

# Run the application
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]

#uv: pip보다 10~100배 빠른 최신 Python 패키지 관리자입니다. 전역 캐시와 병렬 다운로드를 사용하여 컨테이너 빌드 시간을 크게 줄임
# Python 3.13-slim: 최소한의 시스템 종속 항목이 있는 최신 Python 버전으로, 컨테이너 크기와 공격 표면을 줄임
# 다단계 빌드: 공식 이미지에서 uv를 복사하면 최신 최적화 바이너리를 가져올 수 있음