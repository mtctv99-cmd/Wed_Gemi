FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY gemnix_api/ ./gemnix_api/
COPY config.example.json ./config.json
EXPOSE 8081

CMD ["python", "-m", "gemnix_api", "--config", "/app/config.json"]
