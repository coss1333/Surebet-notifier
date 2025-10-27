# Optional Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD [ "bash", "-lc", "export $(grep -v '^#' .env | xargs) && python main.py" ]
