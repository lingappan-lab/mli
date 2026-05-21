FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

COPY . ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .[app]

EXPOSE 7860
CMD ["python", "app.py"]
