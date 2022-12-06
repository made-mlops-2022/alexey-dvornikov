FROM python:3.8.15-slim-buster

RUN pip install --upgrade pip

WORKDIR mlflow/
COPY requirements.txt .
RUN pip install -r requirements.txt

EXPOSE 5000

ENV ARTIFACT_ROOT ./mlruns

CMD mlflow server \
    --backend-store-uri sqlite:///:memory \
    --default-artifact-root $ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port 5000