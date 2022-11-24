FROM python:3.9
RUN python -m pip install --upgrade pip

COPY main.py schema.py online_inference/
COPY requirements.txt online_inference/
COPY model.pkl online_inference/
WORKDIR online_inference/

# ENV TOKEN=<YOUR TOKEN>
# ENV MODEL_PATH=<PATH TO MODEL>

RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]