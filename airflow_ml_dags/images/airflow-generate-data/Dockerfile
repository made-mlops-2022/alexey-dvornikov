FROM python:3.8.15-slim-buster
RUN python3 -m pip install --upgrade pip

COPY generate_data.py requirements.txt /
COPY holdout.csv /

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3", "generate_data.py"]
