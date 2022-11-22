# MADE-2022 | MLOps | HW2

## Quick start
Build docker image from inside the `./online_inference/` directory:
```commandline
docker build -t woodkeeper/weather-model:v2 .
```

Or pool a docker image from docker-hub:
```commandline
docker pull woodkeeper/weather-model:v2
```

## Quick run
- Run docker container :
   ```commandline
   docker run -p 8000:8000 --name WeatherModel woodkeeper/weather-model:v1
   ```
 - Go to http://127.0.0.1:8000/docs.

## Quick test
```commandline
python -m pytest test.py
```

## Project tree
```
.
|-- Dockerfile        <- Image configuration;
|-- main.py           <- FastAPI app;
|-- model.pkl         <- serialized model;
|-- readme.md         <- documentation;
|-- requests          
|   |-- holdout.csv   <- test data;
|   `-- requests_.py  <- requests.script;
|-- requirements.txt  <- requirements;
|-- schema.py         <- data scheme;
`-- test.py           <- test scripts.
```

## Docker Optimization
1. [(v1)](https://hub.docker.com/layers/woodkeeper/weather-model/v1/images/sha256-a09f36fbcd2639812d53f04bb0e20970aa0940f52e1347a73b080b879cf445ad?context=repo) Using `python:3.9`:
   - size ~ 1.46GB
   - build time ~ 41s
2. [(v2)](https://hub.docker.com/layers/woodkeeper/weather-model/v2/images/sha256-f362e072638c184cb7bddc00a39578013066dbc870367d1ea48ce8576e5ee911?context=repo) Using `python:3.9-slim-buster`:
   - size ~ 661MB (-63% from v1)
   - build time ~ 32s (-3% from v1)