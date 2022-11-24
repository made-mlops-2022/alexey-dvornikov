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
   docker run -p 8000:8000 --name WeatherModel woodkeeper/weather-model:v2
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
1. [(v1)](https://hub.docker.com/layers/woodkeeper/weather-model/v1/images/sha256-32453170fefb0876c78e7b03339d0cff1f1b51aadbf6ad2637d14bf5aa17c96a?context=repo) Using `python:3.9`:
   - size ~ 1.46GB
   - build time ~ 51s
2. [(v2)](https://hub.docker.com/layers/woodkeeper/weather-model/v2/images/sha256-a0a1bc7369279f2dbdf9b4e1b4cfb09d24d3b71cde62e1e00b2fc2c8c415e843?context=repo) Using `python:3.9-slim-buster`:
   - size ~ 660MB (-65% from v1)
   - build time ~ 47s (-8% from v1)