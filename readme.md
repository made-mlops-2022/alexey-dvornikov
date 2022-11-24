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
1. [(v1)](https://hub.docker.com/layers/woodkeeper/weather-model/v1/images/sha256-7026eac4cf99681d5f4541b4f1afd40d0261baaf462318b86d20a8027a667f5c?context=repo) Using `python:3.9`:
   - size ~ 1.46GB
   - build time ~ 51s
2. [(v2)](https://hub.docker.com/layers/woodkeeper/weather-model/v2/images/sha256-24199e90ee206c5459c90912b75e472e90e47a0915c2b3288f99886b7304e7fd?context=repo) Using `python:3.9-slim-buster`:
   - size ~ 660MB (-65% from v1)
   - build time ~ 47s (-8% from v1)