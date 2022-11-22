# MADE-2022 | MLOps | HW2

## Quick start
Build docker image from inside the `./online_inference/` directory:
```commandline
docker build -t woodkeeper/weather-model:v1 .
```

Or pool a docker image from docker-hub:
```commandline
docker pull woodkeeper/weather-model:v1
```

## Quick run
- Run docker container :
   ```commandline
   docker run -p 8000:8000 --name WeatherModel woodkeeper/weather-model:v1
   ```
 - Go to http://127.0.0.1:8000/docs.

## Project tree
```
.
|-- Dockerfile
|-- main.py
|-- model.pkl
|-- readme.md
|-- requests
|   |-- holdout.csv
|   `-- requests_.py
|-- requirements.txt
|-- schema.py
`-- test.py
```