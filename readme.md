[![Build status](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml/badge.svg?branch=01)](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml)

## Quick start

### Requirements
```
$ pip install -r ./ml_project/requirements.txt
```

### Train
```
$ python ./ml_project/source/train.py ./ml_project/config/1_config.yaml
```

### Predict
```
$ python ./ml_project/source/predict.py ./ml_project/data/pipeline.pickle 
./ml_project/data/holdout.csv ./ml_project/data/prediction.csv
```

### Test
```
$ python -m pytest ./ml_project/source/test.py 
```