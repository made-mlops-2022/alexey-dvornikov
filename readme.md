![Build status](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml/badge.svg?branch=01)

## Quick start

### Requirements
```
$ pip install -r ./01/requirements.txt
```

### Train
```
$ python ./01/source/train.py ./01/config/1_config.yaml
```

### Predict
```
$ python ./01/source/predict.py ./01/data/pipeline.pickle ./01/data/holdout.csv
./01/data/prediction.csv
```

### Test
```
$ python -m pytest ./01/source/test.py 
```