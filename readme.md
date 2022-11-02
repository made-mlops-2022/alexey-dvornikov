## Quick start

### Requirements
```
$ pip install -r ./requirements.txt
```

### Train
```
$ python ./source/train.py ./config/1_config.yaml
```

### Predict
```
$ python ./source/predict.py ./data/pipeline.pickle ./data/holdout.csv ./data/prediction.csv
```

### Test
```
$ python -m pytest source
```