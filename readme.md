[![Build status](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml/badge.svg?branch=homework1)](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml)

## Quick start
All actions should be done from the `./ml_project/` directory.

### Requirements
```commandline
pip install -r ./requirements.txt
```

### Train
Config example:
```yaml
# config with random forest and label encoding
dataset:
  path: ./data/train.csv
  target_col: raintomorrow
  id_col: row_id
  is_fake: False
  fake_size: 15000
model:
  mode: forest
transformer:
  mode: forest
split:
  test_size: 0.25
artifacts:
  path: ./data/pipeline.pickle
logging:
  path: ./data/cache.log
```
Execution example:
```commandline
python ./source/train.py --config-name=config_1.yaml split.test_size=0.33
```

### Predict
```commandline
python ./source/predict.py ./data/pipeline.pickle ./data/holdout.csv ./data/prediction.csv
```

### Test
```commandline
python -m pytest ./source/test.py 
```

## Project tree
```
.
|-- __init__.py            
|-- config                 <- configuration files for training;
|   |-- config_1.yaml      
|   |-- config_2.yaml      
|   `-- config_3.yaml
|-- data                   <- input/output directory;
|   |-- holdout.csv        <- data for predicton;
|   |-- pipeline.pickle    <- serialized model;
|   |-- prediction.csv     <- model predictions;
|   `-- train.csv          <- train data;
|-- requirements.txt       <- dependencies list;
|-- research               <- EDA directory;
|   |-- notebook.ipynb     <- jupyter notebook;
|   |-- report.py          <- script that generates EDA report;
|   `-- result             <- report results (with plots);
|       |-- correlation_matrix.pdf
|       |-- feature_importances.pdf
|       |-- possible_outliers.pdf
|       |-- result.txt
|       `-- target_distribution.pdf
`-- source                 <- source directory;
    |-- dataclass_.py      <- train dataclass;
    |-- logger.py          <- setup logger;
    |-- model.py           <- model module;
    |-- predict.py         <- prediction function;
    |-- test.py            <- test function;
    |-- train.py           <- train function;
    `-- transformer.py     <- custom transformer module.
```