[![Build status](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml/badge.svg?branch=homework1)](https://github.com/made-mlops-2022/alexey-dvornikov/actions/workflows/checks.yaml)

## Quick start
All actions should be done from the `ml_project` directory.

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
$ python -m pytest ./source/test.py 
```

## Project tree
```
.
|-- __init__.py            
|-- config                 <- configuration files for training;
|   |-- 1_config.yaml      
|   |-- 2_config.yaml      
|   `-- 3_config.yaml
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