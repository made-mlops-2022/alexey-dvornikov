"""
Test module.
"""

from train import train
from predict import predict
from dataclass_ import TrainData

CONFIG_PATH_1 = "ml_project/config/1_config.yaml"
CONFIG_PATH_2 = "ml_project/config/2_config.yaml"
CONFIG_PATH_3 = "ml_project/config/3_config.yaml"


def test_1():
    """
    Test train&predict with 1st config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_1))
    predict(
        "ml_project/data/pipeline.pickle",
        "ml_project/data/holdout.csv",
        "ml_project/data/prediction.csv",
    )


def test_2():
    """
    Test train&predict with 2nd config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_2))
    predict(
        "ml_project/data/pipeline.pickle",
        "ml_project/data/holdout.csv",
        "ml_project/data/prediction.csv",
    )


def test_3():
    """
    Test train&predict with 3rd config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_3))
    predict(
        "ml_project/data/pipeline.pickle",
        "ml_project/data/holdout.csv",
        "ml_project/data/prediction.csv",
    )


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
