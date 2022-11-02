"""
Test module.
"""

from train import train
from predict import predict
from dataclass_ import TrainData

CONFIG_PATH_1 = "01/config/1_config.yaml"
CONFIG_PATH_2 = "01/config/2_config.yaml"
CONFIG_PATH_3 = "01/config/3_config.yaml"


def test_1():
    """
    Test train&predict with 1st config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_1))
    predict(
        "01/data/pipeline.pickle",
        "01/data/holdout.csv",
        "01/data/prediction.csv",
    )


def test_2():
    """
    Test train&predict with 2nd config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_2))
    predict(
        "01/data/pipeline.pickle",
        "01/data/holdout.csv",
        "01/data/prediction.csv",
    )


def test_3():
    """
    Test train&predict with 3rd config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_3))
    predict(
        "01/data/pipeline.pickle",
        "01/data/holdout.csv",
        "01/data/prediction.csv",
    )


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
