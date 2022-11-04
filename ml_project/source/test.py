"""
Test module.
"""

from train import train
from predict import predict
from dataclass_ import TrainData

CONFIG_PATH_1 = "./config/1_config.yaml"
CONFIG_PATH_2 = "./config/2_config.yaml"
CONFIG_PATH_3 = "./config/3_config.yaml"


def test_1():
    """
    Test train&predict with 1st config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_1))
    predict(
        "./data/pipeline.pickle",
        "./data/holdout.csv",
        "./data/prediction.csv",
    )


def test_2():
    """
    Test train&predict with 2nd config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_2))
    predict(
        "./data/pipeline.pickle",
        "./data/holdout.csv",
        "./data/prediction.csv",
    )


def test_3():
    """
    Test train&predict with 3rd config;
    :return: None.
    """
    train(TrainData(CONFIG_PATH_3))
    predict(
        "./data/pipeline.pickle",
        "./data/holdout.csv",
        "./data/prediction.csv",
    )


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
