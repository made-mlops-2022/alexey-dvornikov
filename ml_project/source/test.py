"""
Test module.
"""

import yaml
from train import train
from predict import predict

CONFIG_PATH_1 = "./config/config_1.yaml"
CONFIG_PATH_2 = "./config/config_2.yaml"
CONFIG_PATH_3 = "./config/config_3.yaml"


def test_1():
    """
    Test train&predict with 1st config;
    :return: None.
    """
    with open(CONFIG_PATH_1, "r", encoding="UTF-8") as file:
        cfg_1 = yaml.safe_load(file)
        train(cfg_1)
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
    with open(CONFIG_PATH_1, "r", encoding="UTF-8") as file:
        cfg_2 = yaml.safe_load(file)
        train(cfg_2)
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
    with open(CONFIG_PATH_1, "r", encoding="UTF-8") as file:
        cfg_3 = yaml.safe_load(file)
        train(cfg_3)
        predict(
            "./data/pipeline.pickle",
            "./data/holdout.csv",
            "./data/prediction.csv",
        )


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
