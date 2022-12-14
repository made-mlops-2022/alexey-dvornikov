"""
Dataclasses module.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sdv.tabular import GaussianCopula
from omegaconf import DictConfig
from model import WeatherModel
from transformer import WeatherTransformer

dataclass_logger = logging.getLogger("dataclass")


def generate_fake_data(
    real_data: pd.DataFrame, n_samples: int
) -> pd.DataFrame:
    """
    Generate fake data, based on the given data;
    :param real_data: real dataframe;
    :param n_samples: number or samples;
    :return: fake dataframe;
    """
    model = GaussianCopula()
    model.fit(real_data)
    return model.sample(n_samples)


class TrainData:
    """
    Train dataclass.
    """

    def __init__(self, cfg: DictConfig):
        """
        Class constructor;
        :param cfg: path to config.
        """
        self.config = cfg

        data = pd.read_csv(self.config["dataset"]["path"])
        target_col = self.config["dataset"]["target_col"]
        id_col = self.config["dataset"]["id_col"]

        is_fake = self.config["dataset"]["is_fake"]
        fake_size = self.config["dataset"]["fake_size"]
        if is_fake:
            dataclass_logger.info(
                "Generating fake train data; size=%d.", fake_size
            )
            data = generate_fake_data(data, fake_size)

        self.x_train, self.y_train = (
            data.drop([target_col, id_col], axis=1),
            data[target_col],
        )

        self.x_val = None
        self.y_val = None

        dataclass_logger.debug("Created an instance of TrainData.")

    def get_data(
        self,
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Return data;
        :return: train&validation dataframes.
        """
        return self.x_train, self.y_train, self.x_val, self.y_val

    def split_data(self) -> None:
        """
        Split data into train&validation with defined validation size;
        :return: None.
        """
        dataclass_logger.info("Splitting train data...")
        test_size = self.config["split"]["test_size"]
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train,
            self.y_train,
            test_size=test_size,
            shuffle=True,
            random_state=42,
        )

    def get_artifacts_path(self) -> str:
        """
        Get path to serialized model;
        :return: path.
        """
        return self.config["artifacts"]["path"]

    def get_logging_path(self) -> str:
        """
        Get path to logging file;
        :return: path.
        """
        return self.config["logging"]["path"]

    def get_model(self) -> Pipeline:
        """
        Get model pipeline;
        :return: sklearn pipeline.
        """
        model = Pipeline(
            steps=[
                (
                    "transformer",
                    WeatherTransformer(
                        mode=self.config["transformer"]["mode"]
                    ),
                ),
                (
                    "model",
                    WeatherModel(mode=self.config["model"]["mode"]),
                ),
            ]
        )
        dataclass_logger.info("Pipeline created.")
        return model
