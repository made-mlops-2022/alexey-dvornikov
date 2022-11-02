"""
Transformer module.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    PolynomialFeatures,
)

transformer_logger = logging.getLogger("transformer")


class WeatherTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for weather dataset.
    """

    def __init__(self, mode="linear"):
        """
        Class constructor;
        :param mode: "linear" or "forest".
        Uses different preprocessing technics:
            1) Adding numerical polynomial features;
            2) Label-encoding categorical features.
        """
        self.mode = mode
        self.numeric_cols = None
        self.categorical_cols = None
        self.scaler = StandardScaler()
        self.transformer = PolynomialFeatures()

        transformer_logger.debug(
            "Created an instance of WeatherTransformer; mode=%s.", self.mode
        )

    def fit(self, data: pd.DataFrame, target=None):
        """
        Fit transformer with data;
        :param target: target array;
        :param data: dataframe;
        :return: None.
        """
        target *= 1
        data = self._replace(data, {"Yes": 1, "No": 0})

        self.numeric_cols = list(data.select_dtypes(include=np.number).columns)
        self.categorical_cols = list(
            data.select_dtypes(include="object").columns
        )

        data = self._fill_nans(data, self.numeric_cols, self.categorical_cols)

        self.scaler.fit(data[self.numeric_cols])
        self.transformer.fit(data[self.numeric_cols])

        transformer_logger.info("WeatherTransformer fitted.")
        return self

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Transform data;
        :param data: dataframe;
        :return: numpy array.
        """
        data = self._replace(data, {"Yes": 1, "No": 0})

        data = self._fill_nans(data, self.numeric_cols, self.categorical_cols)

        data[self.numeric_cols] = self.scaler.transform(
            data[self.numeric_cols]
        )

        data = self._encode(data)

        if self.mode == "linear":
            transformer_logger.debug("WeatherTransformer preprocessed data.")
            return self.transformer.transform(data[self.numeric_cols])
        if self.mode == "forest":
            data = np.array(data)
            transformer_logger.debug("WeatherTransformer preprocessed data.")
            return data
        transformer_logger.warning(
            "mode=%s is not supported, using default=linear.", self.mode
        )
        return self.transformer.transform(data[self.numeric_cols])

    @staticmethod
    def _drop(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        data = data.drop(columns, axis=1)
        return data

    @staticmethod
    def _replace(data: pd.DataFrame, replace_dict: dict) -> pd.DataFrame:
        data = data.replace(replace_dict)
        return data

    @staticmethod
    def _fill_nans(
        data: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        for column in data.columns:
            if column in numeric_cols:
                data[column] = data[column].fillna(np.mean(data[column]))
            elif column in categorical_cols:
                data[column] = data[column].fillna("unknown")
        return data

    @staticmethod
    def _encode(data: pd.DataFrame) -> pd.DataFrame:
        encoder = LabelEncoder()
        for column in data.columns:
            data[column] = encoder.fit_transform(data[column])
        return data
