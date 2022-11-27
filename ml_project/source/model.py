"""
Model module.
"""

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

model_logger = logging.getLogger("model")


class WeatherModel:
    """
    Prediction model for weather dataset.
    """

    def __init__(self, mode="logreg"):
        """
        Class constructor;
        :param mode: str, "logreg" or "forest" --
        Logistic Regression or Random Forest.
        """
        self._lr = LogisticRegression()
        self._rf = RandomForestClassifier()
        self.mode = mode

        if self.mode == "logreg":
            self._estimator = self._lr
        elif self.mode == "forest":
            self._estimator = self._rf
        else:
            model_logger.warning(
                "mode=%s is not supported, using default=logreg.", self.mode
            )
            self._estimator = self._lr

        model_logger.debug(
            "Created an instance of WeatherModel; mode=%s.", self.mode
        )

    def fit(self, x_train: np.array, y_train: np.array) -> None:
        """
        Fit the model;
        :param x_train: 2D array of train data;
        :param y_train: array of train labels;
        :return: None.
        """
        model_logger.debug("WeatherModel started fitting...")
        self._estimator.fit(x_train, y_train)
        model_logger.debug("WeatherModel fitted.")

    def predict(self, x_test: np.array) -> np.array:
        """
        Predict labels for the test data;
        :param x_test: 2D array of test data;
        :return: array of predicted labels.
        """
        model_logger.debug("WeatherModel predicted probabilities.")
        return self._estimator.predict(x_test)

    def predict_proba(self, x_test: np.array) -> np.array:
        """
        Predict probabilities for the test data;
        :param x_test: 2D array of test data;
        :return: array of predicted probabilities.
        """
        model_logger.debug("WeatherModel predicted labels.")
        return self._estimator.predict_proba(x_test)
