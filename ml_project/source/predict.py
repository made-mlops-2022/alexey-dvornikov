"""
Prediction script.
"""

import sys
import pickle
import warnings
import logging
import pandas as pd
from logger import set_logger

warnings.simplefilter("ignore")

predict_logger = logging.getLogger("predict")
ID_COL = "row_id"
TARGET_COL = "raintomorrow"


def predict(
    path_to_artifacts: str,
    path_to_test: str,
    path_to_output: str,
    path_to_cache="ml_project/data/cache.log",
    predict_proba=False,
) -> None:
    """
    Prediction function;
    :param path_to_artifacts: path to serialized module;
    :param path_to_test: path to test data;
    :param path_to_output: path to output;
    :param path_to_cache: path to logging file;
    :param predict_proba: if predict probabilities;
    :return: None.
    """
    set_logger(path_to_cache)

    with open(path_to_artifacts, "rb") as file:
        model = pickle.load(file)
        predict_logger.info("Pipeline loaded from %s.", path_to_artifacts)

        x_test = pd.read_csv(path_to_test)
        x_test = x_test.drop([ID_COL, TARGET_COL], axis=1)

        if predict_proba:
            prediction = model.predict_proba(x_test)[:, 1]
        else:
            prediction = model.predict(x_test)
        pd.DataFrame({"prediction": prediction}).to_csv(path_to_output)
        predict_logger.info("Saved predictions to %s.", path_to_output)


if __name__ == "__main__":
    ARTIFACTS_PATH = sys.argv[1]
    TEST_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3]
    predict(ARTIFACTS_PATH, TEST_PATH, OUTPUT_PATH)
