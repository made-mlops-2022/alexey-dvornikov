"""
Train script.
"""

import sys
import pickle
import warnings
import logging
from sklearn.metrics import roc_auc_score
from dataclass_ import TrainData
from logger import set_logger

warnings.simplefilter("ignore")

train_logger = logging.getLogger("train")


def train(data: TrainData) -> None:
    """
    Train function;
    :param data: TrainData object;
    :return: None.
    """
    cache_path = data.get_logging_path()
    set_logger(cache_path)

    data.split_data()
    model = data.get_model()

    x_train, y_train, x_val, y_val = data.get_data()
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    train_logger.info("ROC-AUC on validation: %f.", score)

    path_to_pickle = data.get_artifacts_path()
    with open(path_to_pickle, "wb") as file:
        pickle.dump(model, file)
        train_logger.info("Pipeline saved to %s.", path_to_pickle)


if __name__ == "__main__":
    CONFIG_PATH = sys.argv[1]
    train(TrainData(CONFIG_PATH))
