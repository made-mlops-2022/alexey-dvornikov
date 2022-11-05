"""
Train script.
"""

import pickle
import warnings
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from mlflow import log_metric, log_artifact
from sklearn.metrics import roc_auc_score
from dataclass_ import TrainData
from logger import set_logger

warnings.simplefilter("ignore")
train_logger = logging.getLogger("train")

CONFIG_PATH = "../config"
CONFIG_NAME = "config_1"


@hydra.main(
    version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME
)
def train(cfg: DictConfig) -> None:
    """
    Train function;
    :param cfg: TrainData object;
    :return: None.
    """
    print("[CONFIG]")
    print(OmegaConf.to_yaml(cfg))
    data = TrainData(cfg)
    cache_path = data.get_logging_path()
    set_logger(cache_path)

    data.split_data()
    model = data.get_model()

    x_train, y_train, x_val, y_val = data.get_data()
    model.fit(x_train, y_train)
    train_logger.info("Pipeline fitted.")

    y_pred = model.predict_proba(x_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    train_logger.info("ROC-AUC on validation: %f.", score)
    log_metric("roc-auc", score)

    path_to_pickle = data.get_artifacts_path()
    with open(path_to_pickle, "wb") as file:
        pickle.dump(model, file)
        train_logger.info("Pipeline saved to %s.", path_to_pickle)
    log_artifact(path_to_pickle)


if __name__ == "__main__":
    train()
