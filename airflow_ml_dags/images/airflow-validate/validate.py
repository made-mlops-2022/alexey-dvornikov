"""
Validation function.
"""
import os
import json
import pickle
import warnings
import click
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.simplefilter("ignore")


@click.command(name="validate")
@click.option("--input-dir", help="Directory of the validation data.")
@click.option("--model-dir", help="Directory of the serialized model.")
@click.option("--output-dir", help="Directory to store the metrics.")
def validate(input_dir: str, model_dir: str, output_dir: str) -> None:
    """
    Validate model and log metrics;
    :param input_dir: input directory;
    :param model_dir: model directory;
    :param output_dir: output directory;
    :return: None.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    model_name = os.listdir(model_dir)[0]
    run_id = model_name[5:-7]

    with mlflow.start_run(run_id=run_id):
        os.makedirs(name=output_dir, exist_ok=True)
        x_val = pd.read_csv(os.path.join(input_dir, "x_val.csv"))
        y_val = pd.read_csv(os.path.join(input_dir, "y_val.csv"))

        with open(os.path.join(model_dir, model_name), "rb") as file:
            model = pickle.load(file)

        y_pred = model.predict(x_val, )
        y_pred_proba = model.predict_proba(x_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred_proba),
        }

        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="UTF-8") as file:
            json.dump(metrics, file)

        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    validate()
