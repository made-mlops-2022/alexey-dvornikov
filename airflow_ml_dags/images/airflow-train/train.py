"""
Training function.
"""
import os
import pickle
import warnings
import click
import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression

warnings.simplefilter("ignore")


@click.command(name="train")
@click.option("--input-dir", help="Directory of the train data.")
@click.option(
    "--output-dir",
    help="Directory to store the serialized model.",
)
def train(input_dir: str, output_dir: str) -> None:
    """
    Train model;
    :param input_dir: input directory;
    :param output_dir: output directory;
    :return: None.
    """
    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run() as run:
        os.makedirs(name=output_dir, exist_ok=True)
        x_train = pd.read_csv(os.path.join(input_dir, "x_train.csv"))
        y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv"))

        model = LogisticRegression(max_iter=1000, C=10.0)
        model.fit(x_train, y_train)

        with open(
            os.path.join(output_dir, f"model{run.info.run_id}.pickle"), "wb"
        ) as file:
            pickle.dump(model, file)

        model_parameters = model.get_params()
        mlflow.log_params(model_parameters)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="LogisticRegression",
        )


if __name__ == "__main__":
    train()
