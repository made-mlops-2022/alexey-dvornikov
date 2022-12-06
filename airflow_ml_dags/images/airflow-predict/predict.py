"""
Predicting function.
"""
import os
import click
import pandas as pd
import mlflow


@click.command("predict")
@click.option("--input-dir", help="Directory of the test data.")
@click.option("--output-dir", help="Directory to store the prediction.")
def predict(input_dir: str, output_dir: str) -> None:
    """
    Make prediction;
    :param input_dir: input directory;
    :param output_dir: output directory;
    :return: None.
    """
    os.makedirs(output_dir, exist_ok=True)
    x_test = pd.read_csv(os.path.join(input_dir, "train_data.csv"))

    mlflow.set_tracking_uri("http://localhost:5000")

    model = mlflow.pyfunc.load_model(
        model_uri="models:/LogisticRegression/Production"
    )

    y_pred = model.predict(x_test)
    pd.DataFrame(y_pred).to_csv(
        os.path.join(output_dir, "predictions.csv"), index=False
    )


if __name__ == "__main__":
    predict()
