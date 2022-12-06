"""
Preprocessing function.
"""
import os
import click
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


@click.command(name="preprocess")
@click.option("--input-dir", help="Directory of the non-preprocessed data.")
@click.option("--output-dir", help="Directory to store the preprocessed data.")
def preprocess_data(
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Fill nans, scale and add polynomial features;
    :param input_dir: input directory;
    :param output_dir: output directory;
    :return: None.
    """
    os.makedirs(name=output_dir, exist_ok=True)
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    data = data.replace({"Yes": 1, "No": 0})
    target = target.replace({"Yes": 1, "No": 0})

    numeric_cols = list(data.select_dtypes(include=np.number).columns)
    data = data[numeric_cols]

    for column in data.columns:
        data[column] = data[column].fillna(np.mean(data[column]))

    scaler = StandardScaler()
    transformer = PolynomialFeatures()

    data = scaler.fit_transform(data)
    data = transformer.fit_transform(data)
    pd.DataFrame(data).to_csv(
        os.path.join(output_dir, "train_data.csv"), index=False
    )
    target.to_csv(os.path.join(output_dir, "train_target.csv"), index=False)


if __name__ == "__main__":
    preprocess_data()
