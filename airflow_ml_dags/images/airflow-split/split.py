"""
Splitting function.
"""
import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command(name="split")
@click.option("--input-dir", help="Directory of the preprocessed data.")
@click.option("--output-dir", help="Directory to store the splitted data.")
def split_data(input_dir: str, output_dir: str) -> None:
    """
    Split data into train and validation;
    :param input_dir: input directory;
    :param output_dir: output directory;
    :return: None.
    """
    os.makedirs(name=output_dir, exist_ok=True)
    preprocessed_x = pd.read_csv(os.path.join(input_dir, "train_data.csv"))
    preprocessed_y = pd.read_csv(os.path.join(input_dir, "train_target.csv"))
    x_train, x_val, y_train, y_val = train_test_split(
        preprocessed_x, preprocessed_y, test_size=0.2
    )
    x_train.to_csv(os.path.join(output_dir, "x_train.csv"), index=False)
    x_val.to_csv(os.path.join(output_dir, "x_val.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(output_dir, "y_val.csv"), index=False)


if __name__ == "__main__":
    split_data()
