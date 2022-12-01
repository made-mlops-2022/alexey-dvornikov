"""
Generating function.
"""
import os
import warnings
import click
import pandas as pd
from sdv.tabular import GaussianCopula

warnings.simplefilter("ignore")


@click.command(name="generate")
@click.option("--output-dir", help="Directory to store the fake data.")
def generate_data(output_dir: str, n_samples: int = 10000) -> None:
    """
    Generate fake data, based on the given data;
    :param output_dir: output directory;
    :param n_samples: number of samples;
    :return: None.
    """
    os.makedirs(name=output_dir, exist_ok=True)
    real_data = pd.read_csv("holdout.csv")
    target_column = real_data.columns[-1]
    id_column = real_data.columns[0]

    model = GaussianCopula()
    model.fit(real_data)
    fake_data = model.sample(n_samples)

    fake_target = fake_data[target_column]
    fake_data = fake_data.drop([target_column, id_column], axis=1)

    fake_data.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    fake_target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == "__main__":
    generate_data()
