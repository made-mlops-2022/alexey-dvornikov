"""
Requests module.
"""
import json
import warnings
import requests
import pandas as pd

warnings.simplefilter("ignore")

ID_COL = "row_id"
TARGET_COL = "raintomorrow"
TEST_PATH = "holdout.csv"

if __name__ == "__main__":
    x_test = pd.read_csv(TEST_PATH)

    numeric_cols = list(x_test.select_dtypes(exclude="object").columns)
    categorical_cols = list(x_test.select_dtypes(include="object").columns)

    x_test[categorical_cols] = x_test[categorical_cols].fillna(method="bfill")
    x_test[numeric_cols] = x_test[numeric_cols].fillna(method="bfill")
    x_test = x_test.drop([ID_COL, TARGET_COL], axis=1)

    records = x_test.to_dict(orient="records")
    for record in records:
        response = requests.post(
            "http://localhost:8000/predict", json.dumps(record), timeout=2
        )
        print(response.json())
