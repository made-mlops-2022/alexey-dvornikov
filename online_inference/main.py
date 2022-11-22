"""
App module.
"""
import warnings
import cloudpickle
import pandas as pd
from fastapi import FastAPI
from schema import WeatherFeatures

warnings.simplefilter("ignore")
app = FastAPI()
MODEL = None


@app.on_event("startup")
def load() -> None:
    """
    Load serialized model;
    :return: None.
    """
    global MODEL
    with open("./model.pkl", "rb") as file:
        MODEL = cloudpickle.load(file)


@app.get("/")
def root() -> dict:
    """
    Root.
    :return: status dict.
    """
    return {"ping": "pong!"}


@app.post("/health", status_code=200)
def health() -> bool:
    """
    Check if model is ready;
    :return: bool.
    """
    return MODEL is not None


@app.post("/predict")
async def predict(data: WeatherFeatures) -> dict:
    """
    Make prediction.
    :param data: WeatherFeatures data object;
    :return: prediction dict.
    """
    data_df = pd.DataFrame([data.dict()])
    prediction = MODEL.predict(data_df)
    return {
        "Rain": "YES" if prediction[0] == 1 else "NO",
    }
