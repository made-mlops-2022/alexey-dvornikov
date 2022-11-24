"""
App module.
"""
import os
import warnings
import cloudpickle
import yadisk
import pandas as pd
from fastapi import FastAPI
from schema import WeatherFeatures

warnings.simplefilter("ignore")
app = FastAPI()

TOKEN = None
MODEL_PATH = None
MODEL = None


@app.on_event("startup")
def load() -> None:
    """
    Load serialized model;
    :return: None.
    """
    global MODEL, MODEL_PATH, TOKEN
    TOKEN = os.getenv("TOKEN")
    MODEL_PATH = os.getenv("MODEL_PATH")
    source = yadisk.YaDisk(token=TOKEN)
    source.download(MODEL_PATH, "./model.pkl")
    with open("./model.pkl", "rb") as file:
        MODEL = cloudpickle.load(file)


@app.get("/")
def root() -> dict:
    """
    Root.
    :return: status dict.
    """
    return {"ping": "pong!"}


@app.post("/health")
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
