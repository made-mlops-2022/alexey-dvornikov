"""
App module.
"""
import os
import warnings
import logging
import cloudpickle
import yadisk
import pandas as pd
from config import TOKEN  # Yandex Disk API token
from fastapi import FastAPI
from schema import WeatherFeatures

warnings.simplefilter("ignore")
app = FastAPI()

logger = logging.getLogger()
logging.basicConfig(
    filename="./appcache.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

MODEL_PATH = None
MODEL = None


@app.on_event("startup")
def load() -> None:
    """
    Load serialized model;
    :return: None.
    """
    global MODEL, MODEL_PATH
    MODEL_PATH = os.getenv("MODEL_PATH")

    logger.info("Application started.")

    logger.info("Downloading model...")
    source = yadisk.YaDisk(token=TOKEN)
    source.download(MODEL_PATH, "./model.pkl")
    logger.info("Model downloaded.")

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
def health() -> dict:
    """
    Check if model is ready;
    :return: bool.
    """
    return {"model": "ready"}


@app.post("/predict")
async def predict(data: WeatherFeatures) -> dict:
    """
    Make prediction.
    :param data: WeatherFeatures data object;
    :return: prediction dict.
    """
    logger.info(f"Got request from user:\n{data}")
    data_df = pd.DataFrame([data.dict()])
    prediction = MODEL.predict(data_df)
    logger.info(f"Sending prediction:\n{prediction[0]}")
    return {
        "Rain": "YES" if prediction[0] == 1 else "NO",
    }
