"""
Schema module.
"""
from typing import Literal
from pydantic import BaseModel, validator


class WeatherFeatures(BaseModel):
    mintemp: float
    maxtemp: float
    rainfall: float
    evaporation: float
    sunshine: float
    windgustdir: Literal[
        "W",
        "NNW",
        "ENE",
        "SSE",
        "S",
        "NE",
        "SSW",
        "N",
        "WSW",
        "SE",
        "ESE",
        "E",
        "NW",
        "NNE",
        "SW",
        "WNW",
    ]
    windgustspeed: float
    winddir9am: Literal[
        "W",
        "NNW",
        "ENE",
        "SSE",
        "S",
        "NE",
        "SSW",
        "N",
        "WSW",
        "SE",
        "ESE",
        "E",
        "NW",
        "NNE",
        "SW",
        "WNW",
    ]
    winddir3pm: Literal[
        "W",
        "NNW",
        "ENE",
        "SSE",
        "S",
        "NE",
        "SSW",
        "N",
        "WSW",
        "SE",
        "ESE",
        "E",
        "NW",
        "NNE",
        "SW",
        "WNW",
    ]
    windspeed9am: float
    windspeed3pm: float
    humidity9am: float
    humidity3pm: float
    pressure9am: float
    pressure3pm: float
    cloud9am: float
    cloud3pm: float
    temp9am: float
    temp3pm: float
    raintoday: Literal["Yes", "No"]

    @validator("mintemp", "maxtemp", "temp9am", "temp3pm")
    def temp_validator(cls, value):
        if value < -25 or value > 60:
            raise ValueError("Wrong temperature value")
        return value

    @validator("windgustspeed", "windspeed9am", "windspeed3pm")
    def wind_validator(cls, value):
        if value < 0 or value > 150:
            raise ValueError("Wrong wind speed value")
        return value

    @validator("humidity9am", "humidity3pm")
    def humidity_validator(cls, value):
        if value < 0 or value > 100:
            raise ValueError("Wrong humidity value")
        return value

    @validator("pressure9am", "pressure3pm")
    def pressure_validator(cls, value):
        if value < 950 or value > 1100:
            raise ValueError("Wrong pressure value")
        return value

    @validator("cloud9am", "cloud3pm")
    def cloud_validator(cls, value):
        if value < 0 or value > 15:
            raise ValueError("Wrong cloud value")
        return value

    @validator("rainfall")
    def rainfall_validator(cls, value):
        if value < 0 or value > 100:
            raise ValueError("Wrong rainfall value")
        return value

    @validator("evaporation")
    def evaporation_validator(cls, value):
        if value < 0 or value > 90:
            raise ValueError("Wrong evaporation value")
        return value

    @validator("sunshine")
    def sunshine_validator(cls, value):
        if value < 0 or value > 20:
            raise ValueError("Wrong sunshine value")
        return value
