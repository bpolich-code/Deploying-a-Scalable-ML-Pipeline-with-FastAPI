from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from ml.data import process_data
from ml.model import load_model, inference
import os

app = FastAPI()

model_path = os.path.join('model', 'model.pkl')
encoder_path = os.path.join('model', 'encoder.pkl')

model = load_model(model_path)
encoder = load_model(encoder_path)

cat_features = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]


class CensusData(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example='Private')
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example='HS-grad')
    education_num: int = Field(..., alias='education-num', example=10)
    marital_status: str = Field(..., alias='marital-status', example='Married-civ-spouse')
    occupation: str = Field(..., example='Prof-specialty')
    relationship: str = Field(..., example='Husband')
    race: str = Field(..., example='White')
    sex: str = Field(..., example='Male')
    capital_gain: int = Field(..., alias='capital-gain', example=0)
    capital_loss: int = Field(..., alias='capital-loss', example=0)
    hours_per_week: int = Field(..., alias='hours-per-week', example=40)
    native_country: str = Field(..., alias='native-country', example='United-States')

    class Config:
        populate_by_name = True


@app.get('/')
async def root():
    return {'message': 'Welcome to the Census Income Prediction API!'}


@app.post('/predict')
async def predict(data: CensusData):
    input_dict = {
        'age': [data.age],
        'workclass': [data.workclass],
        'fnlgt': [data.fnlgt],
        'education': [data.education],
        'education-num': [data.education_num],
        'marital-status': [data.marital_status],
        'occupation': [data.occupation],
        'relationship': [data.relationship],
        'race': [data.race],
        'sex': [data.sex],
        'capital-gain': [data.capital_gain],
        'capital-loss': [data.capital_loss],
        'hours-per-week': [data.hours_per_week],
        'native-country': [data.native_country]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=None
    )
    
    pred = inference(model, X)
    
    prediction = '>50K' if pred[0] == 1 else '<=50K'
    
    return {'prediction': prediction}
