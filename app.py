from keras.models import load_model
from fastapi import FastAPI
import numpy as np
import uvicorn


model = load_model('bank_churn.keras')
app = FastAPI()

@app.post("/predict")
def predict():
    new_data = np.array([[1,0,43,3,500000,2,1,0,0,0,1,0]])
    y_predict = ['Churn' if i>=0.5 else 'Retained' for i in model.predict(new_data)[0]]
    return {"class": y_predict}


