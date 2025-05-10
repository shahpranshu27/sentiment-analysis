from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

class TextData(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(data: TextData):
    # Process the text
    text_vector = vectorizer.transform([data.text])
    
    # Make prediction
    prediction = model.predict(text_vector)
    probability = model.predict_proba(text_vector)
    
    # Return results
    return {
        "text": data.text,
        "sentiment": "positive" if prediction[0] == 1 else "negative",
        "confidence": float(np.max(probability))
    }

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API - Send a POST request to /predict with text"}