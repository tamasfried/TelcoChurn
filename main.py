from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load("models/churn_model.joblib")

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Telco Churn Prediction API!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    data = data.model_dump()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0])
    }