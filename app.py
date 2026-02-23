from fastapi import FastAPI
import pandas as pd
from model import train_model
from stress_test import simulate_stress

app = FastAPI()

DATA_PATH = "data/sample_credit_data.csv"

@app.get("/train")
def train():
    data = pd.read_csv(DATA_PATH)
    model, auc = train_model(data)
    return {"message": "model trained", "roc_auc": round(auc, 4)}

@app.get("/stress-test")
def stress(shock: float = 0.1):
    data = pd.read_csv(DATA_PATH)
    portfolio_pd = data["default"]
    expected_loss = simulate_stress(portfolio_pd, shock)
    return {
        "shock": shock,
        "expected_portfolio_loss": round(float(expected_loss), 4)
    }
