# credit risk modeling and portfolio stress testing system

this project implements a simplified credit risk modeling engine with portfolio stress testing capability.

it trains a machine learning model to predict borrower default risk and simulates macroeconomic shocks to evaluate portfolio exposure.

the system demonstrates practical financial risk analytics workflows used in banking and fintech environments.

## architecture

- xgboost for credit risk modeling
- sklearn for evaluation metrics
- shap for model interpretability
- fastapi for serving inference endpoints
- modular stress testing simulation

## features

- supervised default prediction model
- roc-auc evaluation
- portfolio stress simulation
- macro shock parameter adjustment
- api-based interaction
- structured financial dataset

## project structure

credit-risk-engine/
│
├── data/
│   └── sample_credit_data.csv
│
├── model.py
├── stress_test.py
├── app.py
├── requirements.txt
└── README.md

## how to run locally

### 1. clone the repository

git clone <your-repo-url>
cd credit-risk-engine

### 2. install dependencies

pip install -r requirements.txt

### 3. train the model

start the api server:

uvicorn app:app --reload

then open:

http://127.0.0.1:8000/train

this trains the model and returns roc-auc score.

### 4. run portfolio stress test

http://127.0.0.1:8000/stress-test?shock=0.2

this simulates a 20 percent macroeconomic shock to default probability.

## stress testing logic

1. train default prediction model
2. compute probability of default
3. apply macroeconomic shock multiplier
4. estimate expected portfolio loss

## use cases

- credit underwriting systems
- portfolio risk monitoring
- regulatory stress simulation
- fintech lending risk engines
- financial analytics platforms

## future improvements

- probability of default calibration
- loss given default modeling
- capital adequacy simulation
- monte carlo macro scenario testing
- model monitoring dashboard

---

this project demonstrates applied financial machine learning, risk modeling methodology, stress testing logic, and production-style api deployment.
