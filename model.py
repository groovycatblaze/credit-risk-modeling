import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_model(data):
    X = data.drop("default", axis=1)
    y = data["default"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = XGBClassifier()
    model.fit(X_train, y_train)
    
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    
    return model, auc
