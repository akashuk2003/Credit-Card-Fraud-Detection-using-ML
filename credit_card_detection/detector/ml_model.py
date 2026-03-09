import joblib
import numpy as np

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_fraud(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return "FRAUD" if prediction[0] == 1 else "NOT FRAUD"