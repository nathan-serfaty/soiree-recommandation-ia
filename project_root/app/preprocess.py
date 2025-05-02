import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data(df):
    features = ["age", "genre", "ville", "preference"]
    target = "type"

    df = df[features + [target]].dropna()
    encoders = {}
    for col in features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        joblib.dump(le, f"model/{col}_encoder.joblib")

    target_encoder = LabelEncoder()
    df[target] = target_encoder.fit_transform(df[target])
    joblib.dump(target_encoder, "model/target_encoder.joblib")

    return df[features], df[target]
