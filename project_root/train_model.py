import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib, os, json

from app.preprocess import preprocess_data

# Load data
df = pd.read_csv("data/dataset_sorties_500k.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèles à entraîner
MODELS = {
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

metrics = {}

for name, model in MODELS.items():
    print(f"🚀 Entraînement modèle : {name}")
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.joblib")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    metrics[name] = {
        "accuracy": accuracy,
        "report": report
    }

# Save metrics
os.makedirs("model", exist_ok=True)
with open("model/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("✅ Tous les modèles sont entraînés et les métriques sauvegardées.")
