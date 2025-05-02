import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from app.preprocess import preprocess_data
from app.model import train_model
import joblib
import os

# Charger les données
df = pd.read_csv("data/dataset_sorties_500k.csv")

# Prétraiter les données
X, y = preprocess_data(df)
print(y.value_counts())

# Split pour évaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = train_model(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Calcul des métriques
report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)

# Sauvegarde des métriques dans model/metrics.txt
os.makedirs("model", exist_ok=True)
with open("model/metrics.txt", "w") as f:
    f.write(f"📊 Accuracy: {accuracy:.4f}\n\n")
    f.write("📈 Classification Report:\n")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            f.write(f"\nClasse: {label}\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")

# Affichage dans console
print("✅ Modèle entraîné et métriques sauvegardées dans model/metrics.txt")
print(f"🎯 Accuracy: {accuracy:.4f}")
print("📁 Rapport détaillé : model/metrics.txt")
