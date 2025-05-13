import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

# 1. Charger les données
df = pd.read_csv("data/dataset_sorties_500k.csv").dropna()

# 2. Ajouter des colonnes simulées (si tu veux les utiliser)
import random
df["moment"] = random.choices(["soir", "nuit", "après-midi"], k=len(df))
df["ambiance"] = random.choices(["chill", "romantique", "festive"], k=len(df))
df["budget"] = random.choices(["petit", "moyen", "élevé"], k=len(df))
df["avec"] = random.choices(["amis", "amoureux", "famille", "solo"], k=len(df))

# 3. Colonnes à encoder
features = ["genre", "ville", "moment", "ambiance", "budget", "avec"]
df[features] = df[features].apply(lambda col: col.str.lower())

# 4. Encoder avec sortie sparse (par défaut)
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded_sparse = encoder.fit_transform(df[features])  # --> type csr_matrix

# 5. Sauvegarde
os.makedirs("model", exist_ok=True)
joblib.dump(encoder, "model/similarity_encoder.joblib")
joblib.dump(X_encoded_sparse, "model/similarity_encoded_array.joblib")

print("✅ Entraînement du système de similarité terminé (sparse format utilisé proprement).")
