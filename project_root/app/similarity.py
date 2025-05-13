import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import joblib

# ✅ Encoder un utilisateur en vecteur one-hot
def encode_user_data(data):
    df = pd.DataFrame([data])
    df = df[["genre", "ville", "moment", "ambiance", "budget", "avec"]].apply(lambda col: col.str.lower())
    encoder = joblib.load("model/similarity_encoder.joblib")
    return encoder.transform(df)  # renvoie un vecteur sparse

# ✅ Rechercher les lieux les plus similaires
def get_similar_places(user_input, top_k=5):
    user_vector = encode_user_data(user_input)  # 1 x n sparse

    # 🔁 Charger la matrice encodée des lieux (sparse matrix)
    encoded_places = joblib.load("model/similarity_encoded_array.joblib")  # n x m sparse

    # 🧠 Calcul des similarités
    similarities = cosine_similarity(user_vector, encoded_places)[0]  # shape: (n,)

    # 📊 Récupérer les indices des lieux les plus proches
    top_indices = similarities.argsort()[::-1][:top_k]

    # 🔁 Charger les infos originales
    places_df = pd.read_csv("data/dataset_sorties_500k.csv")[["nom_lieu", "type", "note_moyenne"]]

    # 🧾 Extraire les lignes correspondantes
    top_places = places_df.iloc[top_indices]

    return top_places.to_dict(orient="records")
