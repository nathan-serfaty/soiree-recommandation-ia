import pandas as pd
import joblib
from xgboost import XGBClassifier
import random

def train_model(X, y):
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X, y)
    joblib.dump(model, "model/model.joblib")
    return model

def load_model_and_encoders():
    model = joblib.load("model/model.joblib")
    encoders = {
        "genre": joblib.load("model/genre_encoder.joblib"),
        "ville": joblib.load("model/ville_encoder.joblib"),
        "preference": joblib.load("model/preference_encoder.joblib"),
    }
    target_encoder = joblib.load("model/target_encoder.joblib")
    places_df = pd.read_csv("data/dataset_sorties_500k.csv")[["nom_lieu", "type", "note_moyenne"]]
    return model, encoders, target_encoder, places_df


def predict(model, encoders, target_encoder, places_df, age, genre, ville, preference):
    try:
        input_data = pd.DataFrame([{
            "age": age,
            "genre": encoders["genre"].transform([genre])[0],
            "ville": encoders["ville"].transform([ville])[0],
            "preference": encoders["preference"].transform([preference])[0]
        }])
    except ValueError as e:
        raise ValueError(f"Valeur inconnue dans les inputs utilisateur : {e}")


    pred_class = model.predict(input_data)[0]
    place_type = target_encoder.inverse_transform([pred_class])[0]

    # Filtrer tous les lieux correspondant au type prédit
    filtered_places = places_df[places_df['type'] == place_type][["nom_lieu", "note_moyenne"]]

    # Export CSV
    filtered_places.to_csv("model/recommandations_du_jour.csv", index=False)

    # Retourne les 5 meilleurs à afficher côté front
    top_places = filtered_places.sort_values(by="note_moyenne", ascending=False).head(5)

    return {
        "type": place_type,
        "lieux": top_places.to_dict(orient="records")
    }
