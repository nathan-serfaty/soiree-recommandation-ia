import pandas as pd
import joblib
import json

def load_encoders():
    return {
        "genre": joblib.load("model/genre_encoder.joblib"),
        "ville": joblib.load("model/ville_encoder.joblib"),
        "preference": joblib.load("model/preference_encoder.joblib"),
    }

def predict_with_model(model_name, age, genre, ville, preference):
    # Load model
    model = joblib.load(f"model/{model_name}.joblib")

    # Load encoders
    encoders = load_encoders()
    target_encoder = joblib.load("model/target_encoder.joblib")

    # Input encoding
    input_data = pd.DataFrame([{
        "age": age,
        "genre": encoders["genre"].transform([genre.lower()])[0],
        "ville": encoders["ville"].transform([ville.lower()])[0],
        "preference": encoders["preference"].transform([preference.lower()])[0]
    }])

    # Prediction
    pred_class = model.predict(input_data)[0]
    place_type = target_encoder.inverse_transform([pred_class])[0]

    # Get top places
    places_df = pd.read_csv("data/dataset_sorties_500k.csv")[["nom_lieu", "type", "note_moyenne"]]
    filtered_places = places_df[places_df["type"] == place_type][["nom_lieu", "note_moyenne"]]
    filtered_places.to_csv("static/recommandations_du_jour.csv", index=False)

    top_places = filtered_places.sort_values(by="note_moyenne", ascending=False).head(5)

    # ✅ Sauvegarde complète pour /download
    with open("model/resultats_du_jour.json", "w", encoding="utf-8") as f:
        json.dump({
            "type": place_type,
            "model": model_name,
            "lieux": top_places.to_dict(orient="records")
        }, f, ensure_ascii=False, indent=2)

    # ✅ Charger les métriques
    with open("model/metrics.json", "r") as f:
        metrics_dict = json.load(f)

    metrics = metrics_dict.get(model_name, {})
    accuracy = metrics.get("accuracy", "N/A")
    classification = metrics.get("report", {})

    # Format classification report
    formatted_report = "\n".join([
        f"{label} — précision: {round(values.get('precision',0)*100, 2)}%, rappel: {round(values.get('recall',0)*100, 2)}%, F1: {round(values.get('f1-score',0)*100, 2)}%"
        for label, values in classification.items() if isinstance(values, dict)
    ])

    return {
        "type": place_type,
        "lieux": top_places.to_dict(orient="records"),
        "model": model_name,
        "accuracy": round(accuracy * 100, 2),
        "metrics": formatted_report
    }
