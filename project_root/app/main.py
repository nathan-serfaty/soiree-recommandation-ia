from flask import Flask, request, jsonify, render_template, send_file
import traceback
from app.model import predict_with_model
import joblib
from app.similarity import get_similar_places
import json

app = Flask(__name__, template_folder="templates")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/download', methods=['GET'])
def download_csv():
    try:
        with open("model/resultats_du_jour.json", "r", encoding="utf-8") as f:
            result = json.load(f)
        result["download_url"] = "/static/recommandations_du_jour.csv"
        return jsonify(result)
    except FileNotFoundError:
        return jsonify({"error": "Aucune recommandation disponible pour le moment."}), 404


@app.route('/predict', methods=['POST'])
def get_recommendation():
    data = request.get_json()
    try:
        age = int(data['age'])
        genre = data['genre']
        ville = data['ville']
        preference = data['preference']
        model_name = data.get("model", "xgboost")

        result = predict_with_model(model_name, age, genre, ville, preference)
        return jsonify(result)
    except Exception as e:
        print("❌ Erreur backend :")
        traceback.print_exc()  # Affiche l'erreur complète dans le terminal
        return jsonify({"error": str(e)}), 400
@app.route('/labels', methods=['GET'])
def get_labels():
    def load_encoder_labels(name):
        return list(joblib.load(f"model/{name}_encoder.joblib").classes_)

    labels = {
        "genres": load_encoder_labels("genre"),
        "villes": load_encoder_labels("ville"),
        "preferences": load_encoder_labels("preference")
    }
    return jsonify(labels)
@app.route('/similar', methods=['POST'])
def get_similar():
    from app.similarity import get_similar_places
    data = request.get_json()
    try:
        results = get_similar_places(data)
        return jsonify({"lieux": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
