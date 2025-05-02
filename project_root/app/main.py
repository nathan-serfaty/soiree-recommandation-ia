from flask import Flask, request, jsonify, render_template
from app.model import load_model_and_encoders, predict
from flask import send_file

app = Flask(__name__, template_folder="templates")

model, encoders, target_encoder, places_df = load_model_and_encoders()

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/download', methods=['GET'])
def download_csv():
    try:
        return send_file("model/recommandations_du_jour.csv", as_attachment=True)
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

        result = predict(model, encoders, target_encoder, places_df, age, genre, ville, preference)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5050)

