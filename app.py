from flask import Flask, jsonify, render_template
from pymongo import MongoClient

app = Flask(__name__)

# Configuration de la connexion à MongoDB
client = MongoClient("mongodb://127.0.0.1:27017/")
db = client["bigData"]
collection = db["predictions"]

@app.route('/')
def index():
    return "Bienvenue sur l'application de prédiction de sentiments Twitter!"

def map_prediction_to_label(prediction):
    labels = {
        0: "Negative",
        1: "Positive",
        2: "Neutral",
        3: "Irrelevant"
    }
    return labels.get(prediction, "Unknown")

@app.route('/predictions')
def predictions():
    data = list(collection.find({}, {'_id': 0, 'Tweet ID': 1, 'Tweet content': 1, 'prediction': 1}))
    for item in data:
        item['prediction'] = map_prediction_to_label(item['prediction'])
    return render_template('predictions.html', predictions=data)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
