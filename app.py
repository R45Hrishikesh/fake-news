import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer

with open('model.pkl', 'rb') as file:
    models = pickle.load(file)

LR = models["LogisticRegression"]
DT = models["DecisionTree"]
RF = models["RandomForest"]

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data['text']
    input_vector = vectorizer.transform([input_text])
    
    pred_lr = LR.predict(input_vector)
    pred_dt = DT.predict(input_vector)
    pred_rf = RF.predict(input_vector)

    # Convert predictions to human-readable format
    prediction_map = {0: "Fake News", 1: "Real News"}
    
    response = {
        "LogisticRegression": prediction_map[int(pred_lr[0])],
        "DecisionTree": prediction_map[int(pred_dt[0])],
        "RandomForest": prediction_map[int(pred_rf[0])]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
