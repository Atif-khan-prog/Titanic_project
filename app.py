from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)