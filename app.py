from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os
import sklearn

app = Flask(__name__)
CORS(app)

sklearn.set_config(assume_finite=True)

# Load model safely
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([[
            float(data['Age']),
            int(data['Sex']),
            int(data['Pclass']),
            int(data['Embarked'])
        ]], columns=['Age', 'Sex', 'Pclass', 'Embarked'])

        df = df.astype(float)

        prediction = model.predict(df)[0]

        try:
            proba = model.predict_proba(df)[0][1]
        except:
            proba = None

        return jsonify({
            "prediction": int(prediction),
            "survival_probability": proba
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)