from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Enable CORS for all routes
CORS(app)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # ✅ FORCE correct order
        df = pd.DataFrame([[ 
            data['Age'],
            data['Sex'],
            data['Pclass'],
            data['Embarked']
        ]], columns=['Age','Sex','Pclass','Embarked'])

        prediction = model.predict(df)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)