from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ✅ Ensure correct types
        age = float(data['Age'])
        sex = int(data['Sex'])
        pclass = int(data['Pclass'])
        embarked = int(data['Embarked'])

        # ✅ EXACT column order
        df = pd.DataFrame([[age, sex, pclass, embarked]],
                          columns=['Age', 'Sex', 'Pclass', 'Embarked'])

        print("INPUT DF:\n", df)

        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        print("Prediction:", prediction)
        print("Probabilities:", proba)

        return jsonify({
            'prediction': int(prediction),
            'probability_survive': float(proba[1])
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)