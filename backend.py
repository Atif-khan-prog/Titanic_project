from app import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    result = model.predict(df)[0]
    
    return jsonify({'prediction': int(result)})

if __name__ == '__main__':
    app.run(debug=True)