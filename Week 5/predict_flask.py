import pickle

from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd


model_file = 'model1.bin'
dv_file = 'dv.bin'

model = pd.read_pickle(model_file)
dv = pd.read_pickle(dv_file)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)