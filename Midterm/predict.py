import pandas as pd
import pickle
from flask import Flask, render_template, request
from waitress import serve
import os

app = Flask(__name__)
filename = 'titanic_model.pkl'
model = pickle.load(open(filename, 'rb'))
port = int(os.environ.get("PORT", 5000))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        csv_file = request.files.get('file')
        X_test = pd.read_csv(csv_file, index_col="PassengerId")
        X_test['IsAlone'] = (X_test.SibSp == 0) & (X_test.Parch == 0)
        X_test.Age = pd.cut(X_test.Age, [0,5,12,18,40,120], labels=['0-5', '5-12', '12-18', '18-40', '40-120'])
        X_test.Fare = pd.cut(X_test.Fare, [0,25,100,600], labels=['0-25', '25-100', '100-600'])
        X_test['IsSurvived'] = model.predict(X_test)
        return X_test.to_html()

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=port)
    #serve(app, host='0.0.0.0', port=port)