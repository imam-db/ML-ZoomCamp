import pickle
import pandas as pd

model_file = 'model1.bin'
dv_file = 'dv.bin'

model = pd.read_pickle(model_file)
dv = pd.read_pickle(dv_file)

customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 19.7}

X = dv.transform([customer])

y_pred = model.predict_proba(X)[0,1]

print('customer : {}'.format(customer))
print('probability : {}'.format(y_pred))