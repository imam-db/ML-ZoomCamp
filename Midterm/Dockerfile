FROM python:3.8-slim

WORKDIR flask-app

COPY ./data ./data

COPY ./templates ./templates

COPY requirements.txt titanic_model.pkl notebook.ipynb predict.py train.py ./

RUN pip install -r requirements.txt

CMD [ "python", "./predict.py" ]



