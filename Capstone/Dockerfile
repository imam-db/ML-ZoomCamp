FROM python:3.8.12-slim

WORKDIR mushroom-app

COPY ./templates ./templates

COPY requirements.txt mushroom_model.pkl notebook.ipynb predict.py train.py ./

RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT [ "python" ]

CMD [ "predict.py" ]