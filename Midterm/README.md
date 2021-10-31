## **Predicting the survival of titanic passengers.**


### Problem Description

The sinking of the Titanic is one of the most infamous shipwrecks in history. On its maiden voyage on April 15, 1912, the famous "unsinking" Titanic hit an iceberg and sank. Unfortunately, there were not enough lifeboats for everyone on board, killing 1,502 of the 2,224 passengers and crew. Survival involves some luck, but some groups seem more likely to survive than others. This problem explain to build a predictive model that answers the question what types of people are more likely to survive using passengers data. In this task, I will explain how the process of creating ML model until it deploy in the cloud. For the data can be checked at the following address: https://www.kaggle.com/c/titanic


### **EDA**

Can be checked on file notebook.ipynb.


### **Model Training**

Can be checked on file notebook.ipynb or training.py.


### **Exporting notebook to script**
File notebook.ipynb exported to 2 files. 1 files for training called training.py and 1 files for predict, called predict.py.


### **Model deployment**

Model is deployed with Flask.


### **Dependency and environment management**

For environment management, I use the feature from anaconda, namely conda environment. To use it is quite easy, from the anaconda prompt type the command "conda create --name name_of_environment python=3.8" , which means we will create a new environment with the name "name_of_environment", (we can replace it with another name) and at the same time will install python version 3.8 After that, we have to activate it to use it by typing the command "conda activate name_of_environment"
For the dependency library that we use, it can be put into a single file with the name "requirements.txt", which we will use to install in docker. To retrieve all installed libraries can use the command "pip freeze > requirements.txt".


### **Containerization**

This is the code from my Dockerfile

```
# get base image from dockerhub for python 3.8.12-slim
FROM python:3.8.12-slim 

# get working directory named “flask-app”
WORKDIR flask-app 

# copy folder data to folder named data in docker
COPY ./data ./data 

# copy folder templates to folder named templates in docker
COPY ./templates ./templates 

# copy all file in root folder
COPY requirements.txt titanic_model.pkl notebook.ipynb predict.py train.py ./ 

# get update
RUN apt-get update -y 

# get update for python
RUN apt-get install -y python3-pip python3-dev build-essential 

# install the requirements
RUN pip install -r requirements.txt 

# expose port 5000
EXPOSE 5000 

# configure python as entry point
ENTRYPOINT [ "python" ] 

# run file predict.py
CMD [ "predict.py" ] 
```

After preparing its Dockerfile. the next step is to build it, simply by typing the command 
```
docker build -t titanic-flask .
```
which "titanic-flask" will be the docker image for our application. 
After successfully built, it can be continued to running by typing 
```
docker run -p 5000:5000 titanic-flask
```
which means we will run a docker image with the name **titanic-flask** on port 5000.


### **Cloud deployment**
To deploy containers to Heroku is quite easy. Just a few short steps and it will automatically save to the Heroku cloud.
First make sure we install Heroku CLI. You can download it at the following link. We start by typing 
```
heroku login
```
And you will be prompted to enter your Heroku credentials. Once logged in, create an application by running: 
```
heroku create name_of_app
```
Next login to Heroku container registry by typing : 
```
heroku container:login
```
Which should produced **Login Succeded**. Next we push the container to Heroku using : 
```
heroku container:push web – app name_of_app
```
After the app container build, release it using : 
```
heroku container:release web –app name_of_app
```
Which will produce following **Releasing image web to name_of_app… done**. Once is releease we can check the result on the web by typing **name_of_app.herokuapp.com**. In this task, you can check my app in link **midterm-titanic.herokuapp.com**. It is simple app, contain link to download testing data in csv file and can be uploaded to check the result of probability the passenger is survived or not. 



