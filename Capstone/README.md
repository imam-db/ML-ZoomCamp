## **Predict whether mushroom is edible or poisonous.**


### Problem Description

A mushroom or toadstool is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground, on soil, or on its food source.

The standard for the name "mushroom" is the cultivated white button mushroom, Agaricus bisporus; hence the word "mushroom" is most often applied to those fungi (Basidiomycota, Agaricomycetes) that have a stem (stipe), a cap (pileus), and gills (lamellae, sing. lamella) on the underside of the cap.

Many mushrooms are not poisonous and some are edible.

Edible mushrooms are the fleshy and edible fruit bodies of several species of macrofungi (fungi which bear fruiting structures that are large enough to be seen with the naked eye). They can appear either below ground (hypogeous) or above ground (epigeous) where they may be picked by hand.

Many mushroom species produce secondary metabolites that can be toxic, mind-altering, antibiotic, antiviral, or bioluminescent. Although there are only a small number of deadly species, several others can cause particularly severe and unpleasant symptoms.

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.

You can check the data from this [link](https://www.openml.org/d/24)


### **EDA**

Can be checked on file notebook.ipynb.


### **Model Training**

Can be checked on file notebook.ipynb or training.py.


### **Exporting notebook to script**
File notebook.ipynb exported to 2 files. 1 files for training called training.py and 1 files for predict, called predict.py.


### **Model deployment**

Model is deployed with Flask.


### **Dependency and environment management**

For environment management, I use the feature from anaconda, namely conda environment. To use it is quite easy, from the anaconda prompt type the command 
```
conda create --name name_of_environment python=3.8
```
which means we will create a new environment with the name "name_of_environment", (we can replace it with another name) and at the same time will install python version 3.8 After that, we have to activate it to use it by typing the command 
```
conda activate name_of_environment
```
For the dependency library that we use, it can be put into a single file with the name "requirements.txt", which we will use to install in docker. To retrieve all installed libraries can use the command 
```
pip freeze > requirements.txt
```


### **Containerization**

This is the code from my Dockerfile

```
# get base image from dockerhub for python 3.8.12-slim
FROM python:3.8.12-slim 

# get working directory named “flask-app”
WORKDIR flask-app 

# copy folder templates to folder named templates in docker
COPY ./templates ./templates 

# copy all file in root folder
COPY requirements.txt mushroom_model.pkl notebook.ipynb predict.py train.py ./ 

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
docker build -t mushroom-flask .
```
which "titanic-flask" will be the docker image for our application. 
After successfully built, it can be continued to running by typing 
```
docker run -p 5000:5000 mushroom-flask
```
which means we will run a docker image with the name **mushroom-flask** on port 5000.


### **Cloud deployment**
To deploy containers to Heroku is quite easy. Just a few short steps and it will automatically save to the Heroku cloud.
First make sure we install Heroku CLI. You can download it at the following [link](https://devcenter.heroku.com/articles/heroku-cli). We start by typing 
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
heroku container:push web –-app name_of_app
```
After the app container build, release it using : 
```
heroku container:release web –-app name_of_app
```
Which will produce following **Releasing image web to name_of_app… done**. Once is releease we can check the result on the web by typing **name_of_app.herokuapp.com**. In this task, you can check my app in link [capstone-mushroom.herokuapp.com](http://capstone-mushroom.herokuapp.com/). It is simple app, contain link to download testing data in csv file and can be uploaded to check the result whether mushroom is edible or poisonous.
You can also check via [streamlit](https://share.streamlit.io/imam-db/ml-zoomcamp/Capstone/streamlit.py) 



