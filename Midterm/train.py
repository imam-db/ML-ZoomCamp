import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

import pickle

def train_save(train_file, export_filename):
    df = pd.read_csv('data/'+train_file, index_col='PassengerId')
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

    df['IsAlone'] = (df.SibSp == 0) & (df.Parch == 0)
    df.Age = pd.cut(df.Age, [0,5,12,18,40,120], labels=['0-5', '5-12', '12-18', '18-40', '40-120'])
    df.Fare = pd.cut(df.Fare, [0,25,100,600], labels=['0-25', '25-100', '100-600'])

    X = df.drop(columns='Survived')
    y = df.Survived

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer([
        ('numeric', numerical_pipeline,['SibSp','Parch']),
        ('categoric', categorical_pipeline, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'IsAlone'])
    ])


    pipeline = Pipeline([
        ('prep', preprocessor),
        ('algo', KNeighborsClassifier())
    ])


    parameter = {
        'algo__n_neighbors': range(1, 51, 2),
        'algo__weights' : ['uniform', 'distance'],
        'algo__p' : [1,2]
    }

    model = GridSearchCV(pipeline, parameter, cv=3, n_jobs=-1, verbose=1)
    model.fit(X_train, y_train)


    print(model.best_params_)
    print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))


    #save model
    filename = export_filename
    pickle.dump(model, open(filename, 'wb'))
    print('Model file {} saved'.format(filename))


if __name__ == "__main__":
    train_save(train_file='train.csv', export_filename='titanic_model.pkl')
