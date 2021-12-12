# import package
import pandas as pd
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV


def train_save(train_file, export_filename):
    df = pd.read_csv(train_file)

    # change object value to ordinal
    df_ordinal = df.copy()
    labelencoder = preprocessing.LabelEncoder()
    for column in df_ordinal.columns:
        df_ordinal[column] = labelencoder.fit_transform(df_ordinal[column])

    columns_X = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']
    
    # Getting X and y
    X = df_ordinal[columns_X]
    y = df_ordinal['class']

    # Splitting data
    X_fulltrain, X_test, y_fulltrain, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_fulltrain, y_fulltrain, random_state=42, test_size=0.25)

    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier()

    param_grid = {'criterion':['gini','entropy'],'max_depth':[None,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

    # defining parameter range
    grid = GridSearchCV(dt, param_grid, cv=10, scoring='accuracy', return_train_score=False,verbose=1)
    
    # fitting the model for grid search
    grid_search=grid.fit(X_train, y_train)

    print(grid_search.best_params_)
    print(grid_search.score(X_train, y_train), grid_search.best_score_, grid_search.score(X_val, y_val))


    #save model
    filename = export_filename
    pickle.dump(grid_search, open(filename, 'wb'))
    print('Model file {} saved'.format(filename))


if __name__ == "__main__":
    train_save(train_file='dataset_24_mushroom.csv', export_filename='mushroom_model.pkl')
