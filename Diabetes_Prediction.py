#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File name: Diabetes_Prediction.py
# Author: Hasani Perera
# Contact: heperera826@gmail.com
# Date created: 25/10/2021
# Date last modified: 02/11/2021
# Python Version: 3.8.5

# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Logging 
logging.basicConfig(filename='output.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s - %(name)s - %(threadName)s -  %(levelname)s -%(lineno)d - %(message)s')

# warning control - ignore all warnings
warnings.filterwarnings('ignore')

# Help Functions
# Exploratory Data Analysis
def data_eda(dataset):
    """
    INPUT:
    dataset - a pandas dataframe holding comma-separated values(csv)

    OUTPUT:
    return

    This function perform the exploratory data analysis for the dataset and
    provides the summary statistics and data visualisations to provide a general view of the dataset.

    """
    # summary statistics
    # information about the dataset (index data type, columns, non-null values, memory usage)
    dataset.info(verbose=True)
    # missing values of the dataset
    logging.info('EDA-Missing values\n{}'.format(dataset.isnull().sum()))
    # number of values in 'outcome' category
    logging.info('EDA-Response Variable\n{}'.format(dataset['Outcome'].value_counts()))
    # summary of each numerical attribute
    logging.info('EDA-Dataset Description\n{}'.format(dataset.describe().T))

    # data visualisation
    # histogram showing the general trends of the dataset
    dataset.hist(bins=50, figsize=(20,15))
    # pair plots showing the distribution of the features in dataset.
    #    diagonal plots are kernel density plots others are scatter-plots,
    #    the pairwise relation between attribute/features cannot clearly
    #    separate the two outcome class instances
    sns.pairplot(data=dataset, hue='Outcome',diag_kind='kde')

    # show plots
    plt.show(block=False)
    plt.pause(3)
    plt.close()

    return

# Model Data
def ml_model(df, response_var, test_size=.3, rand_state=42):
    """
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test

    OUTPUT:
    X - cleaned X matrix
    y - cleaned response
    my_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model

    This function models the data and provides the accuracy scores for the output.
    """
    # Split into explanatory and response variables
    X = df.drop(response_var, axis=1)
    y = df[response_var]

    # Split into train and test
    logging.info('Split Train and Test set\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state,
                                                        stratify=df[response_var])
    list_1 = []
    classifiers = ['Logistic Regression', 'SGD Classifier','K Neighbors Classifier',
                   'SVM', 'Decision Tree','Naive Bayes']
    models = [LogisticRegression(), SGDClassifier(), KNeighborsClassifier(),
              LinearSVC(), DecisionTreeClassifier(), GaussianNB()]
    for model in models:
        # Fit
        logging.info("Fit\n")
        model.fit(X_train,y_train)

        # Predict
        logging.info("Predict\n")
        y_test_pred = model.predict(X_test)

        # Evaluate
        logging.info("Evaluate\n")
        list_1.append(metrics.accuracy_score(y_test_pred, y_test))
    models_dataframe = pd.DataFrame(list_1, index=classifiers, columns=['Accuracy'])
    return models_dataframe


if __name__ == '__main__':
    # read data
    file_name = "diabetes.csv"
    diab_data = pd.read_csv(file_name)
    logging.info("Read Data")

    # perform EDA
    show_eda = data_eda(diab_data)
    logging.info(show_eda)

    # model data
    show_models = ml_model(diab_data, ['Outcome'])
    logging.info(show_models)





