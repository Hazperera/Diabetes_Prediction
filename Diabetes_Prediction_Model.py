#!/usr/bin/env python
# -*- coding: utf-8 -*-

# File name: Diabetes_Prediction_Model.py
# Author: Hasani Perera
# Contact: heperera826@gmail.com
# Date created: 25/10/2021
# Date last modified: 27/10/2021
# Python Version: 3.8.5

# import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import KFold
from sklearn import metrics
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# #read data
diab_data = pd.read_csv("diabetes.csv")
print(diab_data.head())

# ##EXPLORATORY DATA ANALYSIS
# #dataframe information
# diab_data.info(verbose=True)
# print("number of rows:{rows} , number of columns:{cols}".format(rows=diab_data.shape[0], cols=diab_data.shape[1]))
# print(diab_data.columns)
#
# # #missing values
# print(diab_data.isnull().sum())
# # #number of subjects with each attribute
# print('{outcome},{bp}'.format(outcome=diab_data['Outcome'].value_counts(),
#                               bp=diab_data['BloodPressure'].value_counts().head(5)))

# # #basic statistics
# print(diab_data.describe())
# print(diab_data.describe().T)

# #basic plots
# #counts of observations in "outcome" category
sns.countplot(x='Outcome',data=diab_data, palette="Blues_d")
# #histograms
histo = diab_data.hist(figsize=(10,8))
# #distribution of the features in dataset
sns.pairplot(data=diab_data, hue='Outcome', diag_kind='kde')
# #diagonal plots are kernel density plots
# #scatter-plots - the pairwise relation between attribute/features
# - no two attributes can clearly seperate the two outcome-class instances
# plt.show()


## MODELING

# #split into explanatory and response variables
X= diab_data.drop('Outcome',axis = 1)
y= diab_data['Outcome']
test_size = 0.3
random_state=42

# #split into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size,
                                                    random_state=random_state)

my_model = LinearRegression(normalize=True) # Instantiate
my_model.fit(X_train, y_train) #Fit

# #predict using my model
y_test_preds = my_model.predict(X_test)
y_train_preds = my_model.predict(X_train)

# #score using my model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)

print("The r-squared score for the {} was {} on {} values.".format(my_model,
    r2_score(y_test, y_test_preds), len(y_test)))

# print(X_train.head())
# print(X_test.head())
# print(y_train.head())
# print(y_test.head())

# #most influential coefficients of the model - along with the name of the variable attached to the coefficient
coefs_df = pd.DataFrame()
coefs_df['vars'] = X_train.columns
coefs_df['coefs'] = my_model.coef_
coefs_df['abs_coefs'] = np.abs(my_model.coef_)
coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
print("\n most influential features \n",coefs_df.head(10))








