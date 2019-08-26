# References:
# Book: Hands_on Machine Learning with Scikit-Learn and TensorFlow, Aurélien Géron, 2017.
# Chapter 2: End to end Machine Learning Project

# Objective: predict the district's median housing price in any block group of California.
# Current solution: manually estimated by experts, with a typical error of 15%.
# Framework: supervised multivariate regression problem, using batch learning.
# Performance measure: RMSE or MAE

import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
from ML_tools import preprocessing
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer

# GET DATA
# DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = "datasets/housing"
# HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
# -> downloaded


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Explore data
housing_dataset = load_housing_data()
print(housing_dataset.head())
print(housing_dataset.info())

print(housing_dataset["ocean_proximity"].value_counts())

# stats for each variable
print(housing_dataset.describe())

# display histograms of variables
housing_dataset.hist(bins=50, figsize=(20, 15))
# plt.show()

# CREATE A TEST SET
housing_train, housing_test, tmp, tmp2 = preprocessing.split_data(housing_dataset, None, training_split_rate=0.80, display=True)

# PREPARE THE DATA FOR ML ALGORITHMS (p.59)
housing_labels = housing_train.loc[:, "median_house_value"].copy()
housing = housing_train.drop(labels="median_house_value", axis=1)
print(housing_labels.head())
print(housing.info())

# Data cleaning
""" 3 options to deal with missing values
- get rid of corresponding rows
- get rid of the whole attribute (columns)
- set the values to some value (zero, mean, median, etc)
"""
median_bedrooms = housing['total_bedrooms'].median()
print(median_bedrooms)
housing['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print(housing.info())
# Data cleaning : other method
imputer = Imputer(strategy="median")
#   compute df with only numerical values
housing_num = housing.drop('ocean_proximity', axis=1)
# fit the imputer with the training data
imputer.fit(housing_num)
print(imputer.statistics_)
print(housing_num.median().values)
# apply the imputer to all the numerical values (result is a numpy array)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# HANDLING TEXT AND CATEGORICAL ATTRIBUTES (p.62)
# to convert text labels into numbers
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)
# One-hot encoding : binary classes from the encoded category vector, output is sparse matrix
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(housing_cat_1hot)
# do integer encoding and binary encoding more quickly, output is a dense numpy array by default
# unless you add sparse_output=True
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

# CUSTOM TRANSFORMERS (p.64)

