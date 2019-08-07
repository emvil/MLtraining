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

# GET DATA
# DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = "datasets/housing"
# HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"
# -> downloaded


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Explore data
housing = load_housing_data()
print(housing.head())
print(housing.info())

print(housing["ocean_proximity"].value_counts())

# stats for each variable
print(housing.describe())

# display histograms of variables
housing.hist(bins=50, figsize=(20, 15))
plt.show()
