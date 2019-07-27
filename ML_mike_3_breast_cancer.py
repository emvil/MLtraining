import numpy as np
from ML_tools import metrics, preprocessing, display
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

# References:
# Author Mike Allen
# https://pythonhealthcare.org/2018/04/15/65-machine-learning-feature-scaling/
# https://pythonhealthcare.org/2018/04/16/67-machine-learning-adding-standard-diagnostic-performance-metrics-to-a-ml-diagnosis-model/
# https://pythonhealthcare.org/2018/04/17/69-machine-learning-how-do-you-know-if-you-have-gathered-enough-data-by-using-learning-rates/
#

# Data loading and feature overview
data_set = datasets.load_breast_cancer()
print('content of dataset: ')
print(list(data_set))

print('feature names: ')
print(data_set.feature_names)

print('data content: ')
print(data_set.data[0: 2])
# 30 columns for features

print('categories: ')
print(data_set.target_names)

x = data_set.data
y = data_set.target

# [69} Learning rates

number_of_training_points = range(25, 450, 25)
# set up lists to save results
training_accuracy = []
test_accuracy = []
n_results = []

for n in number_of_training_points:
    # repeat ML model/prediction 1000 times for each number of runs
    for i in range(1000):
        # Repeat test x times per level of split
        n_results.append(n)

        # split data into training and test sets
        x_train, x_test, y_train, y_test = preprocessing.split_data(x, y, training_split_rate=0.75, n_points=n,
                                                                    display=0)
        # Scale Features
        # initialise a new scaling object
        sc = StandardScaler()
        # set up the scaler just on the training set
        sc.fit(x_train)
        # apply the scaler to the training and test sets
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)

        # Run logistic regression model
        lr = LogisticRegression(C=100, random_state=0)
        lr.fit(x_train_std, y_train)

        # Record performances on training set
        y_pred = lr.predict(x_train_std)
        performance = metrics.calculate_diagnostic_performance(y_train, y_pred)
        training_accuracy.append(performance['accuracy'])

        # Record performances on test set
        y_pred = lr.predict(x_test_std)
        performance = metrics.calculate_diagnostic_performance(y_test, y_pred)
        test_accuracy.append(performance['accuracy'])

results = pd.DataFrame()
results['n'] = n_results
results['training_accuracy'] = training_accuracy
results['test_accuracy'] = test_accuracy
summary = results.groupby('n').median()

print(summary)
display.chart_learning_rate(list(summary.index), summary['training_accuracy'], summary['test_accuracy'])

