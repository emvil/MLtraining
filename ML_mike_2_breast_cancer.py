import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from ML_tools import display, metrics

# Logistic regression
# https://pythonhealthcare.org/2018/04/15/66-machine-learning-your-first-ml-model-using-logistic-regression-to-diagnose-breast-cancer/


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

# 63  Splitting data into training and test datasets
# https://pythonhealthcare.org/2018/04/14/63-machine-learning-splitting-data-into-training-and-test-sets/

x = data_set.data
y = data_set.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# random state is integer seed
# If this is omitted than a different seed will be used each time
print('size of train / test dataset: ')
print('Shape of X:', x.shape)
print('Shape of y:', y.shape)
print('Shape of X_train:', x_train.shape)
print('Shape of y_train:', y_train.shape)
print('Shape of X_test:', x_test.shape)
print('Shape of y_test:', y_test.shape)

# 65 feature scaling
# https://pythonhealthcare.org/2018/04/15/65-machine-learning-feature-scaling/

# initialise a new scaling object
sc = StandardScaler()
# set up the scaler just on the training set
sc.fit(x_train)
# apply the scaler to the training and test sets
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

# Run logistic regression model from sklearn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100, random_state=0)
lr.fit(x_train_std, y_train)
print(lr)

# predict the outcomes for the test set
y_pred = lr.predict(x_test_std)

# calculate an accuracy score: % of test cases correctly precdicted
correct = (y_test == y_pred).sum()
incorrect = (y_test != y_pred).sum()
accuracy = correct / (correct+incorrect) * 100
print('\nPercent Accuracy: %0.1f' %accuracy)

# detailed results
prediction = pd.DataFrame()
prediction['actual'] = data_set.target_names[y_test]
prediction['predicted'] = data_set.target_names[y_pred]
prediction['correct'] = prediction['actual'] == prediction['predicted']

print('\nDetailed results for first 20 tests:')
print(prediction.head(20))

# 67 performance metrics
# https://pythonhealthcare.org/2018/04/16/67-machine-learning-adding-standard-diagnostic-performance-metrics-to-a-ml-diagnosis-model/
# These metrics can be used when the outcome can be classified as true or false

performance = metrics.calculate_diagnostic_performance(y_test, y_pred)
display.print_dict_value_decimal(performance)



