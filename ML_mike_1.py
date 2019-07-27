# 61 machine learning : the iris data set
# https://pythonhealthcare.org/2018/04/14/61-machine-learning-the-iris-data-set/

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
print('iris content: ')
print(list(iris))

print('feature names: ')
print(iris.feature_names)

print('data content: ')
print(iris.data[0: 10])
# 4 columns for features

print('target names (categories of iris): ')
print(iris.target_names)

print('categories for each sample:')
print(iris.target)

print('description of the dataset: ')
print(iris.DESCR)

# 63  Splitting data into training and test datasets
# https://pythonhealthcare.org/2018/04/14/63-machine-learning-splitting-data-into-training-and-test-sets/
from sklearn.model_selection import train_test_split

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
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
from sklearn.preprocessing import StandardScaler


# function for readibility
def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format((x)) })


set_numpy_decimal_places(3, 6)

# initialise a new scaling object
sc = StandardScaler()
# set up the scaler just on the training set
sc.fit(x_train)
# apply the scaler to the training and test sets
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

print('Original training set data:')
print('Mean: ', x_train.mean(axis=0))
print('StDev:', x_train.std(axis=0))

print('\nScaled training set data:')
print('Mean: ', x_train_std.mean(axis=0))
print('StDev:', x_train_std.std(axis=0))

print('\nOriginal test set data:')
print('Mean: ', x_test.mean(axis=0))
print('StDev:', x_test.std(axis=0))

print('\nScaled test set data:')
print('Mean: ', x_test_std.mean(axis=0))
print('StDev:', x_test_std.std(axis=0))


