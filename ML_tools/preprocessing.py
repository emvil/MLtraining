from sklearn.model_selection import train_test_split


def split_data(x, y, training_split_rate=0.75, n_points=0, display=0, seed=42):
    """ Split data into training and test sets (default: 75% / 25%).
    Inputs:
    x = dataset
    y = categories
    training_split_rate, between 0 and 1
    n_points = number of points to keep in the train set. If = 0, then keep all the points
    display = option to display the size of split arrays
    """
    # Todo make outputs optional
    y_train = None
    y_test = None
    if y is not None:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-training_split_rate, random_state=seed)
    else:
        x_train, x_test = train_test_split(x, test_size=1-training_split_rate, random_state=seed)
    # random state is integer seed
    # If this is omitted than a different seed will be used each time
    if display:
        print('size of train / test data set: ')
        print('Shape of X:', x.shape)
        print('Shape of X_train:', x_train.shape)
        print('Shape of X_test:', x_test.shape)
        if y is not None:
            print('Shape of y:', y.shape)
            print('Shape of y_train:', y_train.shape)
            print('Shape of y_test:', y_test.shape)
    if n_points > 0:
        x_train = x_train[0:n_points]
        y_train = y_train[0:n_points]
    return x_train, x_test, y_train, y_test

