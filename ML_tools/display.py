import matplotlib.pyplot as plt


def chart_learning_rate(x, training_accuracy, test_accuracy):
    """
    Author: Mike Allen
    :param: x, training_accuracy, test_accuracy
    :return: figure
    """
    # x = results['n']
    y1 = training_accuracy  # results['training_accuracy']
    y2 = test_accuracy  # results['test_accuracy']

    # Create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.plot(x, y1, color='k', linestyle='solid', label='Training set')
    ax.plot(x, y2, color='b', linestyle='dashed', label='Test set')
    ax.set_xlabel('training set size (cases)')
    ax.set_ylabel('Accuracy')
    plt.title('Effect of training set size on model accuracy')
    plt.legend()
    plt.show()


def print_dict_value_decimal(dictionary, digits=3):
    """Iterate through, and print the dictionary
    Input: dictionary with value = number
    Print the values with a fixed number of decimal digits
    Author: Mike Allen, modified by EV
    """
    # ToDo turn number of digits into an option
    print('-------------------------------------------------')
    for key, value in dictionary.items():
        print(key, '= %0.3f' % value)  # print 3 decimal places
    return
