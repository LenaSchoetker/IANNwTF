import matplotlib.pyplot as plt

def visualization(train_losses , train_accuracies, test_losses ,test_accuracies):
    """ Visualizes accuracy and loss for training and test data using the mean of each epoch.
    Loss is displayed in a regular line , accuracy in a dotted line.
    Training data is displayed in blue , test data in red.

    Parameters
    ----------
    train_losses : numpy.ndarray training losses
    train_accuracies : numpy.ndarray training accuracies
    test_losses : numpy.ndarray test losses
    test_accuracies : numpy.ndarray test accuracies
    """
    plt.figure()
    line1 , = plt.plot(train_losses , "b-")
    line2 , = plt.plot(test_losses , "r-")
    line3 , = plt.plot(train_accuracies , "b:")
    line4 , = plt.plot(test_accuracies , "r:")
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend ((line1 , line2 , line3 , line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
    plt.show()
