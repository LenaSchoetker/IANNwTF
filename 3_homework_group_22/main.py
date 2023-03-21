"""
Assignment 3
Group 22
MNIST classification with only pre-built layers and functions
"""

# dependencies
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.optimizers import SGD

from preprocess import preprocess
from training import training_loop
from classifier import MNIST_classifier
from visualization import visualization


# 2.1 Loading the MNIST dataset and split it into training and test data
(train_ds, test_ds), ds_info = tfds.load("mnist", split=['train', 'test'], as_supervised = True , with_info = True)

print(ds_info)
#tfds.show_examples(train_ds , ds_info)

# 2.2 Setting up the data pipeline

learning_rate = 0.1
optimizer = SGD(learning_rate, momentum=0)
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
net = MNIST_classifier()

loss_train = []
loss_test = []
train_accuracy = []
test_accuracy = []

train_dataset = train_ds.apply(preprocess)
test_dataset = test_ds.apply(preprocess)

train_loss, test_loss, test_accuracy, train_accuracy = training_loop(10, net, train_dataset , test_dataset, cross_entropy_loss, optimizer, loss_train, loss_test, test_accuracy, train_accuracy)

visualization(train_loss, train_accuracy, test_loss, test_accuracy)
