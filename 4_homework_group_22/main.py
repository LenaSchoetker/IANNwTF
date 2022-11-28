# main
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from prepocess import *
from training import*

# get mnist from tensorflow_datasets
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

print("Which subtask do you want? \n A. a+b > 5 B. a-b ")
answer = str(input())

if answer == "A":
  subtask = "bigger_5"
elif answer == "B":
  subtask = "subtract"
else:
  print("Answer is not A or B ")

optimizers = [(tf.keras.optimizers.Adam(),"with Adam"),(tf.keras.optimizers.SGD(),"with SGD"), 
              (tf.keras.optimizers.experimental.SGD(momentum = 0.5),"with SGD+Momentum"),
(tf.keras.optimizers.experimental.Adagrad(),"with Adagard"),(tf.keras.optimizers.experimental.RMSprop(),"with RMSprop")]

# preprocess the data using the map method
train_dataset = prepare_mnist_data(train_ds, 32, subtask)
test_dataset = prepare_mnist_data(test_ds, 32, subtask)

#%tensorboard --logdir logs/
for optimizer,name in optimizers:
  training_loop(subtask, optimizer, train_dataset , test_dataset,name)