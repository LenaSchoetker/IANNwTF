from model import *
from training import *
from visualization import *
from preprocess import *
import tensorflow_datasets as tfds
import keras_cv
import matplotlib.pyplot as plt

# original
train_ds, val_ds = get_cifar10(32)
train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob = training_loop(tf.keras.optimizers.Adam(),2,12,train_ds,val_ds, 15,"original")
visualize(train_loss,test_loss,test_accuracy,train_accuracy,"original")
visualize_frob(train_frob,test_frob,"original")

#1. Data augmentation
#add slightly transformed copies of already included data
augmentation_model = tf.keras.Sequential([keras_cv.layers.RandAugment(value_range=[0,1],magnitude=0.1)])
train_ds, val_ds = get_cifar10(32,augmentation_model = augmentation_model)
train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob = training_loop(tf.keras.optimizers.Adam(),2,12,train_ds,val_ds, 15,"Data_aug")
visualize(train_loss,test_loss,test_accuracy,train_accuracy,"Data augmentation")
visualize_frob(train_frob,test_frob,"Data augmentation")

#2. Penalties L2
#add penalties to minimizing parameter magnitude
train_ds, val_ds = get_cifar10(32)
train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob = training_loop(tf.keras.optimizers.Adam(),2,12,train_ds,val_ds, 15,"L2",L2_reg=0.01)
visualize(train_loss,test_loss,test_accuracy,train_accuracy,"L2")
visualize_frob(train_frob,test_frob,"L2")

# 3. Drop out -> init, call
#randomly drop out units during training
train_ds, val_ds = get_cifar10(32)
train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob = training_loop(tf.keras.optimizers.Adam(),2,12,train_ds,val_ds, 15,"Drop_out",dropout_rate=0.2)
visualize(train_loss,test_loss,test_accuracy,train_accuracy,"Drop out")
visualize_frob(train_frob,test_frob,"Drop out")

# 4. batch normalization
#applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1
train_ds, val_ds = get_cifar10(32)
train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob = training_loop(tf.keras.optimizers.Adam(),2,12,train_ds,val_ds, 15,"batch_Norm",batch_norm = True)
visualize(train_loss,test_loss,test_accuracy,train_accuracy,"Batch Normalization")
visualize_frob(train_frob,test_frob,"Batch Normalization")

#5. weights initializer
# set the initial random weights of layers
train_ds, val_ds = get_cifar10(32)
train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob = training_loop(tf.keras.optimizers.Adam(),2,12,train_ds,val_ds, 15,"weights",initializer = True)
visualize(train_loss,test_loss,test_accuracy,train_accuracy,"weight initializer")
visualize_frob(train_frob,test_frob,"Weight initializer")