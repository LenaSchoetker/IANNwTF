import tensorflow_datasets as tfds
from model import *
from training import *
# load cifar10
ds, ds_info = tfds.load("cifar10", as_supervised=True, with_info =True)

# split data into train and test
train_ds = ds["train"]
val_ds = ds["test"]

tfds.show_examples ( train_ds , ds_info )

# preprocessing the cifar10 data
train_ds = train_ds.map(lambda x,y: (x/255, tf.one_hot(y, 10, dtype=tf.float32)),\
                        num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(lambda x,y: (x/255, tf.one_hot(y, 10, dtype=tf.float32)),\
                    num_parallel_calls=tf.data.AUTOTUNE).cache().shuffle(5000).batch(32).prefetch(tf.data.AUTOTUNE)


optimizers = [(tf.keras.optimizers.Adam(),"Adam-"),(tf.keras.optimizers.SGD(),"SGD-"), 
              (tf.keras.optimizers.experimental.SGD(momentum = 0.5),"SGD+Mom-"),
              (tf.keras.optimizers.experimental.Adagrad(),"Adagard-"),(tf.keras.optimizers.experimental.RMSprop(),"RMSprop-")]

depths = [(2,"Depth=2-"),(3,"Depth=3-"),(4,"Depth=4-")]
filter_sizes = [(12,"Filter=12"),(24,"Filter=24"),(48,"Filter=48")]


# %tensorboard --logdir logs/
for optimizer,name1 in optimizers:
  for depth,name2 in depths:
    for filter_size ,name3 in filter_sizes:
      training_loop(optimizer,depth,filter_size,train_ds,val_ds, 15, name1+name2+name3)
