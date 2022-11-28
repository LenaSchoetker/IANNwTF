import tensorflow as tf

def prepare_mnist_data(mnist, batch_size, condition):
  """
    preprocesses the data to to perfectly fit the input of our neural network depending on the subtask

    Args:
        dataset (_type_): dataset 
  """

  # Step 1
  # flatten the images into vectors
  mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
  # convert data from uint8 to float32
  mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
  # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
  mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
  

  # Step 2

  zipped_ds = tf.data.Dataset.zip((mnist.shuffle(2000), mnist.shuffle(2000)))

  # dependent on condition
  if condition == "bigger_5":
    zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], (x1[1]+x2[1]) > 5))
      # transform boolean target to int
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.cast(t, tf.int32)))

  elif condition == "subtract":
    zipped_ds = zipped_ds.map(lambda x1, x2: (x1[0], x2[0], x1[1]-x2[1]))
    # transform boolean target to float
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1,x2, tf.cast(t, tf.float32)))

  zipped_ds = zipped_ds.cache()
  
  # Step 3
      # batch the dataset
  zipped_ds = zipped_ds.batch(batch_size)
      # prefetch
  zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)

  return zipped_ds