import tensorflow as tf


def preprocess(dataset):
    """
    preprocesses the data to to perfectly fit the input of our neural network

    Args:
        dataset (_type_): dataset 
    """
    # using map + lamda function to change every value but don't change order
    # convert to float so it is readable for nn
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float64), target))
     #print("Shape after converting to float:", tf.shape(dataset))
    # flatten images -> using -1 to flatten into 1-D 
    dataset = dataset.map(lambda img, target: (tf.reshape(img, (-1,)), target))
     #print("Shape after flattening with -1:", tf.shape(dataset))
    # normalize values -> bringing them between -1 and 1 (why not 0 and 1?), maybe typecast with float()
    dataset = dataset.map(lambda img, target: ((img/128.) - 1., target))
     #print("Shape after normalizing values:", tf.shape(dataset))
    # one hot encode targets using tf.one_hot with depth of 10 (numbers 0 to 9)
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
     #print("Shape after one hot encoding:", tf.shape(dataset))
    
    dataset = dataset.cache()
    #shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(20)
    
    #return preprocessed dataset

    return dataset