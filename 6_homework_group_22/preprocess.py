import tensorflow_datasets as tfds
import tensorflow as tf
def get_cifar10(batch_size, augmentation_model = None):
    """
    Load and prepare CIFAR-10 as a tensorflow dataset.
    Returns a train and a validation dataset.

    Args:
    batch_size(int): batch size
    augmentation_model: using augementation

    Returns:
    prefetched training and testing data
    """
    train_ds, val_ds = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True)

    one_hot = lambda y: tf.one_hot(y, 10)

    map_func = lambda x,y: (tf.cast(x, dtype=tf.float32)/255.,
                            tf.cast(one_hot(y),tf.float32))

    map_func_2 = lambda x: (x["image"],x["label"])

    train_ds = train_ds.map(map_func_2).map(map_func).cache()
    val_ds   = val_ds.map(map_func_2).map(map_func).cache()
    
    train_ds = train_ds.shuffle(4096).batch(batch_size)
    val_ds   = val_ds.shuffle(4096).batch(batch_size)
    
    if augmentation_model:
        train_ds = train_ds.map(lambda x,y : (augmentation_model(x), y),num_parallel_calls=tf.data.AUTOTUNE)

    return (train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE))