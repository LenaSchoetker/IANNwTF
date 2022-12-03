import tensorflow as tf
import datetime
import tqdm
from model import *

def training_loop(optimizer,depth, filter_size,train_ds, val_ds,epochs,config_name):
  """
  trains a neural net for 15 epochs

  Args:
    optimizer(tf.keras.optimizer): declaring the to be used optimizer
    depth(int): number of layer blocks. Consiting of two Conv2D layers and one pooling layer
    filter_size(int): 
    train_ds(tf.tensor): preprocessed training data
    train_ds(tf.tensor): preprocessed testing data
    config_name(string) = name of configuration.

  """

  # config_name= "config_name"
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  train_log_path = f"logs/{config_name}/{current_time}/train"
  val_log_path = f"logs/{config_name}/{current_time}/val"

  # log writer for training metrics
  train_summary_writer = tf.summary.create_file_writer(train_log_path)

  # log writer for testing metrics
  val_summary_writer = tf.summary.create_file_writer(val_log_path)

  model  = MyModel(optimizer,depth,filter_size)
  # model.summary()

  for epoch in range(epochs):
    print(f"Epoch {epoch + 1}:")
    
        # Training:
        
    for data in tqdm.tqdm(train_ds, position=0, leave=True):
        metrics = model.train_step(data)
        
        # logging the validation metrics to the log file which is used by tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar(f"{model.metrics[0].name}", model.metrics[0].result(), step=epoch)
            tf.summary.scalar(f"{model.metrics[1].name}", model.metrics[1].result(), step=epoch)

    # print the metrics
    print(f"{model.metrics[0].name}: {model.metrics[0].result()}")
    print(f"{model.metrics[1].name}: {model.metrics[1].result()}")

    # reset all metrics (requires a reset_metrics method in the model)
    model.reset_metrics()    
    
    # Validation:
    for data in val_ds:
        metrics = model.test_step(data)
    
        # logging the validation metrics to the log file which is used by tensorboard
        with val_summary_writer.as_default():
            tf.summary.scalar(f"{model.metrics[2].name}", model.metrics[2].result(), step=epoch)
            tf.summary.scalar(f"{model.metrics[3].name}", model.metrics[3].result(), step=epoch)
                
    print(f"{model.metrics[2].name}: {model.metrics[2].result() }")
    print(f"{model.metrics[3].name}: {model.metrics[3].result()}")

    # reset all metrics
    model.reset_metrics()
    print("\n")