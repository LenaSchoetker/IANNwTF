import tensorflow as tf
import datetime
import tqdm
from model import *

# Define where to save the log
config_name= "config_name"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

train_log_path = f"logs/{config_name}/{current_time}/train"
val_log_path = f"logs/{config_name}/{current_time}/val"


def training_loop(subtask, optimizer, train_ds, test_ds,name):
  """
    trains a neural net for 10 epochs

    Args:
      subtask(string): declaring which subtask is to be trained (subtract/bigger_5)
      optimizer(keras.optimizer): declaring the to be used optimizer
      train_ds(tensor): preprocessed training data
      train_ds(tensor): preprocessed testing data
  """

  # log writer for training metrics
  train_summary_writer = tf.summary.create_file_writer(train_log_path+" "+name)

  # log writer for testing metrics
  val_summary_writer = tf.summary.create_file_writer(val_log_path+" "+name)

  if subtask == "subtract":
    model  = MyModel(tf.keras.losses.MeanSquaredError(), optimizer,subtask)

  elif subtask == "bigger_5":
    model = MyModel(tf.keras.losses.BinaryCrossentropy(from_logits = True), optimizer,subtask)

  for epoch in range(10):
    print(f"Epoch {epoch + 1}:")
    
    # Training:
    
    for data in tqdm.tqdm(train_ds, position=0, leave=True):
      metrics = model.train_step(data)
      
      # logging the validation metrics to the log file which is used by tensorboard
      with train_summary_writer.as_default():
        # for metric in model.metrics:
        tf.summary.scalar(f"{model.metrics[0].name}", model.metrics[0].result(), step=epoch)

    # print the metrics
    print(f"{model.metrics[0].name}: {model.metrics[0].result()}")
    # reset all metrics (requires a reset_metrics method in the model)
    model.reset_metrics()    
    
    # Testing:
    for data in test_ds:
      metrics = model.test_step(data)
  
      # logging the validation metrics to the log file which is used by tensorboard
      with val_summary_writer.as_default():
          #for metric in model.metrics:
          tf.summary.scalar(f"{model.metrics[1].name}", model.metrics[1].result(), step=epoch)
                
    print(f"{model.metrics[1].name}: {model.metrics[1].result()}")
    # reset all metrics
    model.reset_metrics()
    print("\n")