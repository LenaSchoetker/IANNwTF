import tensorflow as tf
import datetime
import tqdm
from model import *

def training_loop(optimizer,depth, filter_size,train_ds, val_ds,epochs,config_name, L2_reg=0, dropout_rate=0, batch_norm = False,initializer=None):
  """
  trains a neural net for 15 epochs

  Args:
    optimizer(tf.keras.optimizer): declaring the to be used optimizer
    depth(int): number of layer blocks. Consiting of two Conv2D layers and one pooling layer
    filter_size(int): 
    train_ds(tf.tensor): preprocessed training data
    val_ds(tf.tensor): preprocessed testing data
    config_name(string) = name of configuration
    L2_reg(float) = regularizer that applies a L2 regularization penalty
    dropout_rate(float) = sets input units to 0 with a frequency dropout_rate
    batch_norm(bool): using tf.keras.layers.BatchNormalization() or None
    intializer(bool): using tf.keras.initializers.GlorotNormal() or None

  Returns:
    train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob 

  """

  # config_name= "config_name"
  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  train_log_path = f"logs/{config_name}/{current_time}/train"
  val_log_path = f"logs/{config_name}/{current_time}/val"

  # log writer for training metrics
  train_summary_writer = tf.summary.create_file_writer(train_log_path)

  # log writer for testing metrics
  val_summary_writer = tf.summary.create_file_writer(val_log_path)

  model  = MyModel(optimizer,depth,filter_size, L2_reg=L2_reg, dropout_rate=dropout_rate, batch_norm =batch_norm)

  #Seaborn
  #visualization = np.zeros((epochs,6))
  test_loss =[]
  test_accuracy=[]
  test_frob = []
  train_loss=[]
  train_accuracy=[]
  train_frob = []

  for epoch in range(epochs):
    print(f"Epoch {epoch + 1}:")
    
        # Training:
        
    for data in tqdm.tqdm(train_ds, position=0, leave=True):
        metrics = model.train_step(data)
        # logging the validation metrics to the log file which is used by tensorboard
        with train_summary_writer.as_default():
            tf.summary.scalar(f"{model.metrics[0].name}", model.metrics[0].result(), step=epoch)
            tf.summary.scalar(f"{model.metrics[1].name}", model.metrics[1].result(), step=epoch)
            tf.summary.scalar(f"{model.metrics[4].name}", model.metrics[4].result(), step=epoch)

    #######################################################################################################################################################
    train_loss.append(model.metrics[0].result().numpy())
    train_accuracy.append(model.metrics[1].result().numpy())
    train_frob.append(model.metrics[4].result().numpy())
    ##################################################################################################################################################
    # print the metrics
    print(f"{model.metrics[0].name}: {model.metrics[0].result()}")
    print(f"{model.metrics[1].name}: {model.metrics[1].result()}")
    print(f"{model.metrics[4].name}: {model.metrics[4].result()}")

    # reset all metrics (requires a reset_metrics method in the model)
    model.reset_metrics()    
    
    # Validation:
    for data in val_ds:
        metrics = model.test_step(data)
    
        # logging the validation metrics to the log file which is used by tensorboard        
        with val_summary_writer.as_default():
            tf.summary.scalar(f"{model.metrics[2].name}", model.metrics[2].result(), step=epoch)
            tf.summary.scalar(f"{model.metrics[3].name}", model.metrics[3].result(), step=epoch)
            tf.summary.scalar(f"{model.metrics[5].name}", model.metrics[5].result(), step=epoch)
    #######################################################################################################################################################
    test_loss.append(model.metrics[2].result().numpy())
    test_accuracy.append(model.metrics[3].result().numpy())
    test_frob.append(model.metrics[5].result().numpy()) 
    ################################################################################################################################################                
    
    print(f"{model.metrics[2].name}: {model.metrics[2].result()}")
    print(f"{model.metrics[3].name}: {model.metrics[3].result()}")
    print(f"{model.metrics[5].name}: {model.metrics[5].result()}")

    # reset all metrics
    model.reset_metrics()
    print("\n")
  return train_loss,test_loss,test_accuracy,train_accuracy,train_frob,test_frob