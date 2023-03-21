import tensorflow as tf
import numpy as np

# 2.4 Training the network


def train_step(model, input, target, loss_function, optimizer):
  """
  

  Args:
      model: already initiated model
      input: input data without target
      target: target for input data
      loss_function: chosen loss function
      optimizer: chosen optimizer

  Returns:
      loss and a sample of the accuracy
  """
  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model.activate(input)
    loss = loss_function(target, prediction)
    sample_train_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_train_accuracy = np.mean(sample_train_accuracy)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  return loss, sample_train_accuracy

def test(model, test_data, loss_function):
  # test over complete test data

  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model.activate(input)
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy

# We train for num_epochs epochs
def training_loop(num_epochs, model, train_dataset, test_dataset, cross_entropy_loss, optimizer, train_losses, test_losses , test_accuracies, train_accuracy ):
  """
    trains a neural net with with further specified arguments

    Args:
        num_epochs (int): number of epochs the net will be trained
        net (slass): net that will be trained
        training_data (_type_): training dataset
        testing_data (_type_): testing dataset
        loss_function (_type_): loss function chosen
        optimizer (_type_): optimizer chosen
        vis_train_loss (array): array to track and save loss of training
        vis_train_accuracy (array): array to track and save accuracy of training
        vis_test_loss (array): array to track and save loss of testing
        vis_test_accuracy (array): array to track and save accuracy testing

    Returns:
  """
  for epoch in range(num_epochs):

      # training (and checking in with training)
      epoch_loss_agg = []
      test_accuracy_aggregator = []
      for input,target in train_dataset:
          train_loss, sample_test_accuracy  = train_step(model, input, target, cross_entropy_loss, optimizer)
          test_accuracy_aggregator.append(sample_test_accuracy)
          epoch_loss_agg.append(train_loss)
      
      # track training loss and accuracy

      train_losses.append(tf.reduce_mean(epoch_loss_agg))
      train_accuracy.append(tf.reduce_mean(test_accuracy_aggregator))

      # testing, so we can track accuracy and test loss
      test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
      test_losses.append(test_loss)
      test_accuracies.append(test_accuracy)
      print(f'Epoch: {str(epoch)} starting had accuracy {test_accuracies[-1]}')

  return train_losses, test_losses, test_accuracies, train_accuracy 


