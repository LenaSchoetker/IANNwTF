import matplotlib.pyplot as plt

def visualize(train_loss,test_loss,test_accuracy,train_accuracy,title):
  """
  visualize training loss and accuracy and testing loss and accuracy using matplotlib

  Args:
  train_loss(list) = training loss across epochs
  test_loss(list) =  test loss across epochs
  test_acc(list) = test accuracy across epochs
  train_acc(list) = training accuracy across epochs
  titel(list) = plot title

  """
  #plt.figure()
  line1, = plt.plot(train_loss)
  line2, = plt.plot(test_loss)
  line3, = plt.plot(test_accuracy)
  line4, = plt.plot(train_accuracy)
  plt.title(title)
  plt.xlabel("Training steps")
  plt.ylabel("Loss/Accuracy")
  plt.legend((line1, line2, line3, line4), ("training loss", "test loss", "test accuracy", "train accuracy"))
  plt.show()

def visualize_frob(train_frob,test_frob,title):
  """
  visualize frobenius Norm for training and testing using matplotlib

  Args:
  train_frob(list) = Frobenius norm for training across epochs
  test_frob(list) =  Frobenius norm for testing across epochs
  """
  #plt.figure()
  line1, = plt.plot(train_frob)
  line2, = plt.plot(test_frob)
  plt.title(title)
  plt.xlabel("Training steps")
  plt.ylabel("Frobenius Norm")
  plt.legend((line1, line2), ("train frobenius norm","test frobenius norm"))
  plt.show()