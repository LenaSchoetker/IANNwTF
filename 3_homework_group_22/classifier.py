# 2.3 Building a deep neural network with TensorFlow
# import Dense layer from keras
import tensorflow as tf
from keras.layers import Dense

class MNIST_classifier(tf.keras.Model):

    def __init__(self):
        """
        subclass of the tf.keras.Model class
        purpose is to classify MNIST images
        2 layers with 256 units each
        1 output layer with 10 units for 10 numbers (0-9)
        """
        super(MNIST_classifier, self). __init__()
        # using ReLu for dense layers
        self.dense1 = Dense(256, activation=tf.nn.relu)
        self.dense2 = Dense(256, activation=tf.nn.relu)
        # using Softmax in output because we want a probability distribution to force network to
        # only choose one number with a certain probability 
        self.output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        
    @tf.function
    def activate(self, input_data):
        """
        activates the net and feeds information forward through layers
        also calculates loss and adjusts weights

        Args:
            input_data: data for nn, input images with corresponding targets                                 
        """
        # give layer1 input data
        outputs = self.dense1(input_data)
        # give layer2 outputs of layer1
        outputs = self.dense2(outputs)
        # give output layer outputs of layer2
        outputs = self.output_layer(outputs)

        return outputs
