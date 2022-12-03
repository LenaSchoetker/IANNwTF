# %load_ext tensorboard
import tensorflow as tf
from keras.layers import Dense

class MyModel(tf.keras.Model):

    def __init__(self,optimizer,depth,filter_size):
        """ 
        subclass of the tf.keras.Model class, creates metrics

        Args:
            optimizer (tf.keras.optimzers): set optimizer
            depth(int): numer of layer blocks. Consiting of two Conv2D layers and one pooling layer
            filter_sizer(int): sets filter size. This will rise exponential to the depth

        """  
        super(MyModel, self).__init__()

        self.optimizer = optimizer
        
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="train_loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="train_acc"),
                        tf.keras.metrics.Mean(name="test_loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="test_acc")
                       ]
        
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()   

        self.layer_list = []

        for num in range(depth):
            layer_part_1 = self.convlayer1 = tf.keras.layers.Conv2D(filters=filter_size * (2**num), kernel_size=3, padding='same', activation='relu')
            layer_part_2 = self.convlayer2 = tf.keras.layers.Conv2D(filters=filter_size * (2**num), kernel_size=3, padding='same', activation='relu')
            self.layer_list.append(layer_part_1)
            self.layer_list.append(layer_part_2)

            if num != depth - 1:
              pooling_layer = self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
              self.layer_list.append(pooling_layer)

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        """
        activates the net and feeds information forward through layers
        also calculates loss and adjusts weights

        Args:
            x(tf.tensor): data for nn, input images with corresponding targets      

        Returns: output from nn                           
        """
        for layer in self.layer_list:
          x = layer(x)

        x = self.global_pool(x)
        x = self.out(x)

        return x
    
    def reset_metrics(self):
        """
        return a list with all metrics in the model
        """
        
        for metric in self.metrics:
            metric.reset_states()
            
    @tf.function
    def train_step(self, data):
        """
        training the network for once

        Args:
            data: input data (image with target)

        Returns:
            Return a dictionary mapping metric names to current value
        """
        
        x, targets = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        self.metrics[1].update_state(targets,predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        testing the network for once

        Args:
            data: input data (image with target)

        Returns:
            Return a dictionary mapping metric names to current value
        """

        x, targets = data
        predictions = self(x, training=False)
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)

        self.metrics[2].update_state(loss)
        # for accuracy metrics:
        self.metrics[3].update_state(targets,predictions)

        return {m.name: m.result() for m in self.metrics}