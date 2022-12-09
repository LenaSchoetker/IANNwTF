#%load_ext tensorboard
import tensorflow as tf
import keras_cv
import numpy as np
# from keras.layers import Dense

class MyModel(tf.keras.Model):

    def __init__(self, optimizer, depth, filter_size, L2_reg=0, dropout_rate=0, batch_norm = False, initializer=False):
        """ 
        subclass of the tf.keras.Model class, creates metrics

        Args:
            optimizer (tf.keras.optimzers): set optimizer
            depth(int): numer of layer blocks. Consiting of two Conv2D layers and one pooling layer
            filter_sizer(int): sets filter size. This will rise exponential to the depth
            L2_reg(float) = regularizer that applies a L2 regularization penalty
            dropout_rate(float) = sets input units to 0 with a frequency dropout_rate
            batch_norm(bool): using tf.keras.layers.BatchNormalization() or None
            intializer(bool): using tf.keras.initializers.GlorotNormal() or None

        """  
        super(MyModel, self).__init__()

        kernel_regularizer=tf.keras.regularizers.L2(L2_reg) if L2_reg else None
        initializer = tf.keras.initializers.GlorotNormal() if initializer else None

        self.dropout_rate = dropout_rate
        
        if self.dropout_rate:
            self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        
        self.optimizer = optimizer
        
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="train_loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="train_acc"),
                        tf.keras.metrics.Mean(name="test_loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="test_acc"),
                        tf.keras.metrics.Mean(name="train_frobenius_norm"),
                        tf.keras.metrics.Mean(name="test_frobenius_norm")
                       ]
        
        self.loss_function = tf.keras.losses.CategoricalCrossentropy()   

        self.layer_list = []

        for num in range(depth):
            
            layer_part_1 = self.convlayer1 = tf.keras.layers.Conv2D(filters=filter_size * (2**num), kernel_size=3, padding='same',
                                                                    activation='relu', kernel_regularizer=kernel_regularizer, kernel_initializer=initializer)
            layer_part_2 = self.convlayer2 = tf.keras.layers.Conv2D(filters=filter_size * (2**num), kernel_size=3, padding='same', 
                                                                    activation='relu', kernel_regularizer=kernel_regularizer, kernel_initializer=initializer)
            self.layer_list.append(layer_part_1)

            if batch_norm:
              self.layer_list.append(tf.keras.layers.BatchNormalization())
            self.layer_list.append(layer_part_2)

            if batch_norm:
              self.layer_list.append(tf.keras.layers.BatchNormalization())

            if num != depth - 1:
              pooling_layer = self.pooling = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
              self.layer_list.append(pooling_layer)

        self.global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.out = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x,training=False):
        """
        activates the net and feeds information forward through layers
        also calculates loss and adjusts weights

        Args:
            x(tf.tensor): data for NN, input images with corresponding targets
            training (boolean) : indicating whether the layer should behave in training mode (adding dropout) or in inference mode (doing nothing)      

        Returns: output from NN                           
        """
        for layer in self.layer_list:
          x = layer(x)
          if self.dropout_rate:
                x = self.dropout_layer(x, training)

        x = self.global_pool(x)
        x = self.out(x)

        return x
    
    def reset_metrics(self):
        """
        return a list with all metrics in the model
        """
        
        for metric in self.metrics:
            metric.reset_states()

    def compute_frobenius(self):
        frobenius_norm = tf.zeros((1,))
        for var in self.trainable_variables:
            frobenius_norm += tf.norm(var, ord="euclidean")
        return frobenius_norm
            
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
        
        # update accuracy
        self.metrics[1].update_state(targets,predictions)

        # update frobenius norm
        self.metrics[4].update_state(self.compute_frobenius())

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
        
        # update loss
        self.metrics[2].update_state(loss)

        # upadte accuracy metrics:
        self.metrics[3].update_state(targets,predictions)

        # update frobenius norm
        self.metrics[5].update_state(self.compute_frobenius())

        return {m.name: m.result() for m in self.metrics}