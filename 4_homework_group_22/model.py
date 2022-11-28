# model
import tensorflow as tf
from keras.layers import Dense


# %load_ext tensorboard -> just for notebook
class MyModel(tf.keras.Model):

    # 1. constructor
    def __init__(self, loss_function, optimizer, subtask):
        super().__init__()
        """
        subclass of the tf.keras.Model class
        creates metrics 
        2 layers with 32 units each
        1 output layer with 1 unit and sigmoid activation function
        """

        # optimizer, loss function and metrics
        if subtask == "bigger_5":
          self.metrics_list = [
                        tf.keras.metrics.Mean(name="bigger_5_train_loss"),
                        tf.keras.metrics.Mean(name="bigger_5_test_loss")
                       ]
        elif subtask == "subtract":
          self.metrics_list = [
                        tf.keras.metrics.Mean(name="subtract_train_loss"),
                        tf.keras.metrics.Mean(name="subtract_test_loss")
                       ]
        
        self.optimizer = optimizer  
        
        self.loss_function = loss_function 
        
        # layers to be used
        self.dense1 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)

        if subtask == "bigger_5":
          self.out= tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
        elif subtask == "subtract":
          self.out= tf.keras.layers.Dense(1)
        
    @tf.function  
    # 2. call method (forward computation)
    def call(self, images, training=True):
        """
        activates the net and feeds information forward through layers
        also calculates loss and adjusts weights

        Args:
            images(tensor): data for nn, input images with corresponding targets      

        Returns: output from nn                           
        """
        img1, img2 = images
        img1_x = self.dense1(img1) 
        img2_x = self.dense2(img2)
        combined_x = tf.concat([img1_x, img2_x ], axis=1)
        
        return self.out(combined_x)

    # 3. metrics property
    @property
    def metrics(self):
        """
        return a list with all metrics in the model
        """
        return self.metrics_list

    # 4. reset all metrics objects
    def reset_metrics(self):
        """
        reset metrices
        """
        for metric in self.metrics:
            metric.reset_states()

    @tf.function
    def train_step(self, data):
        """
        training the network for once

        Args:
            data: input data (2 images with target)

        Returns:
            Return a dictionary mapping metric names to current value
        """
        img1, img2, target = data

        with tf.GradientTape() as tape:
            output = self.call((img1, img2), training=True)

            target = tf.reshape(target,output.shape)

            loss = self.loss_function(target, output) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        """
        testing the network for once

        Args:
            data: input data (2 images with target)

        Returns:
            Return a dictionary mapping metric names to current value
        """
        img1, img2, target = data
        prediction = self.call((img1, img2), training=True)

        target = tf.reshape(target,prediction.shape)

        loss = self.loss_function(target, prediction) + tf.reduce_sum(self.losses)

        self.metrics[1].update_state(loss)

        return {m.name: m.result() for m in self.metrics}