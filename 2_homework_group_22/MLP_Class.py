# dependencies
import numpy as np
import matplotlib.pyplot as plt
from Layer_Class import Layer

class MLP():

    def __init__(self):
        """
        Creates a MLP, starts at input layer, ends at output layer with overall
        shape of 1,10,1
        Stores each Layer in array layers
        """
        #number of layers in the MLP
        self.depth = 0
        # array of layer objects
        self.layers = []

    def create(self, perceptrons_per_layer):
        """
        Method to create the Layers within the MLP and store them in the array layers

        Args:
            perceptrons_per_layer (array): number of perceptrons per layer
        """
        self.depth = len(perceptrons_per_layer)

        # appending the layers from front to back
        for i in range(1, self.depth):
            self.layers.append(Layer(perceptrons_per_layer[i],perceptrons_per_layer[i-1])) 
        

    def forward_step(self, input):
        """_summary_
 
        Args:
            input (_type_): input to the first layer (input layer)                                                               

        Returns:
            prev_act: Array of the previous layer's activation
        """
        # stores previous activations
        prev_act = input

        # make a forward step (calculate the activation) of every layer using the previous layer's output
        for layer in self.layers:
            prev_act = layer.forward_step(prev_act)  

        #returns activation of the last layer (MLP's Output)
        return np.reshape(prev_act, newshape= -2)


    def backpropagation(self, target):
        """
        computes the backpropagation for each layer
        Args:
            target (_type_): for loss computation and adjustment of weights                                       # <- ???
        """

        #iterate the layers of the MLP from back to front 
        for n,layer in enumerate(reversed(self.layers)):                                                        
            # index output = 0  
            p = layer.return_p()
            g = layer.return_g()
            w = layer.return_w()
            layer.backward_step(n,target ,p ,g, w)
