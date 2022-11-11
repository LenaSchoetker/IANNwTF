# dependencies
import numpy as np
import matplotlib.pyplot as plt
from Layer_Class import Layer

class MLP():

    def __init__(self):
        """
        Creates a MLP, starts at firs hidden layer, ends at output layer
        Stores each Layer object in list 
        """
        #number of layers in the MLP (exept the input layer)
        self.depth = 0
        #list of layer objects
        self.layers = []

    def create(self, perceptrons_per_layer):
        """
        Method to create the Layers within the MLP and store them in the list of layers

        Args:
            perceptrons_per_layer (array): number of perceptrons per layer : [1,10,1]
        """
        self.depth = len(perceptrons_per_layer)
        # appending the layers from front to back
        for i in range(1, self.depth):
            self.layers.append(Layer(perceptrons_per_layer[i],perceptrons_per_layer[i-1]))
        

    def forward_step(self, input):
        """_summary_
 
        Args:
            input (array): input of the layer                                                               

        Returns:
            prev_act: Array of the previous layer's activation
        """
        # stores previous activations
        prev_act = input

        # make a forward step (calculate the activation) of every layer using the previous layer's output
        for layer in self.layers:
            prev_act = layer.forward_step(prev_act)  

        #returns reshaped activation of the last layer (MLP's Output)
        return np.reshape(prev_act, newshape= -1)


    def backpropagation(self, target):
        """
        computes the backpropagation for each layer
        Args:
            target (int): for loss computation and adjustment of weights                                    
        """

        #iterate the layers of the MLP from back to front 
        for n,layer in enumerate(reversed(self.layers)):                                                        
            # index output = 0  
            p = layer.return_preactivation()    #preactivations of l+1 as we iterate backwards
            e = layer.return_error()            #error signal of l+1
            w = layer.return_weights()          #weights of l+1
            layer.backward_step(n,target ,p ,e, w)       
