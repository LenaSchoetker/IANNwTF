# dependencies
import numpy as np
import matplotlib.pyplot as plt

class Layer():  

    def __init__(self, n_units, input_units):
        """
        simple class to init a layer using perceptrons

        Args:
            n_units (int): number of units in the layer
            input_units (int): number of units in preceding layer
        """

        # store number of input units and random weights
        self.input_units = input_units
        self.weights = np.random.random((input_units, n_units))

        # bias is initialized as np.zeroes
        self.bias = np.zeros((1, n_units))
        # empty attributes with correct dims
        self.layer_input = np.empty((1, input_units))                 #input of a layer               
        self.layer_preactivation = np.empty((1, n_units))             #preactivation of a layer      
        self.layer_activation = np.empty((1, n_units))                #activation of a layer        
        self.error_signal = np.empty((1, n_units))                    #∂L/∂activation of a layer                
        


    def forward_step(self, layer_inputs):
        """
        feeds input forward, checks for activation

        Args:
            layer_inputs (array): feeds input into layer                                                        

        Returns:
            self.layer_activation: Array of the layer's activation
        """
        self.layer_input = np.full((1, self.input_units), layer_inputs)              
        self.layer_preactivation = self.layer_input @ self.weights  + self.bias     
        self.layer_activation = np.maximum(0,self.layer_preactivation)              #activation = ReLu(preactivation)

        return self.layer_activation

    def return_preactivation(self):
        """
        Get the preactivation of the layer  

        Returns:
            self.layer_preactivation: Array of the layer's preactivation                                                        
        """
        return self.layer_preactivation

    def return_error(self):
        """                                                      
        Get the error signal of the layer  

        Returns:
            self.error_signal: Array of the layer's error_signal
        """
        return self.error_signal

    def return_weights(self):
        """
        Get the weights of the layer  

        Returns:
            self.weights: Array of the layer's weights
        """
        return self.weights

    def backward_step(self, index, t, preactivation, error_signal, weights):         
        """
        method to update each unit's parameter by computing gradients etc.

        Args:
            index (int): index of the current layer                                                                        
            t (int): targets
            preactivation (array): preactivation of the layer l+1
            error_signal (array): error_signal of the Layer l+1
            weights (array): weight matrix of Layer l+1
        """ 

        # check if it is the last layer
        if  index == 0:
        #∂L/∂activation = derivative of loss function
            l_a = self.layer_activation - t
        else:
            # ∂L/∂activation l= (ReLu(preactivation l+1)')                  ∗ ∂L/∂activation l+1) @  weights l+1^T)
            l_a = (np.maximum(0,preactivation).astype(bool).astype(int) * (error_signal)) @ np.transpose(weights)
        
        #store ∂L/∂activation of layer
        self.error_signal = l_a

        # ∂L/∂W l       = input l^T                       @          ReLu(preactivation)'                                     ° ∂L/∂activation
        gradient_weights = np.transpose(self.layer_input) @ (np.maximum(0, self.layer_preactivation).astype(bool).astype(int) * (l_a)) 

        #∂L/∂bl       = ReLu(preactivation)'                                            ° ∂L/∂activation
        gradient_bias = np.maximum(0,self.layer_preactivation).astype(bool).astype(int) * (l_a)
        
        
        # learning rates of weights and biases (shaped for multiplication)                                                                               
        learning_rate_w = np.full(np.shape(self.weights),0.0001)
        learning_rate_b = np.full(np.shape(self.bias),0.0001)

        # update the layer's parameters
        # θnew = θold − η∇θL
        self.weights -= learning_rate_w * gradient_weights
        self.bias -= learning_rate_b * gradient_bias