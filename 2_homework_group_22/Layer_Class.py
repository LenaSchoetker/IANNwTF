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

        # input units and random input weights
        self.input_units = input_units
        self.weights = np.random.random((input_units, n_units))

        # bias is initialized as np.zeroes
        self.bias = np.zeros((1, n_units))

        # empty attributes with correct dims
        self.layer_input = np.empty((1, input_units))                                #array
        self.layer_preactivation = np.empty((1, n_units))                            #array
        self.layer_activation = np.empty((1, n_units))                               #array
        self.gradient_act = np.empty((1, n_units))                                   #array
        

    def forward_step(self, layer_inputs):
        """
        feeds input forward, checks for activation

        Args:
            layer_inputs (_type_): feeds input into layer                                                         # <- TYPE/DIMS

        Returns:
            self.layer_activation: Array of the layer's activation
        """
        self.layer_input = np.full((1, self.input_units),layer_inputs)               #array
        self.layer_preactivation = self.layer_input @ self.weights  + self.bias     #array
        self.layer_activation = np.maximum(0,self.layer_preactivation)              #array

        return self.layer_activation

    def return_p(self):
        """
        Get the preactivation of the layer  

        Returns:
            self.layer_preactivation: Array of the layer's preactivation                                                        
        """
        return self.layer_preactivation

    def return_g(self):
        """                                                      
        Get the activation gradient of the layer  

        Returns:
            self.gradient_act: Array of the layer's activation gradient
        """
        return self.gradient_act

    def return_w(self):
        """
        Get the weights of the layer  

        Returns:
            self.weights: Array of the layer's weights
        """
        return self.weights

    def backward_step(self, index, t, preactivation, gradient_act, weights):         
        """
        method to update each unit's parameter by computing gradients etc.

        Args:
            index (_type_): index of the current layer                                                                        # <- description, Args & DIMS/TYPE
            t (_type_): targets
            preactivation (_type_): preactivation of the layer L+1
            gradient_act (_type_): activation gradient of the Layer L+1
            weights (_type_): weight matrix of Layer L+1
        """ 

        # check if it is the last layer
        if  index == 0:
            l_a = self.layer_activation - t
        else:
            # ∂L/∂inputl= (σ′(preactivation)                            ∗ ∂L/∂activation) *  Wl^T)
            l_a = (np.maximum(0,preactivation).astype(bool).astype(int) * (gradient_act)) @ np.transpose(weights)
        
        
        self.gradient_act = l_a

        # ∂L/∂Wl = inputl^T                               *           (σ′(preactivation)                     ◦      ∂L/∂activation)
        #        = inputl^T                               *          ReLu(preactivation)'                    °    (activation - target)
        gradient_weights = np.transpose(self.layer_input) @ (np.maximum(0, self.layer_preactivation).astype(bool).astype(int) * (l_a)) 

        # ∂L/∂bl      = σ′(preactivation)                                               ◦ ∂L/∂activation
        #             = ReLu(preactivation)'                                            ° (activation - target)
        gradient_bias = np.maximum(0,self.layer_preactivation).astype(bool).astype(int) * (l_a)
        
        
        # learning rates of weights and biases (shaped for multiplication)                                                                               
        learning_rate_w = np.full(np.shape(self.weights),0.0001)
        learning_rate_b = np.full(np.shape(self.bias),0.0001)

        # update the layer's parameters
        # θnew = θold − η∇θL
        self.weights -= learning_rate_w * gradient_weights
        self.bias -= learning_rate_b * gradient_bias