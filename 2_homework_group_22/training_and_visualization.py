"""
Multi-Layer Perceptron
The universal approximation theorem states that a feed-forward artificial neural
network can learn to approximate any continuous function with just one hidden
layer, given enough units. This notebook contains this multi-layer
perceptron (’MLP’) and it will learn a continious function
"""

# dependencies
import numpy as np
import matplotlib.pyplot as plt



# 01: Building your data set
# building the data set using NumPy
# generate 100 random numbers between 0 and 1
x = np.random.rand(100)

# create targets using x values and applying the continious function on them
t = np.zeros(100)
t[:] = x[:]**3-x[:]**2 + 1

# 02: Perceptrons
# implement a simple layer using perceptrons
from Layer_Class import Layer

# 03: Multi-Layer Perceptron
# Create a MLP class which combines instances of your Layer class into into a MLP.
from MLP_Class import MLP

# 04: Training
# To compute the network’s loss, you should use the mean squared error
# where y is the output of your network and t is the intended target

# create loss array
loss = np.zeros(1000)

# init and create MLP
mlp = MLP()
mlp.create([1,10,1])

# plot arrays
plot_avg_squared_error_loss = []
plot_output = []

# training
for epoch in range(1000):
    avg_squared_error_loss = 0.0

    for i, datapoint in enumerate(x):

        output = mlp.forward_step(datapoint)
        mlp.backpropagation(t[i])
        avg_squared_error_loss += 0.5*(output - t[i])**2      #MSE

        if epoch == 1000-1:                                                                        
            plot_output.append(output)                        #store output after training is completed 
    if epoch ==0:
        print("The average loss before training:", avg_squared_error_loss/100)     

    plot_avg_squared_error_loss.append(avg_squared_error_loss)

print("The average loss after training:", avg_squared_error_loss/100)             


# 05: Visualization
# create the figure and the axes for the plot
fig, axes = plt.subplots(nrows = 1, ncols = 2)

# plot for 1000 epochs & average loss per epoch
axes[0].set(
    title="Average Loss per epoch",
    xlabel="Epoch",
    ylabel="Average Loss",)
axes[0].plot(range(1000), plot_avg_squared_error_loss,"-.")


# plot for comparison of original data points and MLPs approximation
axes[1].set(
    title="Original Data vs. MLP",
    ylabel="Output",  
    xlabel="Data points",)
axes[1].plot(x,t, ".", label = "target")
axes[1].plot(x,plot_output, ".", label = "MLP approximation")
axes[1].plot(x,x, ".", label = "input data")
plt.legend()

plt.show()
