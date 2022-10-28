import numpy as np

array = np.random.normal(0 , 1, size=(5,5))
print("First:{}\n".format(array))


mask = array <= 0.09
array = np.where(array > 0.09, array**2, array)
array[mask] = 42
print("Then:{}\n".format(array))

print("Fourth column: {}".format(array[:,3]))