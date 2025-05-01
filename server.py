import numpy as np

def func(x):
    return x**3-3*x-2

for item in np.arange(-2,3.1,0.1):
    print(f'{float(item)}   ')