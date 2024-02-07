import numpy as np

def Wrapper(input_array, d):
    '''Applied periodic boundary conditions to the input_array by wrapping it around the box of size d'''
    return input_array - np.round(input_array/d)*d + 0.5*d


