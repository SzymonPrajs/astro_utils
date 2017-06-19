"""
Set of routines and classes for interpolating light curves.
There include Gaussian Processes and Spline interpolation
"""
import george
from george.kernels import Matern32Kernel


class GP:
    def __init__(self):
        self.gp = None

    def fit(self, data, kernel_size=1000):
        self.gp = george.GP(Matern32Kernel(kernel_size))
        self.gp.compute(data['mjd'], data['flux_err'])