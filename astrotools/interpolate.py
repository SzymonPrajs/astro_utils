"""
Set of routines and classes for interpolating light curves.
There include Gaussian Processes and Spline interpolation
"""
import numpy as np
import scipy.optimize

import george
from george.kernels import Matern32Kernel


class GP:
    """
    Gaussian process light curves in the common DataFrame
    data format. Must be single band light curves containing:
    mjd, flux, flux_err
    """
    def __init__(self):
        self.gp = None
        self.data = None

    def fit(self, data, kernel_size=1000):
        """
        Fit the data with a gaussian process using the
        specified kernel size.

        Parameters
        ----------
        data : `pandas.DataFrame`
            Standard formatted DataFrame containing:
            mjd, flux, flux_err

        kernel_size : float, optional
            Kernel size passed to the `george.kernel` type

        Returns
        -------
        None
        """
        self.data = data

        self.gp = george.GP(Matern32Kernel(kernel_size))
        self.gp.compute(self.data['mjd'], yerr=self.data['flux_err'])

    def fit_reduce(self, data, init_kernel_size=None):
        """
        Fit the data with a gaussian process and optimise 
        the kernel size to fit with the highest likelihood. 
        
        Parameters
        ----------
        data : `pandas.DataFrame`
            DataFrame object in the standard format containing:
            mjd, flux, flux_err
            
        init_kernel_size : float, optional
            Starting value for the kernel to be fitted.

        Returns
        -------
        None
        """
        self.data = data

        self.gp = george.GP(Matern32Kernel(init_kernel_size))
        self.gp.compute(self.data['mjd'], yerr=self.data['flux_err'])

        p0 = self.gp.kernel.vector
        scipy.optimize.minimize(self._log_likelihood, p0, jac=self._grad_log_likelihood)

    def predict(self, x_new):
        """
        Interpolate for new values of `x_new` using the previously
        fit Gaussian Process model.

        Parameters
        ----------
        x_new : array-like
            Array of new values at which the Gaussian Process model
            is to be evaluated.

        Returns
        -------
        mu : `np.ndarray`
            Mean value of the model

        std : `np.ndarray`
            One sigma uncertainly values
        """
        if self.data is None or self.gp is None:
            raise RuntimeError("""Values cannot be predicted before kernel is fit.
                                Perform `fit` or `fit_reduce` first""")

        mu, cov = self.gp.predict(self.data['flux'], x_new)
        std = np.sqrt(np.diag(cov))

        return mu, std

    def _log_likelihood(self, p):
        self.gp.kernel[:] = p

        return -self.gp.lnlikelihood(self.data['flux'], quiet=True)

    def _grad_log_likelihood(self, p):
        self.gp.kernel[:] = p

        return -self.gp.grad_lnlikelihood(self.data['flux'], quiet=True)
