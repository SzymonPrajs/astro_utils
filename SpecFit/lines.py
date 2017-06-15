import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit, leastsq
from scipy.interpolate import UnivariateSpline as spl
import os

class Lines(object):
    """Fit lines to spectra and estimate their velocities

    Parameters
    ----------
    spec : string
        Path to the data file containing the spectrum
    line_name : {'Fe', 'C', 'Ti', 'Mg', 'MgTiW', 'CO', 'OII', 'OIIL', 'OIIR', 'OIIW', 'FeII', 'HeI'}
        Name of a group of lines

    Attributes
    ----------
    lines : numpy.ndarray
        Array of wavelengths selected to be used by the fitters
    """

    def __init__(self, spec = None, z = 0.0, line_name = "OIIW"):
        if spec == None:
            raise ValueError('No spectrum path was provided')
        else:
            self.spec = spec
        self.line_name = line_name

        self._data = np.loadtxt(self.spec, unpack=True)
        self._z = None
        self.z = z

        self.bg = None
        self.indivisual_profiles = None
        self.composite_profile = None


    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, spec):
        if os.path.exists(spec):
            self._spec = spec
        else:
            raise ValueError('Invalid spectrum path: ' + path)

    @property
    def z(self):
        return self._z

    @z.setter
    def z(self, z):
        if z < 0:
            raise ValueError('Redshift must be greater than 0! What crazy universe do you think you live in?!')
        else:
            if not self.z == None:
                self._data[0] = self._data[0] * (1 + self.z)
            self._data[0] = self._data[0] / (1 + z)
            self._z = z

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, array):
        self._lines = array

    @property
    def line_name(self):
        return self._line_name

    @line_name.setter
    def line_name(self, name):
        self.lines = np.array([])
        self._line_name = name

        if name == "Fe":
            self.lines = np.array([2061.56, 2068.25, 2078.99, 2097.64])
        elif name == "C":
            self.lines = np.array([2325.40])
        elif name == "Ti":
            self.lines = np.array([2512.06, 2516.07, 2527.85, 2541.82])
        elif name == "Mg":
            self.lines = np.array([2795.53, 2836.71])
        elif name == "MgTiW":
            self.lines = np.array([2512.06, 2516.07, 2527.85, 2541.82, 2795.53, 2836.71])
        elif name == "CO":
            self.lines = np.array([3920.68, 3954.36, 3973.26])
        elif name == "OII":
            self.lines = np.array([3377.19])
        elif name == "OIIL":
            self.lines = np.array([4267.26, 4345.57, 4349.43, 4366.91, 4414.89, 4416.97])
        elif name == "OIIR":
            self.lines = np.array([4638.86, 4641.83, 4650.85, 4661.64])
        elif name == "OIIW":
            self.lines = np.array([4267.26, 4345.57, 4349.43, 4366.91, 4414.89, 4416.97, 4638.86, 4641.83, 4650.85, 4661.64])
        elif name == "FeII":
            self.lines = np.array([2585.87, 2598.36, 2599.39, 2607.09, 2611.87])
        elif name == "HeI":
            self.lines = np.array([3888.6456])

    @property
    def data(self):
        return self._data


    def _line(self, x, a, b):
        return (a * x) + b

    def _profile(self, x, v, w, strength):
        prof = np.ones(x.size)

        for s in strength:
            if s < 0.0:
                prof *= 1e9

        for i in range(self.lines.size):
            prof = prof - strength[i] * norm.pdf(x, self._lines[i] / (1.0 + v/2.98e5), w)
        return prof

    def _res(self, p, x, y):
        return y-self._profile(x, p[0], p[1], p[2:])

    def fit(self, bg = None):
        if bg == None:
            raise ValueError('You must specify background boxes for continuum fitting')

        # Fit for the background
        idx = np.where(((self._data[0] > bg[0]) & (self._data[0] < bg[1])) | ((self._data[0] > bg[2]) & (self._data[0] < bg[3])))
        pLine, pcov = curve_fit(self._line, self._data[0][idx], self._data[1][idx])
        self.bg = [self._data[0][idx], self._line(self._data[0][idx], pLine[0], pLine[1])]

        # Fit the profile
        idx = np.where((self._data[0] > bg[1]) & (self._data[0] < bg[2]))
        p = 2.0*np.ones(self.lines.size + 2)
        p[0] = 10000.0
        p[1] = 20.0
        pProf, pRest = leastsq(self._res, p, args=(self._data[0][idx], self._data[1][idx] / self._line(self._data[0][idx], pLine[0], pLine[1])))

        normalise = self._line(self._data[0][idx], pLine[0], pLine[1])
        self.composite_profile = [self._data[0][idx], self._profile(self._data[0][idx], pProf[0], pProf[1], pProf[2:]) * normalise]
        self.indivisual_profiles = np.zeros((2, self.lines.size, self._data[0][idx].size))

        for i in range(self.lines.size):
            self.indivisual_profiles[0][i] = self._data[0][idx]
            self.indivisual_profiles[1][i] = (1.0 - pProf[2:][i] * norm.pdf(self._data[0][idx], self._lines[i] / (1.0 + pProf[0]/2.98e5), pProf[1])) * normalise

        return pProf
