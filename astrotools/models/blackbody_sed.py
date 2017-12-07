import numpy as np
from ..spectrum import Spectrum
import astropy.units as u
import astropy.constants as const


__all__ = ['BlackbodySED']


class BlackbodySED(Spectrum):
    """
    Blackbody SED model

    Child class of the Spectrum class populates the wavelength and flux with
    values calculated based on the

    Parameters
    ----------
    temperature : float or `astropy.units.quantity.Quantity`
        Photospheric temperature of the object. If floats are provided the
        values are assumed to be in the unit of `Kelvin`.

    radius : float or `astropy.units.quantity.Quantity`
        Radius of the photosphere. If float is provided it will be assumed to
        be in unit of `cm`. Alternatively an astropy.units.quantity.Quantity
        can be passed which will be internally converted into the correct units

    redshift : float, optional
        Redshift at which to place the object. Default assumes rest-frame (z=0)

    wavelength : array-like, optional
        Wavelength at which the Planck function is to be evaluated. Can be
        given as an array of floats or `astropy.units.quantity.Quantity`. If no
        input is provided, the default will be set to a range of 3000A - 10000A
    """
    def __init__(self, temperature=10000, radius=1e15, redshift=0, wavelength=None, template=None):
        if type(temperature) != u.quantity.Quantity:
            try:
                temperature = float(temperature) * u.K

            except:
                raise ValueError('Temperature must be numeric or u.Quantity')

        if type(radius) != u.quantity.Quantity:
            try:
                radius = float(radius) * u.cm

            except:
                raise ValueError('Radius must be a float or astropy.units')

        if wavelength is None:
            wavelength = np.arange(3000, 10000, 10, dtype=float) * u.Angstrom

        else:
            try:
                wavelength = np.array(wavelength).astype(float) * u.Angstrom

            except:
                raise ValueError('Wavelength must be a numeric array')

        if template is None:
            transmitance = np.ones(wavelength.size)
            template_wavelength = wavelength.value

        else:
            template_wavelength, transmitance = np.loadtxt(template, unpack=True)

        np.interp(wavelength, transmitance, template_wavelength)

        self._temperature = temperature
        self._radius = radius
        self._input_z = redshift
        self._input_wav = wavelength / (1 + self._input_z)
        self._corr = np.interp(self._input_wav, template_wavelength, transmitance, left=0.5, right=1.0)

        sed = self.compute_absolute_blackbody()
        super().__init__(wavelength=self._input_wav, flux=sed, redshift=0)
        self.adjust_redshift(self._input_z)


    def update_blackbody(self, temperature=None, radius=None):
        # TODO: docstring
        if temperature is not None:
            if type(temperature) != u.quantity.Quantity:
                try:
                    temperature = float(temperature) * u.K

                except:
                    raise ValueError('Temperature must be numeric or u.Quantity')

            self._temperature = temperature

        if radius is not None:
            if type(radius) != u.quantity.Quantity:
                try:
                    radius = float(radius) * u.cm

                except:
                    raise ValueError('Radius must be a float or astropy.units')

            self._radius = radius

        temp_z = self._z
        super(BlackbodySED, self).adjust_redshift(0)
        super(BlackbodySED, self).update_flux(self.compute_absolute_blackbody())
        super(BlackbodySED, self).adjust_redshift(temp_z)

    def compute_absolute_blackbody(self):
        """
        Compute the Planck function for an astronomical object at rest-frame

        Calculated a blackbody SED in an astronomical context for an object
        placed at 10pc (internal distance at z=0). All internal calculations
        are performed in SI units using the astropy.units and astropy.constants
        libraries for consistency.

        Returns
        -------
        sed : np.array of `astropy.units.quantity.Quantity`
            Return an array of fluxes for corresponding input wavelength values
            in the units of `erg/s/c^2/A` using the astropy.units library
        """
        sed = 2.0 * const.h * np.pi * const.c**2 / self._input_wav**5
        sed /= np.exp(const.h * const.c / (self._input_wav * const.k_B * self._temperature)) - 1.0
        sed *= self._radius**2
        sed /= ((10 * u.pc).to(u.cm))**2
        sed *= self._corr

        return sed.to(u.erg / u.s / u.cm**2 / u.Angstrom)