import numpy as np
import astropy.cosmology
import astropy.units as u


__all__ = ['rebin_spectrum',
           'Spectrum']


def rebin_spectrum(wavelength, flux, wavelength_bin):
    """
    Rebin spectrum to a new wavelength bin

    Parameters
    ----------
    wavelength : array-like
        wavelength array

    flux : array-like
        flux array

    wavelength_bin : float
        size of the new wavelength bin

    Returns
    -------
    x_new : `numpy.ndarray`
        Array of new rebinned wavelengths

    y_new : `numpy.ndarray`
        Array of rebinned fluxes
    """
    try:
        hasattr(wavelength, '__iter__')
        hasattr(flux, '__iter__')

    except TypeError:
        print('`wavelength` and `flux` must be array-like')

    wavelength = np.array(wavelength).astype(float)
    flux = np.array(flux).astype(float)
    wavelength_bin = float(wavelength_bin)

    x_new = np.arange(wavelength.min() - wavelength_bin / 2, wavelength.max(), wavelength_bin)
    y_new = np.zeros(x_new.size)
    flat_flux = flux[0]

    for i in range(x_new.size):
        idx = np.where((wavelength > x_new[i]) & (wavelength < x_new[i] + wavelength_bin))
        y_new[i] = np.sum(flux[idx])
        div = idx[0].size

        if idx[0].size > 0 and (idx[0].min() > 0) and (idx[0].max() < (wavelength.size - 1)):
            y_new[i] += ((wavelength[idx].min() - x_new[i]) / wavelength_bin) * flux[idx[0].min()-1]
            y_new[i] += ((x_new[i] + wavelength_bin - wavelength[idx].max()) / wavelength_bin) * flux[idx[0].max()+1]

            div += (wavelength[idx].min() - x_new[i]) / wavelength_bin
            div += ((x_new[i] + wavelength_bin - wavelength[idx].max()) / wavelength_bin)

        if div == 0:
            y_new[i] = flat_flux

        else:
            y_new[i] /= div
            flat_flux = y_new[i]

    return x_new, y_new


class Spectrum:
    """
    Generic spectrum class

    Spectrum class for storing, loading and manipulating spectral data from
    any astronomical source. Build in operations include redshifting,
    synthesising photometry and resampling of spectra.

    Either a file name or arrays containing wavelength and flux values must be
    provided. If both get provided the file will still be read but the inputs
    will be overwritten by the values supplied.

    This class is currently expecting to receive wavelengths in Angstroms and
    fluxes in erg/s/cm^2/A. This may be generalised in the future. While the
    class will continue operating all magnitudes and other statistics provided
    will be completely non sensical without the correct units.

    Parameters
    ----------
    file_name : str, optional
        Path to an ASCII file containing a spectrum. The input is expected to
        have a minimum of two columns where the first column contains
        wavelengths in angstroms and the second column fluxes in units of
        erg/s/cm^2/A.

    wavelength : array-like, optional
        Wavelength array for the spectra, must be a 1d array of floats with the
        same dimension as the corresponding array of fluxes. Wavelengths must
        be provided in Angstroms. Used if an input ASCII file is not provided.

    flux : array-like, optional
        Flux array for the specta, must be a 1d array of floats with the
        same dimension as the corresponding array of wavelengths.
        Flux must be provided in the units of erg/s/cm^2/A.
        Used if an input ASCII file is not provided.

    redshift : float, optional
        Redshift of the supernova. If no value is provided z=0 will be assumed
        and luminosity_distance will be set to a default value of 10pc.

    cosmology : `astropy.cosmology.FlatLambdaCDM`, optional
        Cosmology object used to compute luminosity distances. If no value is
        provided Planck15 cosmology will be assumed by default.
    """
    def __init__(self, file_name=None, wavelength=None, flux=None, redshift=0, cosmology=None):
        if file_name is not None:
            self.load_from_file(file_name)

        elif (wavelength is None) and (flux is None):
            raise IOError("""Either file_name or wavelength and flux must be provided""")

        if wavelength is not None:
            try:
                iter(wavelength)
                wavelength = np.array(wavelength)

            except TypeError:
                print('wavelength must be 1d array-like')

            try:
                wavelength = wavelength.astype(float)

            except ValueError:
                print('wavelength array must numeric')

            self._wavelength = wavelength

        if flux is not None:
            try:
                iter(flux)
                flux = np.array(flux)

            except TypeError:
                print('flux must be 1d array-like')

            try:
                flux = flux.astype(float)

            except ValueError:
                print('flux array must numeric')

            self._flux = flux

        if not self._wavelength.shape == self._flux.shape:
            raise ValueError('wavelength and flux must be have same shape')

        if not self._wavelength.shape[0] == self._wavelength.size:
            raise ValueError('wavelength and flux must be 1d arrays')

        if cosmology is None:
            self.__cosmology = astropy.cosmology.Planck15

        elif type(cosmology) == astropy.cosmology.core.FlatLambdaCDM:
            self.__cosmology = cosmology

        else:
            raise TypeError("""cosmology must be of type astropy.cosmology.core.FlatLambdaCDM""")

        if float(redshift) < 0:
            raise ValueError('Redshift must be positive!')

        else:
            self._z = redshift

        if self._z == 0:
            self._lum_distance = 10e-6 * u.Mpc

        else:
            self._lum_distance = self.__cosmology.luminosity_distance(self._z)

    def load_from_file(self, file_name):
        """
        Load spectrum from file

        Parameters
        ----------
        file_name : str
            Path of the ASCII spectrum input file. It must contain at least
            two columns; wavelength and flux.

        Returns
        -------
        None
        """
        arr = []

        try:
            arr = np.loadtxt(file_name, unpack=True)

        except IOError:
            print('Could not read {}'.format(file_name))

        if arr.shape[0] > 1:
            arr = arr[0:2]

            try:
                arr = arr.astype(float)

            except:
                raise TypeError('ASCII input file must contain floats')

        self._wavelength = arr[0]
        self._flux = arr[1]

    def update_flux(self, flux):
        # TODO: docstring
        if flux is not None:
            try:
                iter(flux)
                flux = np.array(flux)

            except TypeError:
                print('flux must be 1d array-like')

            try:
                flux = flux.astype(float)

            except ValueError:
                print('flux array must numeric')

            self._flux = flux

    def adjust_redshift(self, new_redshift):
        """
        Move the spectrum to a new input redshift

        Moves the redshift of the spectrum by adjusting the wavelength and
        flux by factors of (1 + new_redshift) / (1 + old_redshift) and its
        inverse respectively. Flux is also adjusted by a square of the ratios
        of the luminosity distances at both redshifts.

        Parameters
        ----------
        new_redshift : float
            New redshift of the spectrum

        Returns
        -------
        None
        """
        redshift_shift = (1 + new_redshift) / (1 + self._z)
        self._wavelength *= redshift_shift
        self._flux /= redshift_shift
        self._z = new_redshift

        if self._z == 0:
            new_lum_distance = 10e-6 * u.Mpc

        else:
            new_lum_distance = self.__cosmology.luminosity_distance(self._z)

        distance_factor = (self._lum_distance / new_lum_distance)**2
        self._flux *= distance_factor.value
        self._lum_distance = new_lum_distance

    def synthesis_photometry(self, filter_name, filters):
        """
        Make synthetic photometry from a spectrum

        Parameters
        ----------
        filter_name : str or array-like
            Value or an array of names of filters at which the synthetic
            photometry is to be calculated.

        filters : `astrotools.Filters`
            Filters object that must be preloaded with the filters passed as
            filter_name

        Returns
        -------
        synthetic_flux : ndarray
            Array of synthetic fluxes matching the input filter_name
        """
        if not hasattr(filter_name, '__iter__'):
            filter_name = [filter_name]

        filter_name = np.array(filter_name)
        synthetic_flux = np.zeros(filter_name.size)

        for i, flt in enumerate(filter_name):
            bandpass = np.interp(self._wavelength, filters[flt].wavelength, filters[flt].bandpass, left=0, right=0)
            flux = self._flux * bandpass
            synthetic_flux[i] = (np.trapz(flux, x=self._wavelength) / filters[flt].area)

        return synthetic_flux

    def synthesis_magnitudes(self, filter_name, filters):
        """
        Calculate synthetic magnitudes for the spectrum using provided filters

        Parameters
        ----------
        filter_name : str or array-like
            Value or an array of names of filters at which the synthetic
            photometry is to be calculated.

        filters : `astrotools.Filters`
            Filters object that must be preloaded with the filters passed as
            filter_name

        Returns
        -------
        synthetic_mag: np.ndarray
            Array of synthetic magnitudes matching the input filter_name
        """
        if not hasattr(filter_name, '__iter__'):
            filter_name = [filter_name]

        synthetic_flux = self.synthesis_photometry(filter_name, filters)
        synthetic_mag = np.zeros_like(synthetic_flux)

        for i, flt in enumerate(filter_name):
            synthetic_mag[i] = (-2.5*np.log10(synthetic_flux[i]) - filters[flt].ab_zero_point)

        return synthetic_mag

    def compute_colour_correction(self, observed_filter, referenceframe_filter, filters, redshift=0):
        """
        Compute K-correction from a spectrum

        Calculates the k-correction term between an observed filter specified
        in the input and a filter at a reference frame. If the redshift of the
        reference frame is not specifies the code assumes rest-frame (z=0).

        Parameters
        ----------
        observed_filter : str
            Name of the filter in observer frame that will be k-corrected to
            the reference frame
        referenceframe_filter : str
            Reference frame filter at which we want to place the spectrum
        redshift : float, optional
            redshift of the reference frame. If not set it will default to z=0

        Returns
        kcorr : float
            k-correction between the input and output filter at the new
            redshift.
        """
        pass
        # TODO: This needs to be implemented
