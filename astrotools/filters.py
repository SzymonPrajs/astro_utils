import os
import numpy as np
import pandas as pd
from glob import glob
import scipy.constants as const
from astropy.constants import c

PACKAGE_PATH = os.path.dirname(__file__)
FILTER_DIR = os.path.join(PACKAGE_PATH, 'data/filters/')

__all__ = ['zero_point',
           'list_available_filter',
           'list_available_instruments',
           'is_ab_band',
           'get_filter_path',
           'mask_present_filters',
           'mask_present_instrument']


def is_ab_band(band):
    """
    Check if the default system for the provided
    band is AB.

    Parameters
    ----------
    band : str
        Name of a photometric filter

    Returns
    -------
    bool
        True if the default system for this band is AB
    """
    ab_bands = ['u', 'g', 'r', 'i', 'z', 'y']

    if band in ab_bands:
        return True

    else:
        return False


def list_available_filter():
    """
    List all available filter responses

    Returns
    -------
    filter_dict : dict
        Dictionary of filter names containing lists of available instruments
    """
    filter_list = glob(FILTER_DIR + '*_*.dat')

    filter_dict = {}
    for filter_file in filter_list:
        filter_file = os.path.basename(filter_file)
        instrument, band = filter_file.split('.')[0].split('_')

        if band not in filter_dict:
            filter_dict[band] = [instrument]

        else:
            filter_dict[band].append(instrument)

    return filter_dict


def list_available_instruments():
    """
    List all available instruments

    Returns
    -------
    filter_dict : dict
        Dictionary of instrument names containing available filter responses
    """
    filter_list = glob(FILTER_DIR + '*_*.dat')

    filter_dict = {}
    for filter_file in filter_list:
        filter_file = os.path.basename(filter_file)
        instrument, band = filter_file.split('.')[0].split('_')

        if instrument not in filter_dict:
            filter_dict[instrument] = [band]

        else:
            filter_dict[instrument].append(band)

    return filter_dict


def load_filter(filter_path):
    """
    Load a filter response into a pandas DataFrame object
    Parameters
    ----------
    filter_path : str
        Path to the filter response file

    Returns
    -------
    filter_data : `pandas.DataFrame`
        DataFrame object containing: wavelength and bandpass
    """
    try:
        filter_data = pd.read_csv(filter_path, delim_whitespace=True, header=None)

    except OSError:
        raise ValueError('Filter path does not exist: ' + filter_path)

    filter_data.columns = ['wavelength', 'bandpass']

    return filter_data


def get_filter_path(band, instrument):
    """
    Get the path to a filter response file

    Parameters
    ----------
    band : str
        Name of a photometric band

    instrument : str
        Name of an instrument. This cannot be a synonym.

    Returns
    -------
    filter_path : str
        Path to the filter response file
    """
    return FILTER_DIR + instrument + '_' + band + '.dat'


def mask_present_filters(filter_list):
    # TODO: docstring
    if not hasattr(filter_list, '__iter__'):
        raise ValueError('`filter_list` must be array-like')

    filter_series = pd.Series(filter_list)
    available_filters = list_available_filter()

    filter_mask = filter_series.map(lambda band: band in available_filters)

    return filter_mask.values


def mask_present_instrument(instrument_list):
    # TODO: docstring
    if not hasattr(instrument_list, '__iter__'):
        raise ValueError('`instrument_list` must be array-like')

    instrument_series = pd.Series(instrument_list)
    available_instruments = list_available_instruments()

    instrument_mask = instrument_series.map(lambda band: band in available_instruments)

    return instrument_mask.values


def zero_point(band, system=None, instrument=None, round_output=True):
    """
    Calculate the zero point for a given filter

    Parameters
    ----------
    band : str
        Name of a photometric filter

    system : str, optional
        Choose between AB and Vega photometric system. If not specified
        AB will be used for the SDSS (lower case) bands and Vega for
        everything else.

    instrument : str, optional
        Name of the instrument used (or filter system). Current choices are:
        SDSS, DES, LSST, PS1, SNLS, PTF48, Bessell, 2MASS, Swift, HST, HAWK-I, LSQ.
        If no instrument is specifies, SDSS filters are used for the AB system and
        Bessell for Vega.

    round_output : bool
        If True, the output will be rounded to two decimal points.

    Returns
    -------
    zp : float
        Zero point for the specified band, instrument and photometric system
    """
    band_list = list_available_filter()

    if band not in band_list:
        raise ValueError('No filter response found for: ' + band)

    if instrument is None:
        if len(band_list[band]) == 1:
            instrument = band_list[band][0]

        else:
            if is_ab_band(band):
                instrument = 'SDSS'

            else:
                instrument = 'Bessell'

    if instrument not in band_list[band]:
        raise ValueError('This combination of instrument and band does not exists: ' + instrument + '_' + band)

    # TODO: Check for instrument synonyms

    if system is None:
        if is_ab_band(band):
            system = 'ab'

        else:
            system = 'vega'

    system = system.lower()

    try:
        filter_data = load_filter(get_filter_path(band, instrument))

    except ValueError:
        raise ValueError('Missing filter responses for instrument_band combination: ' + instrument + '_' + band)

    if system == 'vega':
        try:
            vega_data = load_filter(FILTER_DIR + 'Vega.dat')
        except ValueError:
            raise ValueError('Missing Vega spectrum')

        flux = np.interp(filter_data['wavelength'], vega_data['wavelength'], vega_data['bandpass'])
        flux *= filter_data['bandpass']

    elif system == 'ab':
        jansky = 3631 * 1e-23 * const.c * 1e10 / (filter_data['wavelength'])**2
        flux = jansky * filter_data['bandpass']

    else:
        raise ValueError('Unrecognised photometric system')

    flux_area = np.trapz(flux, filter_data['wavelength'])
    filter_area = np.trapz(filter_data['bandpass'], filter_data['wavelength'])
    zp = -2.5 * np.log10(flux_area / filter_area)

    if round_output:
        zp = np.round(zp, 2)

    return zp


class _Filter:
    """
    Filter response data structure (for internal use only)

    Stores individual filter responses and their statistics. The objects are
    loaded and accessed by the user though the main Filters class.

    Parameters
    ----------
    filter_path : str
        Path to the filter response to be loaded
    """
    def __init__(self, filter_path):
        # TODO: Do all appropriate checks
        self.file_path = filter_path
        self.base_name = os.path.basename(self.file_path)
        self.name = os.path.splitext(self.base_name)[0]

        # TODO: Check if file has correct format
        filter_data = np.loadtxt(filter_path, unpack=True)
        self.wavelength = filter_data[0]
        self.bandpass = filter_data[1]
        self.area = np.trapz(self.bandpass, x=self.wavelength)
        jansky = 3631 * 1e-23 * c.value * 1e10 / self.wavelength ** 2
        flux = jansky * self.bandpass
        self.ab_zero_point = -2.5*np.log10(np.trapz(flux, x=self.wavelength) / self.area)
        self.central_wavelength = np.trapz(self.bandpass * self.wavelength,  x=self.wavelength) / self.area


class Filters():
    """
    Filter responses utility

    Generic functionality for loading, providing statistics and using
    astronomical filter responses. This can be used completely independently of
    of other classes that are part of this package and should work with both
    numpy array and pandas DataFrames

    Parameters
    ----------
    load_all : bool, optional
        Filter response files are not loaded by default to minimise the load
        time for the module in case a large number of filter responses are
        provided. This can be changed by setting this flag to True.
    """
    def __init__(self, load_all=False):
        self.__path_list = np.array(glob(FILTER_DIR + '*.dat'))
        self.__base_names = list(map(os.path.basename, self.__path_list))
        self.__filter_names = np.array([b[0] for b in map(os.path.splitext, self.__base_names)])

        self.filters = {}
        self._loaded_filters = []

        if load_all is True:
            self.load_filters(load_all=True)

    def __getitem__(self, filter_name):
        """
        `Official` access point for individual filter

        For ease and readability the __getitem__ method was overwritten to
        returns an object of the class _Filter after checking if a given
        filter has been loaded. This is safer than the accessing the
        Filters.filters dictionary directly however it is marginally slower.

        Parameters
        ----------
        filter_name : str
            Name of the filter to be accessed

        Returns
        -------
        filter : _Filter
            Object of the class _Filter containing filter responses and stats
        """
        return self.filters[filter_name]

    def load_filters(self, filter_name=None, load_all=False):
        # TODO: Docstring
        """
        """
        if load_all is True:
            filters_to_load = np.array(self.__filter_names)
        elif filter_name is not None:
            filters_to_load = np.array(filter_name)
        else:
            raise ValueError('List of filters was not provided')

        for ftl in filters_to_load:
            idx = np.where(self.__filter_names == ftl)
            if idx[0].size == 0:
                raise Warning('No filter response for {}'.format(ftl))
            else:
                self.filters[ftl] = _Filter(self.__path_list[idx][0])
                self._loaded_filters.append(ftl)

    def list_available(self):
        """
        List all available filters
        """
        return self.__filter_names

    def list_loaded(self):
        """
        List all filters that have already been loaded
        """
        return np.unique(np.array(self._loaded_filters))
