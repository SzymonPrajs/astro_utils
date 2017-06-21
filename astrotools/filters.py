import os
import numpy as np
import pandas as pd
from glob import glob
import scipy.constants as const

FILTER_DIR = "data/filters/"

__all__ = ['zero_point', 'list_available_filter', 'list_available_instruments', 'is_ab_band', 'get_filter_path']


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
    # TODO: docstring
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
    # TODO: docstring
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
    # TODO: docstring
    try:
        filter_data = pd.read_csv(filter_path, delim_whitespace=True, header=None)

    except OSError:
        raise ValueError('Filter path does not exist: ' + filter_path)

    filter_data.columns = ['wavelength', 'bandpass']

    return filter_data


def get_filter_path(band, instrument):
    # TODO: docstring
    return FILTER_DIR + instrument + '_' + band + '.dat'


def zero_point(band, system=None, instrument=None, round_output=True):
    # TODO: docstring
    band_list = list_available_filter()

    if band not in band_list:
        raise ValueError('No filter response found for: ' + band)

    if instrument is None:
        if len(band_list[band]) == 1:
            instrument = band_list[band][0]

        else:
            instrument = 'Generic'

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
    mag = -2.5 * np.log10(flux_area / filter_area)

    if round_output:
        mag = np.round(mag, 2)

    return mag
