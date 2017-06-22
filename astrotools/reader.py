import os
import json
import numpy as np
import pandas as pd
from .filters import zero_point, mask_present_filters

__all__ = ['read_slap', 'read_array', 'slice_band', 'slice_band_generator', 'normalize_lc', 'read_osc']


def read_slap(file_name: str) -> pd.DataFrame:
    """
    Read light curve data files as originally formatted for SLAP.
    This is a basic format of: mjd, flux, flux_err, band

    Parameters
    ----------
    file_name : str
        Path to the data file

    Returns
    -------
    data : `pandas.DataFrame`
        DataFrame object containing a light curve in the data structure
        throughout this repository
    """
    if not os.path.exists(file_name):
        raise ValueError('Path does not exists: ' + file_name)

    data = pd.read_csv(file_name, header=None, delim_whitespace=True)
    data.columns = ['mjd', 'flux', 'flux_err', 'band']

    return data


def read_array(mjd=None, flux=None, flux_err=None, band=None) -> pd.DataFrame:
    """
    Convert arrays of MJD, flux, flux_err and band into a common format
    pandas DataFrame object.

    Parameters
    ----------
    mjd : array-like
        Numerical array-like object of MJD values.
        Must be specified.

    flux : array-like
        Numerical array-like object of MJD values.
        Must be specified.

    flux_err : numerical or array-like
        Either a single numerical or array-like object of flux_err.
        If array-like then the shape of this array must match that of `flux`.

    band : str or array-like
        Either a single string or array-like object of bands.
        If array-like then the shape of this array must match that of `flux`.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame object containing light curve data that can be used
        in most routines throughout this repository.
    """
    df = pd.DataFrame()

    if mjd is None or flux is None or mjd.shape != flux.shape:
        raise ValueError('Data must contain both MJD and flux values and be of the same dimension')

    try:
        hasattr(mjd, '__iter__')
        hasattr(flux, '__iter__')

    except TypeError:
        print('`mjd` and `flux` must be array-like')

    try:
        df['mjd'] = np.array(mjd).astype(float)
        df['flux'] = np.array(flux).astype(float)

    except ValueError:
        print('`mjd` and `flux` must be numerical')

    if hasattr(flux_err, '__iter__') and flux.shape == flux_err.shape:
        df['flux_err'] = np.array(flux_err).astype(float)

    else:
        df['flux_err'] = float(flux_err)

    df['band'] = band

    return df


def read_osc(json_file_path):
    """
    Read light curves from Open Supernova Catalogue (OSC)
    JSON files and parse into the common DataFrame format.

    Parameters
    ----------
    json_file_path : str
        Path to the OSC JSON file

    Returns
    -------
    data : `pandas.DataFrame`
        DataFrame object in the common format
    """
    data = None

    if not os.path.exists(json_file_path):
        raise ValueError('File does not exists: ' + json_file_path)

    with open(json_file_path) as json_file:
        json_data = json.load(json_file)
        object_name = list(json_data.keys())[0]

        if 'photometry' in json_data[object_name]:
            data = pd.DataFrame(json_data[object_name]['photometry'])

        else:
            raise ValueError('No photometry found in the JSON file')

    # TODO: Replace row names with common synonyms

    data = data[mask_present_filters(data['band'])]

    data['zp'] = data.apply(lambda x: zero_point(x['band']), axis=1)
    data['flux'] = 10 ** (-0.4 * (data['magnitude'].astype(float) + data['zp']))
    data['flux_err'] = data['e_magnitude'].astype(float) * 0.921034 * data['flux']

    return data


def slice_band(data, band=None):
    """
    Return a slice of the input DataFrame or a dictionary
    of DataFrames indexed by the filter name.
    
    Parameters
    ----------
    data : `pandas.DataFrame`
        DataFrame object in the common format containing:
        MJD, flux, flux_err, band

    band : str or array-like, optional
        If a single band is provided a single DataFrame object
        will be returned, with more than one filter resulting
        in a dictionary of DataFrame objects.

    Returns
    -------
    data_dict : `pandas.DataFrame` or dict
    """
    data_list = list(slice_band_generator(data, band=band))

    if len(data_list) == 1:
        return data_list[0]

    else:
        data_dict = {}

        for data in data_list:
            band_name = data['band'].unique()[0]
            data_dict[band_name] = data

        return data_dict


def slice_band_generator(data, band=None):
    """
    Generator retuning a series of DataFrame objects,
    each containing the light curve for just one, unique band.

    Parameters
    ----------
    band : str or array-like, optional
        If band is specified a DataFrame a generator returning
        objects of only that band are returned.

    data : `pandas.DataFrame`
        DataFrame object in the common format containing:
        MJD, flux, flux_err, band

    Yields
    ------
    data : `pandas.DataFrame`
        Object containing the light curve for one, unique band.
    """
    if band is None:
        unique_bands = data['band'].unique()

    elif hasattr(band, '__iter__'):
        unique_bands = band

    else:
        unique_bands = [band]

    for band_iter in unique_bands:
        yield data.query('band == "{}"'.format(band_iter))


def normalize_lc(data):
    """
    Normalise light curve flux and flux_err

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame object to be normalised. Must contain:
        flux, flux_err
    """
    norm_factor = data['flux'].max()
    data['flux'] /= norm_factor
    data['flux_err'] /= norm_factor
    data['norm_factor'] = norm_factor

    return data
