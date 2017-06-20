import os
import numpy as np
import pandas as pd


__all__ = ['read_slap', 'read_array', 'slice_band', 'normalize_lc']


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


def slice_band(data):
    """
    Generator retuning a series of DataFrame objects,
    each containing the light curve for just one, unique band.

    Parameters
    ----------
    data : `pandas.DataFrame`
        DataFrame object in the common format containing:
        MJD, flux, flux_err, band

    Yields
    ------
    data : `pandas.DataFrame`
        Object containing the light curve for one, unique band.
    """
    unique_bands = data['band'].unique()

    for band in unique_bands:
        yield data.query('band == "{}"'.format(band))


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
