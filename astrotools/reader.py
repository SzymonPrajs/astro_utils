import pandas as pd


__all__ = ['read_slap_files']


def read_slap_files(file_name):
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
    data = pd.read_csv(file_name, header=None, delim_whitespace=True)
    data.columns = ['mjd', 'flux', 'flux_err', 'band']

    return data
