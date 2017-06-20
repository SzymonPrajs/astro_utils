import os


__all__ = ['zero_point', 'list_available_filter', 'is_ab_band']


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
    filter_dir = "../data/filters"
    filter_list = os.listdir(filter_dir)

    if '.DS_Store' in filter_list:
        filter_list.remove('.DS_Store')

    # TODO: Return a dict of filters and with lists of available instruments


# TODO: Function similar to list_available_filters returning a dict of instruments


def zero_point(band, system=None, instrument=None):
    # TODO: docstring

    # TODO: Implement a band and instrument dependant zero_point function
    pass
