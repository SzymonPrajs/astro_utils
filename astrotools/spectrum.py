import numpy as np


def rebin_spectrum(wavelength, flux, wavlength_bin):
    """
    Rebin spectrum to a new wavelength bin

    Parameters
    ----------
    wavelength : array-like
        wavelength array
    flux : array-like
        flux array
    wavlength_bin : float
        size of the new wavelength bin

    Returns
    ----------
    x_new : numpy.ndarray
        Array of new rebinned wavelengths
    y_new : numpy.ndarray
        Array of rebinned fluxes
    """
    try:
        hasattr(wavelength, '__iter__')
        hasattr(flux, '__iter__')
    except TypeError:
        print('`wavelength` and `flux` must be array-like')

    wavelength = np.array(wavelength).astype(float)
    flux = np.array(flux).astype(float)
    wavlength_bin = float(wavlength_bin)

    x_new = np.arange(wavelength.min() - wavlength_bin / 2,
                      wavelength.max(),
                      wavlength_bin)
    y_new = np.zeros(x_new.size)
    flat_flux = flux[0]

    for i in range(x_new.size):
        idx = np.where((wavelength > x_new[i]) &
                       (wavelength < x_new[i] + wavlength_bin))
        y_new[i] = np.sum(flux[idx])
        div = idx[0].size

        if idx[0].size > 0:
            if (idx[0].min() > 0) and (idx[0].max() < (wavelength.size - 1)):
                y_new[i] += (((wavelength[idx].min() - x_new[i]) /
                              wavlength_bin) * flux[idx[0].min()-1])
                y_new[i] += (((x_new[i] + wavlength_bin -
                               wavelength[idx].max()) / wavlength_bin) *
                             flux[idx[0].max()+1])
                div += (wavelength[idx].min() - x_new[i])/wavlength_bin
                div += ((x_new[i] + wavlength_bin - wavelength[idx].max()) /
                        wavlength_bin)

        if div == 0:
            y_new[i] = flat_flux

        else:
            y_new[i] /= div
            flat_flux = y_new[i]

    return x_new, y_new
