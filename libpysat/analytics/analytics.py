import numpy as np
from pandas import Series
from scipy import signal


def within_range(data, rangevals, col):
    mask = (data[('meta', col)] > rangevals[0]) & (data[('meta', col)] < rangevals[1])
    return data.loc[mask]


def band_minima(spectrum, low_endmember=None, high_endmember=None):
    """
    Given two end members, find the minimum observed value inclusively
    between them.

    Parameters
    ==========
    spectrum : pd.series
               Pandas series

    low_endmember : float
                    The low wavelength

    high_endmember : float
                     The high wavelength

    Returns
    =======
    minidx : int
             The wavelength of the minimum value

    minvalue : float
               The observed minimal value
    """
    x = spectrum.index
    y = spectrum

    if not low_endmember:
        low_endmember = x[0]
    if not high_endmember:
        high_endmember = x[-1]

    ny = y[low_endmember:high_endmember + 1]

    minidx = ny.idxmin()
    minvalue = ny.min()

    return minidx, minvalue


def band_center(spectrum, low_endmember=None, high_endmember=None, degree=3):
    x = spectrum.index
    y = spectrum

    if not low_endmember:
        low_endmember = x[0]
    if not high_endmember:
        high_endmember = x[-1]

    ny = y[low_endmember:high_endmember + 1]

    fit = np.polyfit(ny.index, ny, degree)

    center_fit = Series(np.polyval(fit, ny.index), ny.index)
    center = band_minima(center_fit)

    return center, center_fit


def band_area(spectrum, low_endmember=None, high_endmember=None):
    """
    Compute the area under the curve between two endpoints where the
    y-value <= 1.
    """

    x = spectrum.index
    y = spectrum

    if not low_endmember:
        low_endmember = x[0]
    if not high_endmember:
        high_endmember = x[-1]

    ny = y[low_endmember:high_endmember + 1]

    return np.trapz(-ny[ny <= 1.0])


def band_asymmetry(spectrum, low_endmember=None, high_endmember=None):
    """
    Compute the asymmetry of an absorption feature as
    (left_area - right_area) / total_area

    Parameters
    ----------
    specturm : object

    low_endmember : int
        Bottom end of wavelengths to be obversed

    high_endmember : int
        Top end of wavelengths to be obversed

    Returns
    -------
    asymmetry : ndarray
        Array of percentage values of how asymmetrical the two halves of the spectrum are
        Where 100% is completely asymmetrical and 0 is completely symmetrical
    """

    x = spectrum.index
    y = spectrum

    if not low_endmember:
        low_endmember = x[0]
    if not high_endmember:
        high_endmember = x[-1]

    ny = y[low_endmember:high_endmember + 1]

    center, _ = band_center(ny, low_endmember, high_endmember)
    area_left = band_area(ny[:center[0]], low_endmember, high_endmember)
    area_right = band_area(ny[center[0]:], low_endmember, high_endmember)

    asymmetry = (area_left - area_right) / (area_left + area_right)
    return asymmetry


def get_noise(data, n_iter = 3):
    """
    Finds the standard deviation of white gaussian noise in the data

    Parameters
    ----------
    data : ndarray
        IDL array of data

    n_iter : int
        Number of iterations to attempt in a sigma clip

    Returns
    -------
    sigma : float
        Standard deviation of white guassian noise
    """

    vsize = data.shape
    dim = len(vsize)
    sigma = -1
    if dim == 3:
        nco = vsize[0]
        nli = vsize[1]
        npz = vsize[2]
        indices = range(npz - 2) + 1
        d_cube = np.array(nco, nli, npz)
        c1 = -1. / np.sqrt(6.)
        c2 = 2. / np.sqrt(6.)
        d_cube[:, :, 1:npz - 1] = c1 * (data[:, :, indices - 1] + data[:, :, indices + 1]) + c2 * data[:, :, indices]
        d_cube[:, :, 0] = c2 * (data[:, :, 0] - Data[:, :, 1])
        d_cube[:, :, npz - 1] = c2 * (data[:, :, npz - 1] - data[:, :, npz - 2])
        sigma = sigma_clip(d_cube, n_iter=n_iter)
    if dim == 2:
        # ;im_smooth, Data, ima_med, winsize=3, method='median'
        sigma = sigma_clip(data - ima_med, n_iter=n_iter) / 0.969684
    if dim == 1:
        sigma_out, mean = sigma_clip(data - signal.medfilt(data, 3), n_iter=n_iter)
        sigma = sigma_out / 0.893421

    return sigma


def sigma_clip(data, sigma_clip=3.0, n_iter=2.0):
    """
    Returns the sigma obtained by k-sigma. If mean is set, the
    mean (taking into account outsiders) is returned.

    Parameters
    ----------
    data : ndarray
        IDL data array

    sigma_clip : float
        Sigma clip value

    n_iter : float
        Number of iterations

    Returns
    -------
    sig : float
        Sigma obtained via k-sigma

    mean : float
        Mean value
    """

    output = ''

    n_iter = n_iter - 1
    sig = 0.
    buff = data

    mean = np.sum(buff) / len(buff)
    sig = np.std(buff)
    index = np.where(abs(buff - mean) < sigma_clip * sig)
    count = len(buff[index])

    for i in range(1, n_iter):
        if count > 0:
            mean = np.sum(buff[index]) / len(buff[index])
            sig = np.std(buff[index])
            index = np.where(abs(buff - mean) < sigma_clip * sig)
            count = len(buff[index])

    return sig, mean
