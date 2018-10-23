import numpy as np
from pandas import Series


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
