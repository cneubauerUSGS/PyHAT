# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:11:37 2016

@author: rbanderson
"""
import libpysat.transform.baseline_code.watrous as watrous
import numpy

"""
Created on Tue Nov 11 18:29:29 2014
This function is used to denoise a chemcam spectrum.
Based on the function "denoise_spectrum.pro" in IDL
Translated to Python by Ryan Anderson Nov 2014. Modified so that the denoised
spectrum, and the removed noise are returned.
@author: rbanderson
"""


def ccam_denoise(sp_in, sig = 3, n_iter = 4):
    """
    Denoises a chemcam spectrum. Based on the function "denoise_spectrum.pro" in IDL

    Parameters
    ----------
    sp_in : ndarray
        Array of spectrum data

    sig : int
        Unknown

    n_iter : int
        Number of iterations to refine the noise removal

    Returns
    -------
    : float
        Removed noise
    : ndarray
        The denoised spectrum
    """

    s = len(sp_in)
    lv = int(numpy.log(s) / numpy.log(2)) - 1
    ws = watrous.watrous(sp_in, lv)
    ws1 = ws

    for i in range(lv - 2):
        b = get_noise(ws[:, i], n_iter=n_iter)
        tmp = ws[:, i]
        ou = numpy.where(abs(tmp) < sig * b)
        nou = len(tmp[ou])

        if nou > 0:
            tmp[ou] = 0

        ws1[:, i] = tmp

    return numpy.sum(ws1, axis=1), sp_in - numpy.sum(ws1, axis=1)
