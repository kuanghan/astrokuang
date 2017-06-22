#!/usr/bin/env python

"""
Utility functions to calculate magnitudes and flux densities.
"""
import numpy as np


def flux2mag(flux, zpt):
    """Convert flux in micro-Jansky into AB magnitude"""
    if isinstance(flux, (int, float)):
        if flux > 0:
            return zpt - 2.5 * np.log10(flux)
        else:
            return 99.0
    else:
        flux = np.array(flux)
        mag = np.where(flux > 0, zpt - 2.5 * np.log10(flux), 99.0)
        return np.minimum(mag, 99.0)


def mag2flux(mag, zpt):
    """Convert magnitude(s) into flux(es)"""
    if isinstance(mag, (int, float)):
        pass
    else:
        mag = np.array(mag)
    return 10.**(-0.4*(mag-zpt))


def magerr2sn(magerr):
    """Convert magnitude errors to S/N"""
    if isinstance(magerr, (int, float)):
        if magerr > 0:
            return 1. / (10.**(0.4*magerr)-1.)
        else:
            return -1.0
    else:
        magerr = np.array(magerr)
        return np.where(magerr > 0, 1. / (10.**(0.4*magerr)-1.), -1.0)


def sn2magerr(sn):
    """Convert S/N to magnitude errors"""
    if isinstance(sn, (int, float)):
        if sn > 0:
            return 2.5 * np.log10(1. + 1. / sn)
        else:
            return -1.0
    else:
        sn = np.array(sn)
        return np.where(sn > 0, 2.5 * np.log10(1.+1./sn), -1.0)


def flux2mag_werr(flux, fluxerr, zpt):
    """
    Convert flux & flux error into AB mag & errors.
    If flux < fluxerr, return 99 as magnitude and convert fluxerr into 1-sigma
    magnitude limit.
    """
    mag = flux2mag(flux, zpt)
    maglim = flux2mag(fluxerr, zpt)
    sn = flux / fluxerr
    magerr = sn2magerr(sn)
    if isinstance(flux, (int, float)):
        if sn < 1:
            mag = 99.0
            magerr = maglim
    else:
        mag = np.where(sn >= 1, mag, 99.0)
        magerr = np.where(sn >= 1, magerr, maglim)
    return mag, magerr


def mag2flux_werr(mag, magerr, zpt):
    """
    Convert magnitudes & errors into fluxes; assume that mag>=99.0 means
    non-detections and their magerrs are 1-sigma limits, and return flux=0
    and fluxerr=1-sigma limits in that case.
    """
    flux = mag2flux(mag, zpt)
    sn = magerr2sn(magerr)
    fluxlim = mag2flux(magerr, zpt)
    fluxerr = flux / sn

    if isinstance(mag, (int, float)):
        if mag >= 99.0:
            flux = 0.
            fluxerr = fluxlim
    else:
        flux = np.where(mag>=99.0, 0., flux)
        fluxerr = np.where(mag>=99.0, fluxlim, fluxerr)

    return flux, fluxerr


def ABmag2uJy(mag):
    """Convert AB magnitudes to micro-Jansky (zpt=23.9)"""
    return mag2flux(mag, 23.9)


def uJy2ABmag(flux):
    """Convert flux in micro-Jansky into AB magnitude"""
    return flux2mag(flux, 23.9)


def ABmag2uJy_eazy(mag, magerr, verbose=True):
    """
    Convert mag, magerr into micro-Jansky and format for EAZY catalog.
    Only works when mag, magerr are NUMBERS, not ARRAYS/LISTS.
    """
    assert isinstance(mag, (int, float)), "mag & magerr need to be numbers."
    flux, fluxerr = mag2flux_werr(mag, magerr, zpt=23.9)
    fluxstr = "%.6e  %.6e" % (flux, fluxerr)
    if verbose:
       print(fluxstr)
    return fluxstr
    # return flux, fluxerr


# def uJy2ABmag_werr(flux, fluxerr):
#     """
#     Convert flux & flux error into AB mag & mag err.
#     If flux < fluxerr, return 99 as magnitude and convert fluxerr into 1-sigma
#     magnitude limit.
#     """
#     assert isinstance(flux, (int, float)), "flux & fluxerr need to be numbers."
#     mag, magerr = flux2mag_werr(flux, fluxerr, 23.9)

#     return mag, magerr


def calcNsigMag(mag, magerr, N=1.0):
    # Given magnitude & magnitude error, calculate the expected 1-sigma 
    # magnitude. An input magnitude > 90 means it's undetected and it will 
    # just return the magnitude error
    if mag > 90:
        return magerr
    else:
        sn = magerr2sn(magerr)
        return mag + 2.5 * np.log10(sn / N)


def calcColor_mag(mag1, magerr1, mag2, magerr2, zp1=23.9, zp2=23.9):
    """
    A simple calculation of the color mag1-mag2, and add their errors in 
    quadrature. If mag1 is undetected (>90), use magerr1 as the 1-sigma upper 
    limit and calculate the lower limit of color. On the other hand, if mag2 is
    undetected, use magerr2 to calculate the upper limit of color.
    """
    if mag1 > 90:
        color = magerr1 - mag2
        print("color >= %.3f" % color)
        return color
    elif mag2 > 90:
        color = mag1 - mag2err
        print("color <= %.3f" % color)
        return color
    else:
        flux1 = 10. ** (-0.4 * (mag1 - zp1))
        flux2 = 10. ** (-0.4 * (mag2 - zp2))
        fluxerr1 = flux1 / magerr2sn(magerr1)
        fluxerr2 = flux2 / magerr2sn(magerr2)
        fluxerr_tot = np.sqrt(fluxerr1**2 + fluxerr2**2)

        # use error propogation; assuming uncorrelated errors
        ratio = flux1 / flux2
        ratio_err = np.sqrt(ratio**2 * ((1. / magerr2sn(magerr1))**2 + (1. / magerr2sn(magerr2))**2))      
        color = -2.5 * np.log10(ratio)
        colorerr = sn2magerr(ratio / ratio_err)

        # a naive calculation of color & error
        # color = mag1 - mag2
        # colorerr = np.sqrt(magerr1**2 + magerr2**2)
        print("color = %.3f +/- %.3f" % (color, colorerr))
        return color, colorerr
      

def calcColor(abmag1, magerr1, abmag2, magerr2, zp1=23.9, zp2=23.9):
    """
    A simple calculation of the color mag1-mag2, but work in flux space. I think
    this is the correct way to do it.
    """
    flux1 = ABmag2uJy(abmag1)
    fluxerr1 = flux1 / magerr2sn(magerr1)
    flux2 = ABmag2uJy(abmag2)
    fluxerr2 = flux2 / magerr2sn(magerr2)
    fluxratio = flux1 / flux2
    d_fluxratio = np.abs(fluxratio) * np.sqrt((fluxerr1/flux1)**2 + (fluxerr2/flux2)**2)
    color = -2.5 * np.log10(fluxratio)
    d_color = np.abs(d_fluxratio * 2.5 / (fluxratio * np.log(10.)))
    return color, d_color
