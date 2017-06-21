#!/usr/bin/env python

"""
Utility functions to calculate magnitudes and flux densities.
"""
import numpy as np

def ABmag2uJy(mag):
   # Convert from AB magnitude to micro-Jansky
   if isinstance(mag, (int, float)):
      if mag >= 99.0:
         return 1e-10
      else:
         return 10.**((23.9 - mag) / 2.5)
   else:
      mag = np.array(mag)
      return np.where(mag >= 99.0, 1.e-10, 10.**((23.9 - mag) / 2.5))


def ABmag2uJy_eazy(mag, magerr, verbose=True):
   """
   Convert mag, magerr into micro-Jansky. Determine if S/N >= 1, otherwise
   return 0 for the flux and 1-sigma magnitude limit for error. Format for
   EAZY catalog.
   """
   if mag < 90:
       flux = ABmag2uJy(mag)
       sn = magerr2sn(magerr)
       fluxerr = flux / sn
   else:
     if magerr < 0 or magerr > 90:
       flux = -99.0
       fluxerr = -99.0
     else:
       flux = 0.
       fluxerr = ABmag2uJy(magerr)
   if verbose:
      print("%.6e  %.6e" % (flux, fluxerr))
   return flux, fluxerr


def uJy2ABmag(flux):
   """
   Convert flux in micro-Jansky into AB magnitude.
   """
   if isinstance(flux, (int, float)):
      print("flux is a number...")
      if flux > 0:
         return 23.9 - 2.5 * np.log10(flux)
      else:
         return 99.0
   else:
      print("flux is an array...")
      flux = np.array(flux)
      return np.where(flux > 0, 23.9 - 2.5 * np.log10(flux), 99.0)
      

def uJy2ABmag_werr(flux, fluxerr):
   """
   Convert flux & flux error into AB mag & mag err.
   If flux < fluxerr, return 99 as magnitude and convert fluxerr into 1-sigma
   magnitude limit.
   """
   assert fluxerr > 0, "Flux error must >= 0!! ({} is input)".format(fluxerr)
   mag = uJy2ABmag(flux)
   sn = flux / fluxerr
   if sn >= 1:
      magerr = sn2magerr(sn)
   else:
      mag = 99.0
      magerr = uJy2ABmag(fluxerr)

   return mag, magerr


def flux2mag(flux, zpt):
   """
   Convert flux in micro-Jansky into AB magnitude.
   """
   if isinstance(flux, (int, float)):
      print("flux is a number...")
      if flux > 0:
         return zpt - 2.5 * np.log10(flux)
      else:
         return 99.0
   else:
      print("flux is an array...")
      flux = np.array(flux)
      return np.where(flux > 0, zpt - 2.5 * np.log10(flux), 99.0)


def flux2mag_werr(flux, fluxerr, zpt):
   """
   Convert flux & flux error into AB mag & mag err.
   If flux < fluxerr, return 99 as magnitude and convert fluxerr into 1-sigma
   magnitude limit.
   """
   mag = uJy2ABmag(flux)
   sn = flux / fluxerr
   magerr = sn2magerr(sn)
   return mag, magerr
      
      
def magerr2sn(magerr):
   if isinstance(magerr, (int, float)):
      if magerr > 0:
         return 1. / (10. ** (0.4 * magerr) - 1.)
      else:
         return -1.0
   else:
      magerr = np.array(magerr)
      return np.where(magerr > 0, 1. / (10. ** (0.4 * magerr) - 1.), -1.0)


def sn2magerr(sn):
   if isinstance(sn, (int, float)):
      if sn > 0:
         return 2.5 * np.log10(1. + 1. / sn)
      else:
         return -1.0
   else:
      sn = np.array(sn)
      return np.where(sn > 0, 2.5 * np.log10(1. + 1. / sn), -1.0)


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
