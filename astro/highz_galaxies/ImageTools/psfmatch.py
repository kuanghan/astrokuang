#!/usr/bin/env python

import numpy as np
from astropy.io import fits
# from scipy import signal, ndimage
# from hconvolve import fftdeconvolve, hconvolve
from scipy.signal import convolve2d
import os, subprocess
from scipy import ndimage
from skimage.restoration import richardson_lucy
from photutils import CosineBellWindow, create_matching_kernel
window_default = CosineBellWindow(alpha=0.35)


def psfmatch(psf_sharp, psf_broad, kernelname, method='fft', window=window_default, iterations=30):
    """Derive the kernel that matches psf_sharp to psf_broad"""
    psf1 = fits.getdata(psf_sharp)
    psf2 = fits.getdata(psf_broad)

    assert psf1.shape[0] % 2 == 1
    assert psf2.shape[0] % 2 == 1
    assert psf1.shape[0] == psf1.shape[1]
    assert psf2.shape[0] == psf2.shape[1]
    
    psf1 = psf1 / psf1.sum()
    psf2 = psf2 / psf2.sum()
    
    if psf1.shape[0] > psf2.shape[0]:
        pad = (psf1.shape[0] - psf2.shape[0]) / 2
        psf1 = psf1[pad:-pad, pad:-pad]
    elif psf2.shape[0] > psf1.shape[0]:
        pad = (psf2.shape[0] - psf1.shape[0]) / 2
        psf2 = psf2[pad:-pad, pad:-pad]

    if method == 'RL':
        kernel = richardson_lucy(psf2, psf1, iterations=iterations)

    elif method == 'fft':
        kernel = create_matching_kernel(psf1, psf2, window=window)

    # normalize the kernel
    kernel = kernel / kernel.sum()
    hdr2 = fits.getheader(psf_sharp)
    if os.path.exists(kernelname):
       os.remove(kernelname)
    fits.append(kernelname, kernel, hdr2)
    return kernel


def psfmatch_RL(psf_ref, psf_input, kernelname, iterations=30):
    """
    Derive the kernel that matches psf_2match to psf_ref, so that psf_2match,
    when convolved with the kernel, gives psf_ref.
    Make sure that both PSF images have the same pixel scales, are centered,
    and have the same image size.
    USE THE RICHARDSON-LUCY DECONVOLUTION ALGORITHM.
    """
    psf1 = fits.getdata(psf_ref)
    psf2 = fits.getdata(psf_input)
    assert psf1.shape[0] % 2 == 1
    assert psf2.shape[0] % 2 == 1
    assert psf1.shape[0] == psf1.shape[1]
    assert psf2.shape[0] == psf2.shape[1]
     
    psf1 = psf1 / psf1.sum()
    psf2 = psf2 / psf2.sum()
     
    if psf1.shape[0] > psf2.shape[0]:
        pad = (psf1.shape[0] - psf2.shape[0]) / 2
        psf1 = psf1[pad:-pad, pad:-pad]
    elif psf2.shape[0] > psf1.shape[0]:
        pad = (psf2.shape[0] - psf1.shape[0]) / 2
        psf2 = psf2[pad:-pad, pad:-pad]
        
    kernel = richardson_lucy(psf1, psf2)
    # normalize the kernel
    kernel = kernel / kernel.sum()
    hdr2 = fits.getheader(psf_input)
    if os.path.exists(kernelname):
       os.remove(kernelname)
    fits.append(kernelname, kernel, hdr2)
    return kernel


def psfmatch_photutils(psf_broad, psf_sharp, kernelname, window=window_default):
    """
    Use photutils.psf.matching for PSF matching
    The default low-pass window function is CosineBellWindow with alpha=0.35
    """
    # Read the PSFs and make sure they are both normalized and have the
    # same shape
    psf1 = fits.getdata(psf_sharp)
    psf2 = fits.getdata(psf_broad)
    assert psf1.shape[0] % 2 == 1
    assert psf2.shape[0] % 2 == 1
    assert psf1.shape[0] == psf1.shape[1]
    assert psf2.shape[0] == psf2.shape[1]
     
    psf1 = psf1 / psf1.sum()
    psf2 = psf2 / psf2.sum()
     
    if psf1.shape[0] > psf2.shape[0]:
        pad = (psf1.shape[0] - psf2.shape[0]) / 2
        psf1 = psf1[pad:-pad, pad:-pad]
    elif psf2.shape[0] > psf1.shape[0]:
        pad = (psf2.shape[0] - psf1.shape[0]) / 2
        psf2 = psf2[pad:-pad, pad:-pad]

    kernel = create_matching_kernel(psf1, psf2, window=window)
    kernel = kernel / kernel.sum()
    hdr2 = fits.getheader(psf_input)
    if os.path.exists(kernelname):
       os.remove(kernelname)
    fits.append(kernelname, kernel, hdr2)
    return kernel


def rebin_psf(psfname, pixscale_new=0.06, size=61, norm=True, suffix='swp'):
    """
    Use SWarp to resample the PSF.
    """
    hdr0 = fits.getheader(psfname)
    pixscale_old = hdr0['PIXSCALE']  # in arcsec
    # Figure out the new image size if all the current data are to fit the
    # new PSF image with the specified new pixel scale
    nx_old = hdr0['naxis1']
    ny_old = hdr0['naxis2']
    nx_new = int(round(nx_old * pixscale_old / pixscale_new))
    ny_new = int(round(ny_old * pixscale_old / pixscale_new))
    print "New pixel scale would be {:.3f} arcsec".format(nx_old * pixscale_old / nx_new)
    # Now create a default SWarp file and update some parameters
    new_psfname = os.path.splitext(psfname)[0] + '_{}.fits'.format(suffix)
    os.system('swarp -d > psf.swarp')
    lines = open('psf.swarp').readlines()
    for i in range(len(lines)):
        if lines[i].startswith('IMAGEOUT_NAME'):
            l = lines[i].split()
            l[1] = new_psfname
            lines[i] = ' '.join(l) + '\n'
        if lines[i].startswith('CELESTIAL_TYPE'):
            l = lines[i].split()
            l[1] = 'PIXEL'
            lines[i] = ' '.join(l) + '\n'
        if lines[i].startswith('PROJECTION_TYPE'):
            l = lines[i].split()
            l[1] = 'NONE'
            lines[i] = ' '.join(l) + '\n'
        elif lines[i].startswith('COMBINE_TYPE'):
            l = lines[i].split()
            l[1] = 'SUM'
            lines[i] = ' '.join(l) + '\n'
        elif lines[i].startswith('PIXELSCALE_TYPE'):
            l = lines[i].split()
            l[1] = 'FIT'
            lines[i] = ' '.join(l) + '\n'
        elif lines[i].startswith('IMAGE_SIZE'):
            l = lines[i].split()
            l[1] = '{} {}'.format(nx_new, ny_new)
            lines[i] = ' '.join(l) + '\n'
        elif lines[i].startswith('SUBTRACT_BACK'):
            l = lines[i].split()
            l[1] = 'N'
            lines[i] = ' '.join(l) + '\n'
    with open('psf.swarp', 'w') as f:
        for i in range(len(lines)):
            f.write(lines[i])
    x = subprocess.call(['swarp', psfname, '-c', 'psf.swarp'])

    # Now in order for FFT deconvolution to work, we must make resampled PSF
    # images the same (square) size for everyone, given by the "size" argument
    # This means we should run SWarp again.
    hdr = fits.getheader(new_psfname)
    lines = open('psf.swarp').readlines()
    for i in range(len(lines)):
        if lines[i].startswith('IMAGE_SIZE'):
            l = lines[i].split()
            l[1] = "{} {}".format(size, size)
            lines[i] = ' '.join(l) + '\n'
        # elif lines[i].startswith('PIXEL_SCALE'):
        #     l = lines[i].split()
        #     l[1] = "{} {}".format(hdr['cd1_1'] * 3600, hdr['cd2_2'] * 3600)
        #     lines[i] = ' '.join(l) + '\n'
        # elif lines[i].startswith('PIXELSCALE_TYPE'):
        #     l = lines[i].split()
        #     l[1] = 'MANUAL'
        #     lines[i] = ' '.join(l) + '\n'
    with open('psf.swarp', 'w') as f:
        for i in range(len(lines)):
            f.write(lines[i])
    x2 = subprocess.call(['swarp', new_psfname, '-c', 'psf.swarp'])

    # Now normalize the new PSF image & apply the charge diffusion kernel
    # But we have to read the kernel first from the header
    kern = np.array([])
    for cline in hdr0['comment']:
        cl = cline.split()
        try:
            value = float(cl[0])
        except:
            continue
        kern_arr = np.array([float(x) for x in cl])
        kern = np.concatenate([kern, kern_arr])
    kern = kern.reshape(3, 3)
    h2 = fits.open(new_psfname, mode='update')
    h2[0].data = convolve2d(h2[0].data, kern, mode='same')
    if norm:
        h2[0].data = h2[0].data / h2[0].data.sum()
    h2.flush()
    h2.close()
    print "Made resampled PSF image {}".format(new_psfname)


def zoom_psf(psfname, zoom=None, cutpad=0, newpixscale=0.06, psfsize=3.9, norm=True):
   """
   Rescale ACS PSF using scipy.ndimage.interpolation.zoom to match the 
   WFC3/IR pixel scale.
   zoom < 1.0 makes PSF SHARPER.
   Assume that psf image is square.
   """
   psf = fits.getdata(psfname)
   hdr = fits.getheader(psfname)
   if zoom == None:
      oldsize = psf.shape[0]
      newsize = int(round(psfsize / newpixscale))
      zoom = float(newsize) / oldsize
   zoomed = ndimage.interpolation.zoom(psf, zoom)
   if norm:
      zoomed = zoomed / zoomed.sum()
   print zoomed.shape
   newpsf = os.path.splitext(psfname)[0] + '_zoomed.fits'
   if cutpad > 0:
      zoomed = zoomed[cutpad:-cutpad,cutpad:-cutpad]
   if os.path.exists(newpsf):
      os.remove(newpsf)
   fits.append(newpsf, zoomed, hdr)