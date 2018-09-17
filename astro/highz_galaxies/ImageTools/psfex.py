import numpy as np
import os, glob
import subprocess

"""
Some utility functions for PSFEx.
"""

def run_sextractor(img_dir, bands, detect_band, detect_drz, detect_wht, detect_flg, sexfile):
    """
    Run SExtractor over all bands in order to detect sources that could be
    used to construct PSF. Therefore this will be a run with a high-sigma
    detection threshold b/c we only want to use bright sources for making
    PSF images.
    """
    # Make sure all strings are in lower case
    detect_band = detect_band.lower()
    bands = [b.lower() for b in bands]
    # Make sure the science, weight, and flag images in the detection band
    # actually have the detection band string in their file names

    for b in bands:
        drz_image = os.path.join(img_dir, detect_drz.replace(detect_band, b))
        wht_image = os.path.join(img_dir, detect_wht.replace(detect_band, b))
        flg_image = os.path.join(img_dir, detect_flg.replace(detect_band, b))
        seg_image = '{}_detect_seg.fits'.format(b)
        sex_cmd = ["cex", "{drz},{drz}".format(drz=drz_image), "-c", sexfile]
        sex_cmd += ["-CHECKIMAGE_NAME", seg_image]
        sex_cmd += ["-WEIGHT_IMAGE", "{wht},{wht}".format(wht=wht_image)]
        sex_cmd += ["-CATALOG_NAME", "{}_detect_psfex.cat.fits".format(b)]

        subprocess.call(sex_cmd)

    print 'Done!'


def run_psfex(catalogs, psfexfile):
    """
    Run PSFEx on a bunch of bands. Requires SExtractor catalog (in binary FITS
    format) as input.
    """
    for i in range(len(catalogs)):
        # b = bands[i]
        cat = catalogs[i]
        subprocess.call(['psfex', cat, '-c', psfexfile])

    print "Done."