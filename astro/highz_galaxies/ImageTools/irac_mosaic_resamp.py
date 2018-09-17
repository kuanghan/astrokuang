import os
import subprocess
import glob
from astropy.io import fits
import numpy as np

"""
Resample IRAC mosacis from ch1 & ch2 onto the same pixel grid. I'm writing this
right now to deal with Spitzer RELICS mosaics, which might have very different
footprints in ch1 and ch2 and thus cannot be drizzled onto the same pixel grid
by MOPEX. So I have to resample post facto.
Assumes the file structure is such that ch1 images live under the directory
ch1/pixfrac[0.1/0.85], while ch2 images live under ch2/pixfrac[0.1/0.85].
"""

def irac_mosaic_resample(ra_str, dec_str, cluster_name):
    """
    Resamples mosaics. Only works for ch1 & ch2... could expand to include
    more channels in the future.
    Also assumes the following image naming convention: all science images are
    *_drz.fits, all RMS images are *_unc.fits, and all weight images (to be
    created here) are *_wht.fits.

    Arguments:
    -----------
    ra_str, dec_str: strings that specifies the RA, DEC of the field center in
                     the format hh:mm:ss and dd:mm::ss
    cluster_name: a string for the name of the cluster

    """
    assert subprocess.call('which swarp', shell=True) == 0, "SWarp is not installed!!"
    # First pass: SWarp both channels together in order to define a common
    # pixel grid.
    # First, find the ch1/ch2 images.
    curdir = os.getcwd()
    drz_images = {}
    unc_images = {}
    wht_images = {}
    for channel in ['ch1', 'ch2']:
        os.chdir(channel)
        drz_images[channel] = glob.glob('*_drz.fits')[0]
        unc_images[channel] = drz_images[channel].replace('drz', 'unc')
        wht_images[channel] = drz_images[channel].replace('drz', 'wht')
        # Make weight images
        os.system('cp {} {}'.format(unc_images[channel], wht_images[channel]))
        h = fits.open(wht_images[channel], mode='update')
        unc = h[0].data
        wht = np.where(np.isnan(unc), 0., 1. / unc**2)
        wht = np.where(np.isnan(wht), 0., wht)
        h[0].data = wht
        h.flush()
        h.close()
        os.chdir(curdir)
        if os.path.exists(channel + '_weight.fits'):
            os.remove(channel + '_weight.fits')
        if os.path.exists(channel + '_drz.fits'):
            os.remove(channel + '_drz.fits')
        os.system('ln -s {}/{} ./{}_weight.fits'.format(channel,
            wht_images[channel], channel))
        os.system('ln -s {}/{} ./{}_drz.fits'.format(channel,
            drz_images[channel], channel))
        
    # Make a SWarp config file for the first pass
    subprocess.call('swarp -d > ch12_coadd.swarp', shell=True)
    # Update ch12_coadd.swarp
    lines = open('ch12_coadd.swarp').readlines()
    for i in range(len(lines)):
        if lines[i].startswith('IMAGEOUT_NAME'):
            lines[i] = 'IMAGEOUT_NAME    {}_ch12_coadd.fits  # Output filename\n'.format(cluster_name)
        elif lines[i].startswith('WEIGHTOUT_NAME'):
            lines[i] = 'WEIGHTOUT_NAME    {}_ch12_coadd.weight.fits   # Output weight-map filename\n'.format(cluster_name)
        elif lines[i].startswith('WEIGHT_IMAGE'):
            lines[i] = 'WEIGHT_IMAGE   ch1_weight.fits,ch2_weight.fits\n'
        elif lines[i].startswith('CENTER'):
            lines[i] = 'CENTER   {}  {}  # Coordinates of the image center\n'.format(ra_str, dec_str)
    # Now write to the SWarp config file
    with open('ch12_coadd.swarp', 'w') as f2:
        for i in range(len(lines)):
            f2.write(lines[i])
    # Run SWarp
    x = subprocess.call('swarp ch1_drz.fits,ch2_drz.fits -c ch12_coadd.swarp',
        shell=True)

    # Copy the header
    os.system('imhead {}_ch12_coadd.fits > {}_ch12_coadd.head'.format(
        cluster_name, cluster_name))
    drz_images_out = {}
    wht_images_out = {}
    for channel in drz_images.keys():
        drz_images_out[channel] = drz_images[channel].replace(
            '_drz', '_swp_drz')
        wht_images_out[channel] = wht_images[channel].replace(
            '_wht', '_swp_wht')
        os.chdir(channel)
        os.system('cp {}/{}_ch12_coadd.head ./{}.head'.format(curdir,
            cluster_name, drz_images_out[channel][:-5]))
        subprocess.call('swarp -d > {}_resamp.swarp'.format(channel),
            shell=True)
        lines = open('{}_resamp.swarp'.format(channel)).readlines()
        for i in range(len(lines)):
            if lines[i].startswith('IMAGEOUT_NAME'):
                lines[i] = 'IMAGEOUT_NAME   {}   # Output filename\n'.format(
                    drz_images_out[channel])
            elif lines[i].startswith('WEIGHTOUT_NAME'):
                lines[i] = 'WEIGHTOUT_NAME   {}   # Output weight-map filename\n'.format(wht_images_out[channel])
            elif lines[i].startswith('WEIGHT_TYPE'):
                lines[i] = 'WEIGHT_TYPE  MAP_RMS  \n'
            elif lines[i].startswith('WEIGHT_IMAGE'):
                lines[i] = 'WEIGHT_IMAGE  {}  \n'.format(unc_images[channel])
            elif lines[i].startswith('COMBINE_TYPE'):
                lines[i] = 'COMBINE_TYPE  SUM \n'
            elif lines[i].startswith('CENTER'):
                lines[i] = 'CENTER  {}  {}  # Coordinates of the image center\n'.format(ra_str, dec_str)
            elif lines[i].startswith('SUBTRACT_BACK'):
                lines[i] = 'SUBTRACT_BACK    N    # Subtraction sky background (Y/N)?\n'
        with open('{}_resamp.swarp'.format(channel), 'w') as f2:
            for i in range(len(lines)):
                f2.write(lines[i])
        x = subprocess.call('swarp {} -c {}_resamp.swarp'.format(
            drz_images[channel], channel), shell=True)
        os.chdir(curdir)

    print "Done!"
