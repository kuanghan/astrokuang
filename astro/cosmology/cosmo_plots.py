from astropy import cosmology
COSMO_DEF = cosmology.FlatLambdaCDM(H0=70., Om0=0.3)
import matplotlib.pyplot as plt
import numpy as np


def pixel_area_vs_z(pixscale=0.06, cosmo=COSMO_DEF):
    """
    Plot how much area is within a pixel as a function of redshift.
    Input:
    pixscale    - pixel scale in arcsec
    cosmo       - a astropy.cosmology instance
    """
    zarray = np.arange(0.2, 10., 0.2)
    # calculate the area of one pixel in steradians
    area_strad = (pixscale / 3600. * np.pi / 180.) ** 2

    area_kpc = map(lambda z: cosmo.angular_diameter_distance(z).value**2 * 1.e6 * area_strad, zarray)

    # Now make the plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(zarray, area_kpc, marker='s')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('kpc^2')
    ax.text(0.95, 0.95, 'pixscale={}"'.format(pixscale), ha='right',
            va='top', transform=ax.transAxes, size='x-large')
    ax.set_title('H0 = {}; Omega_0 = {}'.format(cosmo.H0, cosmo.Om0))

    return ax

