import pdb
import numpy as np
import matplotlib.pyplot as plt

def select_uvj_red(c, z_bin):
    """
    Use the UVJ color selection to find red & blue galaxies.
    The criteria are taken from Whitaker et al. 2011, which saw slight
    evolution of the criteria in redshift (Equation 15), so an input redshift
    of the bin is required to determine which cut to use.
    Use the rest-frame absolute magnitudes from the CANDELS catalog for this
    selection.

    Arguments:
    -----------
    c: input dataframe containing the rest-frame photometry
    z_bin: the central redshift of the redshift bin

    Returns:
    ---------
    crit: an array of boolean values showing which ones are RED (True) galaxies
    """
    ucol = 'restujohnson'
    vcol = 'restvjohnson'
    jcol = 'restj2mass'
    umv = c[ucol] - c[vcol]
    vmj = c[vcol] - c[jcol]

    if z_bin < 0.5:
        crit1 = (umv > 0.88 * vmj + 0.69)
    else:
        crit1 = (umv > 0.88 * vmj + 0.59)

    if z_bin < 1.5:
        crit2 = umv > 1.3
        crit3 = vmj < 1.6
    elif z_bin < 2.0:
        crit2 = umv > 1.3
        crit3 = vmj < 1.5
    elif z_bin < 3.5:
        crit2 = umv > 1.2
        crit3 = vmj < 1.4

    crit = crit1 & crit2 & crit3

    return crit.values


def plot_uvj(z_bin, xmin=-1., xmax=2.5, ymin=0., ymax=3.0, ax=None, plt_kwargs={'lw': 1.5, 'c': 'black'}):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if z_bin < 0.5:
        a = 0.88
        b = 0.69
    else:
        a = 0.88
        b = 0.59

    if z_bin < 1.5:
        c = 1.3
        d = 1.6
    elif z_bin < 2.0:
        c = 1.3
        d = 1.5
    elif z_bin < 3.5:
        c = 1.2
        d = 1.4
    # calculate the first vertice
    x1 = (c - b) / a
    y1 = c
    # calculate the second vertice
    x2 = d
    y2 = a * x2 + b

    # Now plot
    # print x1, x2, y1, y2
    # print xmin, xmax, ymin, ymax
    # pdb.set_trace()
    ax.plot([xmin, x1], [y1, y1], **plt_kwargs)
    ax.plot([x1, x2], [y1, y2], **plt_kwargs)
    ax.plot([x2, x2], [y2, ymax], **plt_kwargs)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax

