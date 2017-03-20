"""
Make object montage plots for high-z, IRAC-detected objects.
"""

import numpy as np
import aplpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import yaml

#----------- define Image Paths & Files ---------------
hst_imgdir = "/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/hst"
irac1_img = "/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/irac/ch1/ch1_macs2129_cut_drz.fits"
irac2_img = "/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/irac/ch2/ch2_macs2129_cut_drz.fits"
irac_images = {'ch1': irac1_img, 'ch2': irac2_img}
irac_bands = ['ch1', 'ch2']

def get_hstimg(band):
    """
    Get HST image in a given band.
    """
    band = band.lower()
    if band in ['f105w', 'f110w', 'f125w', 'f140w']:
        img = '{}_macs2129_cut_60mas_wglass_drz.fits'.format(band)
    else:
        img = '{}_macs2129_cut_60mas_blksum_drz.fits'.format(band)

    return os.path.join(hst_imgdir, img)


imageData = yaml.load(open('imageData.yml', 'rb'))
rootdir = '/Users/khuang/SurfsUp'
pRange_hst = [20., 95.]  # (pmin, pmax) for HST
pRange_irac = [20., 90.]  # (pmin, pmax) for IRAC
               

# newSubplotPars = mpl.figure.SubplotParams(wspace=0., hspace=0.1)
mpl.rcParams['figure.subplot.wspace'] = 0.

# Below defines the necessary images for the montage plot.
radec = {'Image A': [322.350936, -7.693322],
         'Image B': [322.353239, -7.697442],
         'Image C': [322.353943, -7.681646]}

objids = ['Image A', 'Image B', 'Image C']

def cutoutOne(objid, band, figure=None, subplot=(1,1,1), size=5, moffset=0.3, mlength=0.5):
    """
    Show a cutout (in 'band') centered at the object with ID objid.
    The cutout size is in arcsec.
    mlength:  the length of each marker
    moffset:  how far the edge of the vertical & horizontal offsets are from
              the center of each source.
    """
    ra, dec = radec[objid]
    degsize = size / 3600.
    # convert from arcsec to degree
    band = band.lower()
    if band not in irac_bands:
        image = get_hstimg(band)
        # image = hst_imgdir(clusterName, band)
        pmin, pmax = pRange_hst
    else:
        image = irac_images[band]
        pmin, pmax = pRange_irac  
        # this will be different for different objects
    fig = aplpy.FITSFigure(image, figure=figure, subplot=subplot)
    fig.show_grayscale(pmin=pmin, pmax=pmax, invert=True)
    fig.recenter(ra, dec, width=degsize, height=degsize)
    # Show hairline marker at the center
    # mlength = 1.0  # in arcsec
    dec0 = dec + moffset / 3600. 
    dec1 = dec0 + mlength / 3600.
    ra0 = ra + moffset / 3600.
    ra1 = ra0 + mlength / 3600.
    fig.show_lines([np.array([[ra, ra], [dec0, dec1]])], color='red', lw=4)
    fig.show_lines([np.array([[ra0, ra1], [dec, dec]])], color='red', lw=4)
    return fig


def montageOne(objid, fig=None, bands=['f435w','f850lp','f160w','ch1','ch2'], size=3, nrows=1, ncols=1, rownum=1, colnum=1, xmin=0.1, xmax=0.9, colpad=0.1, ymin=0.1, ymax=0.9, rowpad=0.05, labelsize='large', add_label=True):
    """
    Make montage for one object in the specified bands 
    (in order from left to right).
    Arguments:
    
    """
    if fig == None:
       fig = plt.figure()
    assert rownum <= nrows, "rownum (%d) should not be larger than nrows (%d)!"  % (rownum, nrows)
    assert colnum <= ncols, "colnum (%d) should not be larger than ncols (%d)!"  % (colnum, ncols)
    figures = []
    nbands = len(bands)
    # calculate dx, dy, which are the width and height of the montage of 
    # each object
    dx = (xmax - xmin - (ncols-1)*colpad) / ncols
    dy = (ymax - ymin - (nrows-1)*rowpad) / nrows
    # Also calculate the width and height of each cutout in figure coordinate
    cutdx = dx / float(nbands)
    cutdy = dy 
    for i in range(nbands):
        b = bands[i]
        sub = [xmin + (colnum-1) * (dx + colpad) + i * cutdx,
              ymax - rownum * cutdy - (rownum - 1) * rowpad,
              cutdx, cutdy]
        # f = cutoutOne(objid, b, figure=fig, subplot=(1, nbands, i+1),      size=size)
        f = cutoutOne(objid, b, figure=fig, subplot=sub, size=size)
        f.set_axis_labels(xlabel='', ylabel='')
        f.tick_labels.hide()
        if add_label:
            f.add_label(0.5, 0.1, b.upper(), relative=True, size='x-large', 
                         family='monospace', weight='bold', color='Lime')
        if i == 0:
            f.axis_labels.set_ytext(objid)
            f.axis_labels.set_font(size=labelsize, weight='bold', 
                                   family='monospace')
            f.axis_labels.show_y()
        figures.append(f)
    return figures

def montageAll(figsize=(12, 7), cutsize=5., xmin=0.1, xmax=0.9, ymin=0.1, ymax=0.9, colpad=0.01, rowpad=0.01, labelsize='xx-large', filename='montage_fig1.eps', bands=['f435w', 'f850lp', 'f160w', 'ch1', 'ch2']):
    """
    Show the RGB image of MACS2129 on top and three rows of cutouts for each
    of the three images.
    """
    fig = plt.figure(figsize=figsize)
    # First the RGB image
    # ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    # # Now have to manually crop the RGB image to have aspect ratio of 2:1
    # rgbimg = mpimg.imread('macs2129_rgb.png')
    # ny, nx, nz = rgbimg.shape
    # # xlims = [0, nx]
    # ycenter = ny / 2
    # dy = nx / 2
    # ylims = [ycenter - dy / 2, ycenter + dy / 2]
    # print "ylims =", ylims
    # rgbimg2show = rgbimg[ylims[0]:ylims[1], :, :]
    # print "rgbimg2show.shape =", rgbimg2show.shape
    # ax1.imshow(rgbimg2show[::-1,:,:])
    # ax1.set_xticks([])
    # ax1.set_yticks([])

    # Then show the cutouts
    ncols, nrows = (1, 3)
    objids = radec.keys()
    allfigs = []

    for j in range(nrows):
        for i in range(ncols):
            print "(i, j) =", (i, j)
            k = i * nrows + j
            print "*************************************"
            print "Making montage for %s..." % objids[k]
            print "*************************************"
            f = montageOne(objids[k], fig=fig, bands=bands, size=cutsize,
                nrows=nrows, ncols=ncols, rownum=j+1, colnum=i+1, 
                xmin=xmin, xmax=xmax, colpad=colpad, ymin=ymin, 
                ymax=ymax, rowpad=rowpad, add_label=False, labelsize=labelsize)
        if j == nrows - 1:
            for m in range(len(bands)):
                subfig = f[m]
                subfig.axis_labels.set_xtext(bands[m].upper())
                subfig.axis_labels.set_font(size=labelsize, weight='bold', 
                                   family='monospace')
                subfig.axis_labels.show_x()
            allfigs.append(f)

    # plt.tight_layout()
    fig.savefig(filename)
    for f in allfigs:
        for f2 in f:
            f2.close()  # Free up memory
    return fig


