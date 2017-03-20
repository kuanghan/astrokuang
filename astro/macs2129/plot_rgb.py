#from setup_matplotlib import *
import numpy as np
#from setup_matplotlib import *
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
rc('text.latex', preamble='\usepackage{sfmath}')
rc('text.latex', preamble='\usepackage{amsmath}')
# rc('text', usetex=True)
#####Note, ony SWARPed images work (WCS!!)
import aplpy
import sys, os

# Make 3-color image
#aplpy.make_rgb_image(['ch1.fits','ch12.fits','ch2.fits'], 'RXC0600_rgbpy.png',stretch_r='arcsinh',stretch_g='arcsinh',stretch_b='arcsinh', pmax_r=99.5, pmax_g=99, pmax_b=99.5, vmin_r=0, vmin_g=0, vmin_b=0, vmax_r= 0.5, vmax_b=0.5, vmax_g=0.5)
#sys.exit(0)

img_dir = '/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/hst'
filters = ['f625w']
ra = [322.35878]
dec = [-7.6910806]
delta1 = [180/3600.0]
delta2 = [120/3600.0]
rgb = ['macs2129_f625wf814wf160w.jpg']
for c in range(len(filters)):
    fits = os.path.join(img_dir, filters[c]+'.fits')
    # jpeg = filters[c]+'.pdf'
            
# Show the RGB image
    if (c <2): 
        fig = aplpy.FITSFigure(fits)
        fig.show_rgb(rgb[c])
    else:
        fig = aplpy.FITSFigure(fits)
        fig.show_grayscale(vmin = -0.03, vmax = -0.03, stretch='linear')
        fig.show_regions('ds9.reg')

# add contour from SWUnited
    x7=np.transpose(np.loadtxt('macs2129_zitrin_nfw_z9_v2.con'))
    # x7 = np.transpose(np.loadtxt('contour_m2129_z6.85_highres.con'))
    xx7 = x7[0, :]
    yy7 = x7[1, :]
    fig.show_markers(xx7, yy7, linewidth=0.1, marker='s', s=2, 
            facecolor='red', edgecolor='red')

    # # Also show the z=1.57 cricitcal curve
    # x1 = np.transpose(np.loadtxt('cc_1.57_highres.con'))
    # xx1 = x1[0, :]
    # yy1 = x1[1, :]
    # fig.show_markers(xx1, yy1, linewidth=0.1, marker='s', s=0.3,
    #         edgecolor='yellow', facecolor='yellow')

# add contours from Lenstool
    # fig.show_regions('cc_6.85_lenstool.reg')
    # fig.show_regions('cc_1.57_lenstool.reg')

    # fig.show_regions('magnif_austin_6.86.reg')
# Overlay a grid
#fig.add_grid()
#fig.grid.set_alpha(0.1)

# Save image
    # jpeg = filters[c]+'_'+str(c)+'.pdf'
    output = 'macs2129_rgb_zitrin.pdf'
    # output = 'macs2129_rgb_lenstool.pdf'

    fig.recenter(ra[c], dec[c], width=delta1[c], height=delta2[c])
    #fig.show_markers(ra + delta/np.cos(dec)/50, dec, layer='marker_set_1', edgecolor='red',
    #                         facecolor='none', marker=1, s=2000, alpha=1,lw=4)
    #fig.show_markers(ra, dec+delta/20, layer='marker_set_2', edgecolor='red',
    #                         facecolor='none', marker=2, s=2000, alpha=1, lw=4)
    fig.axis_labels.set_font(size='x-large', weight='medium', \
                             stretch='normal',  \
                             style='normal', variant='normal')
    #fig.add_label(ra - 0.2*delta/np.cos(dec)/2,dec+1.08*delta/2,filters[index], size=36, fontweight='bold', color='black')
    #fig.add_grid()
    fig.show_regions('macs2129_z7multi_v2.reg')
    #fig.show_regions('irfov.reg')
    #fig.grid.set_alpha(0.5)
    fig.ticks.set_color('black')
    # Save image
    fig.set_theme('publication')
    #if (c != 0):
    #    fig.axis_labels.hide()
    #    fig.tick_labels.hide()
    fig.save(output, dpi=100)
    fig.close()
        
