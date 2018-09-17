#!/usr/bin/env python

# run_simulations file1.sim file2.sim ... 
# This script adds fake galaxies to real images, runs sextractor
# with an association list, and outputs the SExtractor parameters
# for the matched sources.
# 
# Configuration parameters (file.sim...this list will probably grow):
#
# SEXFILE		file.sex
# PSFFILE		""
# NITER			1
# REALIMAGE		fakez_drz.fits	
# FLAGIMAGE		fakez_flag.fits	
# SAVE			no
# NGALAXIES		100
# CIRCULAR              no
# DISKFRAC              0.5
# SCALE			0.01
# MAGLOW		20 
# MAGHIGH		28 
# MAGZPT		24.961 
# RMIN			0.01	# minimum input radius arcsec
# MAX			1.5	# maximum input radius arcsec
# RUN_SEXTRACTOR        yes     
# MEASURE_PETROSIAN     yes     
# LOGNORMAL_MAG0	24.0	# Not yet implemented
# LOGNORMAL_PEAK	0.5
# LOGNORMAL_WIDTH	0.5
# 
#

# This works for GALFIT simulations. Without the GALFIT part, it can also be
# used for SExtractor simulations, but in that case one needs to specify 
# the SED of each fake galaxy.

# mkobjects creates noiseless devauc-profile galaxies that are 
# 0.13 mag too faint when dynrange=1.e5
devauc_correction = -0.00  


# from numpy import *
import numpy as np
import os, sys, glob, string, shutil, time
from astropy.io import fits
from pyraf import iraf
from iraf import artdata, images, stsdas
curdir = os.getcwd()
import sextractor_sim
from colorama import Fore, Style
import yaml

datum = time.strftime("%m%d",time.localtime())

def run_sexsim(parfile):
   sim = sextractor_sim.sextractor_sim(parfile)
   params = yaml.load(open(parfile))
   assert 'DIFFIMAGE' in params.keys(), "Make sure to include DIFFIMAGE (the difference between original and ICL-corrected images) in the parameter file!"
   while sim.n < sim.nstop:
      print "-" * 60
      print(Style.BRIGHT + Fore.GREEN + "Iteration {}:".format(sim.n) + Style.RESET_ALL)
      # print "Iteration %d:" % sim.n
      print "-" * 60
      sim.insert_fake_sources()
      # Add the DIFFIMAGES
      # DIFFIMAGES is defined such that input_image + diff_image = icl_corr_image
      detect_image = '{}_detect.fits'.format(os.path.splitext(parfile)[0])
      if os.path.exists(detect_image):
         os.remove(detect_image)
      for i in range(len(sim.bands)):
         b = sim.bands[i]
         if b == sim.detect_band:
            # Make a detection image using the ICL-subtracted image
            fake0 = fits.getdata(sim.fakeimages[b])
            diff = fits.getdata(params['DIFFIMAGE'][i])
            fake0_detect = fake0 + diff
            hdr = fits.getheader(sim.fakeimages[b])
            fits.append(detect_image, fake0_detect, hdr)
            sim.fake_detect_image = detect_image
            if params['SUBTRACT_ICL']:
               # if also measure fluxes on the ICL-subtracted image...
               h = fits.open(sim.fakeimages[b], mode='update')
               h[0].data = h[0].data + diff
               h.flush()
               h.close()
         else:
            # Only update the fake images if params['SUBTRACT_ICL'] == True
            if params['SUBTRACT_ICL']:
               h = fits.open(sim.fakeimages[b], mode='update')
               diff = fits.getdata(params['DIFFIMAGE'][i])
               h[0].data = h[0].data + diff
               h.flush()
               h.close()
      sim.run_sextractor()
      if not sim.save:
         sim.cleanup()
      sim.n = sim.n + 1

   return sim

if __name__ == '__main__':
   # Read the configuration file and construct arguments for 
   #	simulate() and measure_petrosian()
 
   t1 = time.time()
   curdir = os.getcwd()
   
   parfile = sys.argv[1]
   run_sexsim(parfile)
   
   print "Finished simulation"
