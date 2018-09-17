import numpy as np
import pyfits
import os
from pyraf import iraf

def wregister(input_drz, input_unc, input_cov, ref_image):
   """
   Use iraf.wregister to resample the input images onto the pixel grid defined
   by ref_image.
   """
   hdr0 = pyfits.getheader(input_drz)
   drz0 = pyfits.getdata(input_drz)
   unc = pyfits.getdata(input_unc)
   
   # prepare the input variance image
   var = np.where(unc > 0, unc**2, 0.)
   input_var = input_unc.replace('unc', 'var')
   if os.path.exists(input_var):
      os.remove(input_var)
   pyfits.append(input_var, var, hdr0)

   # now use wregister
   output_drz = input_drz.replace('drz', 'wreg_drz')
   if os.path.exists(output_drz):
      os.remove(output_drz)
   iraf.wregister(input_drz, ref_image, output_drz)
   output_var = input_var.replace('var', 'wreg_var')
   if os.path.exists(output_var):
      os.remove(output_var)
   iraf.wregister(input_var, ref_image, output_var)
   output_cov = input_cov.replace('cov', 'wreg_cov')
   if os.path.exists(output_cov):
      os.remove(output_cov)
   iraf.wregister(input_cov, ref_image, output_cov)

   # now convert the wregister-ed variance image back to uncertainty image
   var1 = pyfits.getdata(output_var)
   hdr1 = pyfits.getheader(output_var)
   unc1 = np.where(var1 > 0, np.sqrt(var1), 1.e9)
   output_unc = input_unc.replace('unc', 'wreg_unc')
   if os.path.exists(output_unc):
      os.remove(output_unc)
   pyfits.append(output_unc, unc1, hdr1)

   print 'Done.'