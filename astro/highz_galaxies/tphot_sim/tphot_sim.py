import sys, site
site.addsitedir('/Users/khuang/Dropbox/codes/tphot/lib')
import numpy as np
import fake_galaxies_multires as fgm
from sexsim import sextractor_sim as ss
from sexsim import fake_galaxies
import glob, os
from pygoods import sextractor, Ftable
import yaml
import subprocess
from surfsup import TPHOTpipeline
from pygoods import angsep
from scipy.stats import sigmaclip
from stats import robust, gauss
import pyfits
from PhotomTools import photom_utils as pu
import hconvolve
from pyraf import iraf
iraf.artdata()

match_rad = 3.0   # positional match radius (in pixels) for artificial galaxies


def sn2magerr(sn):
   if isinstance(sn, (int, float)):
      if sn > 0:
         return 2.5 * np.log10(1. + 1. / sn)
      else:
         return -1.0
   else:
      sn = np.array(sn)
      return np.where(sn > 0, 2.5 * np.log10(1. + 1. / sn), -1.0)


class TPHOTSim(ss.SExtractorSim):
   def __init__(self, parfile):
      ss.SExtractorSim.__init__(self, parfile)
      # The first band is the high-res band, and the second band is the 
      # low-res band, and all attributes store the high-res version as the 
      # first element and the low-res attribute as the second element.

      # Only run N galaxies, where N is the number of targets
      self.ngal = len(self.c['TARGETS'])
      self.targetNames = self.c['TARGETS'].keys()
      self.ra_targets = np.array(
         [self.c['TARGETS'][k][0] for k in self.targetNames])
      self.dec_targets = np.array(
         [self.c['TARGETS'][k][1] for k in self.targetNames])

      # whether to insert fake sources into the low-res image or not
      self.insert_lores = self.c['INSERT_LORES']
      self.lores_mags_iter = {}
      if self.insert_lores:
         self.lores_mags = np.array(
            [self.c['TARGETS'][k][2] for k in self.targetNames])
      else:
         self.lores_mags = np.ones(len(self.c['TARGETS'])) * 99.0
         
      self.target_radius = self.c['TARGET_RADIUS']
      # minimum distance from the simulated source to the target
      if 'INNER_RADIUS' in self.c.keys():
         self.inner_radius = self.c['INNER_RADIUS']
      else:
         self.inner_radius = 1.0

      self.tphotparam = self.c['TPHOTPARFILE']

      print "*****************************************"
      print " NUMBER OF SOURCES: %d" % self.ngal
      print "*****************************************"
      
      fg_args = [self.realimages, self.flagimages, self.bands]
      fg_kwargs = dict(ngal=self.ngal, diskfrac=self.diskfrac, 
                  magfile=self.magfile, rdist=self.rdistfunc, mag0=self.maglow,
                  mag1=self.maghigh, logr0=self.logrmin, logr1=self.logrmax, 
                  lognormal_beta=self.lognormal_beta, 
                  lognormal_mag0=self.lognormal_mag0, 
                  lognormal_peak=self.lognormal_peak,
                  lognormal_sigma=self.lognormal_sigma, 
                  flagmax=self.flagmax,
                  othercols_list=self.othercolnames)
      self.fg = fgm.FakeGalaxiesMultiRes(*fg_args, **fg_kwargs)
      
      ra, dec = self.fg.get_radec_around(self.ra_targets, self.dec_targets,
                                         radius=self.target_radius)
      x, y = self.fg.get_xy(RA=ra, DEC=dec, mode='hires')
      self.fg.x = x
      self.fg.y = y
      
      # T-PHOT threshold
      if 'THRESHOLD' not in self.c.keys():
         self.threshold = 0.0
      else:
         self.threshold = self.c['THRESHOLD']

      self.hires_band = self.bands[0]
      self.lores_band = self.bands[1]
      self.pixscale_hires = self.c['SCALE'][0]
      self.pixscale_lores = self.c['SCALE'][1]

   def insert_fake_sources(self, iter, save=False):
      """
      Generate attributes for artificial galaxies, and then insert artificial 
      galaxies into real images.
      """
      # raise NotImplementedError
      if glob.glob('*.list'):
         os.system('rm *.list')
      self.fg.spawn_galaxies()
      if self.c.has_key('TARGETS'):
         ra, dec = self.fg.get_radec_around(self.ra_targets, self.dec_targets,
                                            radius=self.target_radius,
                                            inner_radius=self.inner_radius)
         x, y = self.fg.get_xy(RA=ra, DEC=dec, mode='hires')
         self.fg.x = x
         self.fg.y = y
      
      if self.insert_lores:
         # Also get the fake sources' (x, y) coordinates in the low-res image
         self.fg.x_lores, self.fg.y_lores = self.fg.get_xy(
            RA=self.fg.ra, DEC=self.fg.dec, mode='lores'
            )

      # First, add galaxies to the hi-res image
      self.fg.makegals_hires(self.flagimages[self.hires_band])  
      # write input file for mkobjects
      self.makenoiselessimage(self.hires_band, 
                              self.fg.artfiles[self.hires_band], 
                              self.zeropoints[self.hires_band],
                              self.psffiles[self.hires_band])
      self.addsimulated(self.hires_band, save=save)
      new_fake_image_hr = os.path.splitext(self.fakeimages[self.hires_band])[0] + '_drz.fits'
      os.rename(self.fakeimages[self.hires_band], new_fake_image_hr)
      self.fakeimages[self.hires_band] = new_fake_image_hr
      # designate a detection image
      self.fake_detect_image = self.fakeimages[self.hires_band]

      # set some default value for input lo-res magnitude
      self.lores_mags_iter[iter] = self.lores_mags
      # Now insert galaxies into the low-res image
      if self.insert_lores:
         # Assign IRAC magnitudes of the fake sources, if we decide to insert
         # fake sources into the low-res images
         if 'DMAG_LORES' in self.c.keys():
            print "*" * 40
            print "Perturbing input Lo-res magnitudes..."
            print "*" * 40
            dmag = self.c['DMAG_LORES']
            lores_mags_low = self.lores_mags - dmag
            lores_mags_high = self.lores_mags + dmag
            self.lores_mags_iter[iter] = np.random.uniform(
               list(lores_mags_low), list(lores_mags_high))
         else:
            self.lores_mags_iter[iter] = self.lores_mags
         self.fg.mag[self.c['BANDS'][1]] = self.lores_mags_iter[iter]

         # write the iraf.mkobjects galfile
         self.fg.makegals_lores()
         # self.lores_magin = dict(
         #    zip(self.targetNames, self.fg.mag[self.lores_band]))

         # Calls IRAF.mkobjects directly?
         iraf.unlearn('artdata')
         iraf.unlearn('mkobjects')
         iraf.artdata.dynrange = 1.e5
         print "Running iraf.mkobject for %s..." % self.lores_band
         self.fakeimages[self.lores_band] = '{}_sim.fits'.format(
            self.lores_band)
         
         xmax_lores, ymax_lores = pyfits.getdata(
            self.realimages[self.lores_band]).shape

         # Run IRAF.MKOBJECTS command... note that we first insert (nearly)
         # point sources, and then convolve the fake image with the low-res
         # PSF.
         # iraf.mkobjects(self.realimages[self.lores_band], 
         #                output=self.fakeimages[self.lores_band],
         #                star="gaussian", radius=0.1,
         #                title="", ncols=xmax_lores, nlines=ymax_lores,
         #                header="", background=0.,
         #                objects=self.fg.artfiles[self.lores_band],
         #                xoffset=0., yoffset=0., exptime=1.,
         #                magzero=self.c['MAGZPT'][1],
         #                gain=self.c['GAIN'][1],
         #                rdnoise=0, poisson=0)
         # print "Convolving with PSF..."
         # hconvolve.imhconvolve(outfile_nonoise, self.psffiles[self.lores_band],
         #                       self.fakeimages[self.lores_band], overwrite=True)

         self.makenoiselessimage(self.lores_band,
                                 self.fg.artfiles[self.lores_band],
                                 self.zeropoints[self.lores_band],
                                 self.c['PSFFILE_LORES'],
                                 # self.psffiles[self.lores_band],
                                 xmax=xmax_lores, ymax=ymax_lores)
         self.addsimulated(self.lores_band, save=save)
      else:
         self.fakeimages[self.lores_band] = self.realimages[self.lores_band]
         # self.lores_magin = dict(
         #  zip(self.targetNames, [99.0] * len(self.targetNames)))

   def run_sextractor(self, sex_exe='cex'):
      """
      Run SExtractor in the high-res band.
      Should I also consider running SExtractor in the low-res band?
      """
      assert hasattr(self, 'fg'), "Please generate fake sources first."

      broot = self.root + '_' + self.hires_band
      self.catalogs[self.hires_band] = "%s_%d.cat" % (broot, self.n)
      self.segimages[self.hires_band] = '%s_segnew.fits' % broot         
      self.newcatalogs[self.hires_band] = "%s_%d.newcat" % (broot, self.n)
      n_args = "%s,%s -c %s" % (
         self.fake_detect_image, self.fakeimages[self.hires_band], self.sexfile)
      n_args = n_args + " -CATALOG_NAME %s" % (
         self.newcatalogs[self.hires_band])
      n_args = n_args + " -MAG_ZEROPOINT %9.4f" % (
         self.zeropoints[self.hires_band])
      n_args = n_args + " -GAIN %12.4f" % (self.gains[self.hires_band])
      n_args = n_args + " -FLAG_IMAGE %s" % (self.flagimages[self.hires_band])
      n_args = n_args + " -PARAMETERS_NAME %s" % (self.sextparfile)
      n_args = n_args + " -CHECKIMAGE_TYPE SEGMENTATION"
      n_args = n_args + " -CHECKIMAGE_NAME %s" % (
         self.segimages[self.hires_band])
      n_args = n_args + " -WEIGHT_TYPE %s,%s" % (self.wht_type, self.wht_type)
      if hasattr(self.c, 'WEIGHT_THRESH'):
         wt = self.c['WEIGHT_THRESH']
         n_args = n_args + " -WEIGHT_THRESH %f,%f" % (wt, wt)
      n_args = n_args + " -WEIGHT_IMAGE %s,%s" % (
         self.rmsimages[self.detect_band], self.rmsimages[self.hires_band])
      print "%s %s" % (sex_exe, n_args)
      sys.stdout.flush()
      fpipe = os.popen("%s %s" % (sex_exe, n_args))
      fpipe.close()
      # Identify fake galaxies
      cnew = sextractor(self.newcatalogs[self.hires_band])
      f = open(self.catalogs[self.hires_band], 'w')
      f.write(cnew._header)
      ncolumns = len(cnew._colnames)
      default_columns = ['X_IN', 'Y_IN', 'MAG_IN', 'RE_IN', 
                        'GTYPE_IN [devauc=%d, expdisk=%d' % (fake_galaxies.devauc, fake_galaxies.disk), 
                        'AXIS_RATIO_IN', 'PA_IN', 'ID_THISRUN']
      for j in range(len(default_columns)):
         ncolumns += 1
         f.write('# %d %s\n' % (ncolumns, default_columns[j]))
      for j in range(len(self.fg.othercolnames)):
         ncolumns += 1
         f.write('# %d %s\n' % ((ncolumns), self.fg.othercolnames[j].upper()))
      if self.magfile:
         f.write('# %d IGAL\n' % (ncolumns+1))
      n_fake_gals = 0
      for i in range(self.ngal):
         dist = np.sqrt((self.fg.x[i]-cnew.x_image)**2 + (self.fg.y[i]-cnew.y_image)**2)
         if dist.min() > match_rad:
            continue
         j = np.argsort(dist)[0]  # index in cnew that matches this fake galaxy
         f.write(' '.join(cnew._colentries[j]))  # write the entry from cnew
         f.write(' %.2f %.2f ' % (self.fg.x[i], self.fg.y[i]))
         f.write(' %.2f %.2f ' % (self.fg.mag[self.hires_band][i], self.fg.re[i]))
         f.write(' %4d  %.2f ' % (self.fg.gtype[i], self.fg.axis_ratio[i]))
         f.write(' %.2f  %4d ' % (self.fg.position_angle[i], i))
         for k in range(len(self.fg.othercolnames)):
            f.write(' %s ' % str(self.fg.othercols[self.fg.othercolnames[k]][i]))
         if self.magfile:
            f.write(' %s ' % str(self.fg.igals[i]))
         f.write('\n')
         n_fake_gals += 1
         self.fg.detected[i] = 1

      # Append non-detected galaxies in the end; substitute all values by -1
      for i in range(self.ngal):
         if self.fg.detected[i] == 0:
            f.write(' '.join([repr(-1)]*len(cnew._colnames)))
            f.write(' %.2f %.2f ' % (self.fg.x[i], self.fg.y[i]))
            f.write(' %.2f %.2f ' % (self.fg.mag[self.hires_band][i], self.fg.re[i]))
            f.write(' %4d  %.2f ' % (self.fg.gtype[i], self.fg.axis_ratio[i]))
            f.write(' %.2f  %4d ' % (self.fg.position_angle[i], i))
            for k in range(len(self.fg.othercolnames)):
               f.write(' %s ' % str(self.fg.othercols[self.fg.othercolnames[k]][i]))
            if self.magfile:
               f.write(' %s ' % str(self.fg.igals[i]))
            f.write('\n')
      f.close()
      print "%d fake galaxies identified." % n_fake_gals
      if n_fake_gals == 0:
         return 0
      os.system("mv %s %s/run%d.cat" % (self.catalogs[self.hires_band], broot, self.n))
      os.system("mv %s %s/run%d.newcat" % (self.newcatalogs[self.hires_band], broot, self.n)) 
      os.system("mv %s %s/glart%d.list" % (self.fg.artfiles[self.hires_band], broot, self.n))                
      os.system("cp %s %s" % (self.psffiles[self.hires_band], broot))

      self._finished_sextractor = True
      
      return 1

   def write_tphot_param(self, tphotparfile):
      """
      Writes the T-PHOT parameter file.
      """
      # Only works when self.insert_lores == False
      # assert self.insert_lores == False   # should remove this later!!

      broot_hr = self.root + '_' + self.hires_band
      # Pull TPHOT parameters from self.tphotparam?
      tpar = yaml.load(open(self.tphotparam, 'rb'))
      self.tpar = tpar
      f = open(tphotparfile, 'wb')
      f.write('fitdir:     %s\n' % os.getcwd())
      f.write('hires_dir:   %s\n' % tpar['hires_dir'])
      # Also create a symlink within hires_dir that points to the mock image
      curdir = os.getcwd()
      os.chdir(tpar['hires_dir'])
      try:
         os.symlink('%s/%s' % (curdir, self.fakeimages[self.hires_band]), 
                    self.fakeimages[self.hires_band])
      except:
         pass
      try:
         os.symlink('%s/%s' % (curdir, self.segimages[self.hires_band]), 
                    self.segimages[self.hires_band])
      except:
         pass
      if not os.path.exists('%s_%s' % (self.root, self.hires_band)):
         os.symlink('%s/%s_%s' % (curdir, self.root, self.hires_band), 
                    './%s_%s' % (self.root, self.hires_band))
      os.chdir(curdir)
      f.write('lores_dir:   {}\n'.format(curdir))
      # f.write('lores_dir:   %s \n' % tpar['lores_dir'])      
      f.write('fit_boxsize:   %.2f \n' % tpar['fit_boxsize'])
      f.write('bkgd_boxsize:  %.2f \n' % tpar['bkgd_boxsize'])
      f.write('growsig:       %.2f \n' % tpar['growsig'])
      f.write('hires_scale:     %.3f \n' % self.pixscale_hires)
      f.write('lores_scale:     %.3f \n' % self.pixscale_lores)
      f.write('fitgauss:        %s \n' % str(tpar['fitgauss']))
      f.write('add_bkgdrms:     %s \n' % str(tpar['add_bkgdrms']))
      f.write('hires_drz:       %s \n' % self.fake_detect_image)
      f.write('hires_seg:       %s \n' % self.segimages[self.hires_band])
      f.write('hires_flag:      %s \n' % self.flagimages[self.hires_band])
      f.write('hires_cat:       %s/run%d.newcat \n' % (broot_hr, self.n))
      f.write('hires_cutoutcat:    run%d_cutout.cat \n' % self.n)
      f.write('hr_fitcat:      %s_run%d_tfit.cat \n' % (
              self.hires_band.lower(), self.n))
      f.write('hires_band:      %s\n' % self.hires_band.lower())

      f.write('lores_drz:       %s\n' % self.fakeimages[self.lores_band])
      f.write('lores_seg:       %s\n' % tpar['lores_seg'])  

      f.write('lores_err:       %s\n' % self.rmsimages[self.lores_band])
      f.write('lores_flg:       %s\n' % self.flagimages[self.lores_band])
      f.write('psffile:         %s\n' % self.psffiles[self.lores_band])
      f.write('lores_band:      %s\n' % self.lores_band.lower())
      f.write('fitpars1:        %s_%s_pass1.param\n' % (
              self.lores_band.lower(), self.root))
      f.write('fitpars2:        %s_%s_pass2.param\n' % (
              self.lores_band.lower(), self.root))
      f.write('cutoutdir:       allcut  \n')
      f.write('fitcat:          %s_%s_tphot_pass2.cat\n' % (
              self.lores_band.lower(), self.root))
      f.write('cluster_name:    %s\n' % self.c['CLUSTER_NAME'])
      f.write('chi2box:         %.1f\n' % tpar['chi2box'])
      f.write('skyrms:          %.4e\n' % tpar['skyrms'])
      f.write('excl_target:     no\n')
      f.write('threshold:       {}\n'.format(self.threshold))
      # Now figure out which ones are the fake galaxies to run
      c_hr = sextractor('%s_%s/run%d.cat' % (self.root, self.hires_band, self.n))
      # Write target infor
      f.write('targets:\n')
      self.target_sexid = {}
      for i in range(len(c_hr)):
         objInfo = [c_hr.alpha_j2000[i], c_hr.delta_j2000[i]]
         if c_hr.x_image[i] > 0:
            x_in = c_hr.x_in[i]
            y_in = c_hr.y_in[i]
            # Use the image pixel coordinates to figure out which detected fake
            # source is which input source
            dist2 = (self.fg.x - x_in)**2 + (self.fg.y - y_in)**2
            j = np.argmin(dist2)
            target_name = self.targetNames[j]
            self.target_sexid[c_hr.number[i]] = target_name
            sim_objid = self.n * self.c['ID_OFFSET'] + c_hr.number[i]
            objName = "%s_%d" % (target_name, sim_objid)
            f.write("   %s: %s\n" % (objName, str(objInfo)))
      # write RA
      # f.write('ra:\n')
      # for i in range(len(c_hr)):
      #    if c_hr.x_image[i] > 0:
      #       f.write('   - %.8f \n' % c_hr.alpha_j2000[i])
      # f.write('dec:\n')
      # for i in range(len(c_hr)):
      #    if c_hr.x_image[i] > 0:
      #       f.write('   - %.8f \n' % c_hr.delta_j2000[i])
      # f.write('objnames:\n') 
      # for i in range(len(c_hr)):
      #    if c_hr.x_image[i] > 0:
      #       sim_objid = self.n * self.c['ID_OFFSET'] + c_hr.number[i]
      #       f.write('   - "%s_%d"\n' % (self.root, sim_objid))
      # f.write('objectid:\n')
      # for i in range(len(c_hr)):
      #    if c_hr.x_image[i] > 0:
      #       f.write('   - %d\n' % c_hr.number[i])
      f.close()

   def run_tphot(self, tphotparfile):
      tpipe = TPHOTpipeline.TPHOTpipeline(tphotparfile)
      tpipe.run_all_objects()
      if not self.save:
         # clean up the auxilliary files
         # tpipe.clear_aux()
         hires_tfitcat = "%s_run*_tfit.cat" % (self.root)
         os.system("rm %s" % hires_tfitcat)
      return tpipe

   def cleanup(self):
      os.system('rm -r cutout')
      os.system('rm -r {}*.fits'.format(self.root))
      os.system('rm -r *bkgd*.fits')
      os.system('rm -r *bgcorr*.fits')
      os.system('rm -r {}*.fits'.format(self.lores_band))
      for target_name in self.targetNames:
         os.system('rm -r {}_*'.format(target_name))

   # def clear_aux(self, tphotparfile):
   #    tpipe = TPHOTpipeline.TPHOTpipeline(tphotparfile)
   #    tpipe.clear_aux()

   def collect_results(self, tpipe, iter):
      """
      Collect the magnitudes from sim sources for a single
      """
      print "*********************************"
      print "Collecting Results..." 
      print "*********************************"
      c_hr = sextractor('%s_%s/run%d.cat' % (self.root, self.hires_band, self.n))
      ngals = len(c_hr)
      # below are the columns from TPHOT catalog
      x_lr = np.zeros(ngals)
      y_lr = np.zeros(ngals)
      Cell = np.zeros(ngals, 'int')
      cx = np.zeros(ngals)
      cy = np.zeros(ngals)
      Rcell = np.zeros(ngals)
      fitqty = np.zeros(ngals)
      fitquerr = np.zeros(ngals)
      sexf = np.zeros(ngals)
      totflux = np.zeros(ngals)
      numfits = np.zeros(ngals, 'int')
      maxflag = np.zeros(ngals, 'int')
      sn = np.zeros(ngals)
      maxcvid = np.zeros(ngals, 'int')
      maxcvratio = np.zeros(ngals)
      mag_lr = np.zeros(ngals)
      magerr_lr = np.zeros(ngals)

      # tpars = yaml.load(open(self.tphotparam))
      
      for i in range(len(c_hr)):
         if c_hr.x_image[i] > 0:
            sim_objid = self.n * self.c['ID_OFFSET'] + c_hr.number[i]
            # print "self.n, ID_OFFSET, numbers[i]", self.n, self.c['ID_OFFSET'], c_hr.number[i]
            # print "sim_objid: %d" % sim_objid
            objname = '%s_%s' % (self.root, sim_objid)
            # print "objname: %s" % objname
            tphot_cat = 'lores_tphot.cat_pass2_best_mag'
            # tphot_cat = "%s_%s_tphot_pass2_%s.cat_best_mag" % (self.lores_band.lower(), self.c['CLUSTER_NAME'], objname)
            target_name = self.target_sexid[c_hr.number[i]]
            c_lr = sextractor(
               "%s_%d/" % (target_name, sim_objid) + tphot_cat
               )
            j = np.arange(len(c_lr))[c_lr.objectid==c_hr.number[i]][0]
            x_lr[i] = c_lr.x[j]
            y_lr[i] = c_lr.y[j]
            Cell[i] = c_lr.cell[j]
            cx[i] = c_lr.cx[j]
            cy[i] = c_lr.cy[j]
            Rcell[i] = c_lr.rcell[j]
            fitqty[i] = c_lr.fitqty[j]
            fitquerr[i] = c_lr.fitquerr[j]
            sexf[i] = c_lr.sexf[j]
            totflux[i] = c_lr.totflux[j]
            numfits[i] = c_lr.numfits[j]
            maxflag[i] = c_lr.maxflag[j]
            # sn[i] = c_lr.sn[j]
            maxcvid[i] = c_lr.maxcvid[j]
            maxcvratio[i] = c_lr.maxcvratio[j]
            mag_lr[i] = c_lr.mag[j]
            magerr_lr[i] = c_lr.magerr[j]
      # Now write to an output catalog
      f = open('%s_%s/run%d_comb.cat' % (
               self.root, self.hires_band, self.n), 'w')
      f.write(c_hr._header)
      f.write('# %d X_LR\n' % (c_hr._ncolumns + 1))
      f.write('# %d Y_LR\n' % (c_hr._ncolumns + 2))
      f.write('# %d Cell\n' % (c_hr._ncolumns + 3))
      f.write('# %d CX\n' % (c_hr._ncolumns + 4))
      f.write('# %d CY\n' % (c_hr._ncolumns + 5))
      f.write('# %d RCELL\n' % (c_hr._ncolumns + 6))
      f.write('# %d FITQTY\n' % (c_hr._ncolumns + 7))
      f.write('# %d FITQUERR\n' % (c_hr._ncolumns + 8))
      f.write('# %d SEXF \n' % (c_hr._ncolumns + 9))
      f.write('# %d TOTFLUX\n' % (c_hr._ncolumns + 10))
      f.write('# %d NUMFITS\n' % (c_hr._ncolumns + 11))
      f.write('# %d MAXFLAG\n' % (c_hr._ncolumns + 12))
      f.write('# %d SN\n' % (c_hr._ncolumns + 13))
      f.write('# %d MAXCVID\n' % (c_hr._ncolumns + 14)) 
      f.write('# %d MAXCVRATIO \n' % (c_hr._ncolumns + 15))
      f.write('# %d MAG_LR_IN\n' % (c_hr._ncolumns + 16))
      f.write('# %d MAG_LR\n' % (c_hr._ncolumns + 17))
      f.write('# %d MAGERR_LR\n' % (c_hr._ncolumns + 18))
      for i in range(len(c_hr)):
         if c_hr.x_image[i] > 0:
            target_name = self.target_sexid[c_hr.number[i]]
            j = self.targetNames.index(target_name)
            f.write(' '.join(c_hr._colentries[i]))
            f.write(' ')
            f.write('%.2f %.2f ' % (x_lr[i], y_lr[i]))
            f.write('%d %.2f %.2f ' % (Cell[i], cx[i], cy[i]))
            f.write('%.3f %f %f ' % (Rcell[i], fitqty[i], fitquerr[i]))
            f.write('%f %f %d ' % (sexf[i], totflux[i], numfits[i]))
            f.write('%d %f %d ' % (maxflag[i], sn[i], maxcvid[i]))
            f.write('%f %f ' % (maxcvratio[i], self.lores_mags_iter[iter][j]))
            f.write('%f %f ' % (mag_lr[i], magerr_lr[i]))
            f.write('\n')
      f.close()

   def get_radec(self, objname):
      """
      Returns RA, DEC for a target, matching with either the object name 
      (a string) or object ID (an integer)
      """
      # if isinstance(objname, str):
      assert (objname in self.targetNames), "Object {} not in target list!!".format(objname) 

      i = self.targetNames.index(objname)

         # try:
         #    # i = self.c['OBJNAMES'].index(objname)
         #    i = self.targetNames.index(objname)
         # except IndexError:
         #    print "Object name %s has no match." % objname
         #    return None
      # else:
      #    try:
      #       # i = self.c['OBJECTID'].index(objname)
      #       i = self.id_targets.item(objname)
      #    except IndexError:
      #       print "Object ID %d has no match." % objname
      #       return None
      ra = self.ra_targets[i]
      dec = self.dec_targets[i]
      return (ra, dec)

   def collect_results_all_iter(self, outputfile):
      """
      Merge simulation results from all iterations.
      """
      curdir = os.getcwd()
      os.chdir('%s_%s' % (self.root, self.hires_band))
      cats = glob.glob('run*_comb.cat')
      first = 1
      result_dic = {}
      nobj = 0
      if not len(cats):
         print "No catalogs found."
         return 0
      for cat in cats:
         c = sextractor(cat)
         nobj += len(c)
         if first:
            first = 0
            for d in c._colnames:
               result_dic[d] = c.__getattribute__(d)
         else:
            for d in c._colnames:
               result_dic[d] = np.concatenate([result_dic[d], c.__getattribute__(d)])
      # Write output
      os.chdir(curdir)
      f = open(outputfile, 'wb')
      f.write(c._header)
      for j in range(nobj):
         for i in range(len(c._colnames)):
            f.write('%s ' % str(result_dic[c._colnames[i]][j]))
         f.write('\n')
      f.close()
      # os.chdir(curdir)


class TPHOTSimResults(object):
   """
   Analyze TPHOT simulation results.
   """
   def __init__(self, tphotsim_catalog, tphotsim_paramfile, ra_col='alpha_j2000', dec_col='delta_j2000', magzero=21.581):
      self.c = sextractor(tphotsim_catalog)
      self.params = yaml.load(open(tphotsim_paramfile))
      self.ra_sim = self.c.__getattribute__(ra_col)
      self.dec_sim = self.c.__getattribute__(dec_col)
      self.magzero = magzero
      self.targets = self.params['TARGETS'].keys()

   def select_sim_sources(self, target_name, radius=2.0, mag_tphot=-99.0):
      """
      Returns a filter that selects the simulated sources for a given target.
      """
      ra, dec = self.params['TARGETS'][target_name][:2]
      angdist = angsep.angsep(ra, dec, self.ra_sim, self.dec_sim)
      sim_filter = (angdist <= (radius / 3600.))

      return sim_filter


   def calc_nsigma_limits(self, target_name, nsigma=3.0, radius=5.0, flux_col='fitqty', print_it=False, full_output=False):
      """
      Calculate the n-sigma magnitude limit within a radius (in arcsec) around
      the supplied sky coordinate (ra, dec). n defaults to 3.
      The simulations are supposed to measure fluxes within the low-res image
      where we did NOT put fake sources... so the spread of the measured flux
      should center around zero and has a dispersion equal to the 1-sigma 
      limiting magnitude. At least in theory...
      """
      sim_filter = self.select_sim_sources(target_name, radius=radius)
      # fluxerr = self.c.__getattribute__(fluxerr_col)[pick]
      flux = self.c.__getattribute__(flux_col)[sim_filter]
      ### === Use median and MADN to estimate n-sigma flux limit === ###
      median_flux = np.maximum(0., np.median(flux))
      # nsig_fluxerr = np.std(flux) * nsigma  
      # Use MADN for a more robust estimate...
      nsig_fluxerr = robust.MADN(flux) * nsigma
      nsig_flux = median_flux + nsig_fluxerr
      # print "Median flux: ", median_flux
      # print "MADN flux:", nsig_fluxerr / nsigma
      ### ========================================================== ###
      # other ways to estimate disperson? Fit a Gaussian curve?
      ### === Fit Gaussian to the histogram === ###
      # meanflux, sigmaflux = gauss.fitgauss(flux)
      # print "Best-fit Gaussian: mean=%.3e, sigma=%.3e" % (meanflux, sigmaflux)
      # nsig_flux = meanflux + nsigma * sigmaflux
      ### ===================================== ###
      # print "nsig_flux:", nsig_flux
      nsig_mag = self.magzero - 2.5 * np.log10(nsig_flux)
      if print_it:
         print "%.1f-sigma magnitude error from %d simulated sources: %.3f" % (nsigma, pick.sum(), nsig_mag)
      if full_output:
         return nsig_mag, pick.sum(), flux
      else:
         return nsig_mag

   # def calc_mag_err(self, target_name, radius=2., dmag=-1, dmag_in=-1):
   def calc_mag_err(self, target_name, radius=5., fix_input=False, p=0.68):
      """
      Calculate the median or average magnitude error by comparing the input
      and output magnitudes in the low-res image.
      Collect all the output flux and output flux error, calculate
      the median or average S/N, and from that calculate the magnitude error.
      If dmag > 0, then only collect the simulated results where the MEASURED
      SIMULATED magnitudes are within +/- dmag from the nominal T-PHOT
      measured value.
      """
      magzpt_hr, magzpt_lr = self.params['MAGZPT']

      sim_filter_obj = self.select_sim_sources(target_name, radius=radius)
      sim_filter_obj = np.logical_and(sim_filter_obj, self.c.maxcvratio < 1.0)
      sim_filter_obj = np.logical_and(sim_filter_obj, self.c.mag_lr < 90.0)

      if len(self.params['TARGETS'][target_name]) > 2:
         mag_tphot = self.params['TARGETS'][target_name][2]
         print "T-PHOT measured magnitude for {} is {:.2f}".format(target_name,
                                                                   mag_tphot)
         # maxcvratio = self.params['TARGETS'][target_name][3]
         if fix_input:
            sim_filter_obj = np.logical_and(sim_filter_obj,
               np.abs(self.c.mag_lr_in - mag_tphot) < 0.01)
      else:
         mag_tphot = -99.0

      # Also does sigma-clipping
      mag_in_obj = self.c.mag_lr_in[sim_filter_obj]
      mag_out_obj = self.c.mag_lr[sim_filter_obj]
      mag_hr_in_obj = self.c.mag_in[sim_filter_obj]
      mag_hr_out_obj = self.c.mag_iso[sim_filter_obj]
      
      # calculate color & flux ratios, and average in the flux ratio space
      color_in_obj = mag_hr_in_obj - mag_in_obj
      color_out_obj = mag_hr_out_obj - mag_out_obj
      fluxratio_in_obj = 10. ** (-0.4 * color_in_obj)
      fluxratio_out_obj = 10. ** (-0.4 * color_out_obj)
      # cvratio_obj = self.c.maxcvratio[sim_filter_obj]

      # Further select sources within dmag from the nominal tphot magnitude
      # if dmag > 0:
      #    magfilter = np.logical_and(mag_out >= mag_tphot - dmag,
      #                               mag_out <= mag_tphot + dmag)
      #    mag_in = mag_in[magfilter]
      #    mag_out = mag_out[magfilter]
      # elif dmag_in > 0:
      #    magfilter = np.logical_and(mag_in >= mag_tphot - dmag_in,
      #                               mag_in <= mag_tphot + dmag_in)
      #    mag_in = mag_in[magfilter]
      #    mag_out = mag_out[magfilter]

      assert len(mag_in_obj) > 0, "No source satisfies the measured magnitude criterion..."

      # Convert into flux
      flux_in_obj = 10.**(-0.4 * (mag_in_obj - magzpt_lr))
      flux_out_obj = np.where(mag_out_obj < 90.,
                              10.**(-0.4 * (mag_out_obj - magzpt_lr)), 0.)
      # flux_hr_in_obj = 10.**(-0.4 * (mag_hr_in_obj - magzpt_hr))
      # flux_hr_out_obj = np.where(mag_hr_out_obj < 90., 
      #                         10.**(-0.4 * (mag_hr_out_obj - magzpt_hr)), 0.)

      # Calculate the spread in output flux, and then convert into magnitude
      # error using the median output flux
      dflux_obj = flux_out_obj - flux_in_obj
      
      # perform sigma-clipping
      dflux, dflux_low, dflux_high = sigmaclip(dflux_obj, low=3., high=3.)
      sigma_filter = np.logical_and(dflux_obj >= dflux_low,
                                    dflux_obj < dflux_high)
      # sigma_filter = np.logical_and(sigma_filter, cvratio_obj < maxcvratio)

      fluxratio_in = fluxratio_in_obj[sigma_filter]
      fluxratio_out = fluxratio_out_obj[sigma_filter]

      # Re-calculate flux_in and flux_out
      mag_in = mag_in_obj[sigma_filter]
      mag_out = mag_out_obj[sigma_filter]

      # mag_hr_in = mag_hr_in_obj[sigma_filter]
      # mag_hr_out = mag_hr_out_obj[sigma_filter]

      # flux_in = flux_in_obj[sigma_filter]
      # flux_out = flux_out_obj[sigma_filter]
      # flux_hr_in = flux_hr_in_obj[sigma_filter]
      # flux_hr_out = flux_hr_out_obj[sigma_filter]

      print "{} sources selected around {}:".format(len(mag_out), 
                                                    target_name)
      print "Median input magnitude = {:.3f}".format(np.median(mag_in))
      print "Range of input magnitudes: {:.3f} <-> {:.3f}".format(
         np.min(mag_in), np.max(mag_in))

      if mag_tphot > 0:
         # Calculate the mean input & output flux ratios, convert into colors,
         # and then calculate the color bias
         mean_color_in = -2.5 * np.log10(np.mean(fluxratio_in))
         mean_color_out = -2.5 * np.log10(np.mean(fluxratio_out))
         mean_color_bias = mean_color_out - mean_color_in

         median_color_in = -2.5 * np.log10(np.median(fluxratio_in))
         median_color_out = -2.5 * np.log10(np.median(fluxratio_out))
         median_color_bias = median_color_out - median_color_in

         print "Average color bias: {:.3f}".format(mean_color_bias)
         print "Median color bias:  {:.3f}".format(median_color_bias)

         # # Calculate the mean & median output magnitudes (do the statistics in
         # # flux space)
         # mean_flux_in = np.mean(flux_in)
         # median_flux_in = np.median(flux_in)
         # # calculate mean & median hi-res fluxes
         # mean_flux_hr_in = np.mean(flux_hr_in)
         # median_flux_hr_in = np.median(flux_hr_in)
         
         # mean_mag_in = magzpt_lr - 2.5 * np.log10(mean_flux_in)
         # median_mag_in = magzpt_lr - 2.5 * np.log10(median_flux_in)
         # mean_mag_hr_in = magzpt_hr - 2.5 * np.log10(mean_flux_hr_in)
         # median_mag_hr_in = magzpt_hr - 2.5 * np.log10(median_flux_hr_in)

         # mean_flux_out = np.mean(flux_out)
         # median_flux_out = np.median(flux_out)
         # mean_flux_hr_out = np.mean(flux_hr_out)
         # median_flux_hr_out = np.median(flux_hr_out)

         # mean_mag_out = magzpt_lr - 2.5 * np.log10(mean_flux_out)
         # median_mag_out = magzpt_lr - 2.5 * np.log10(median_flux_out)
         # mean_mag_hr_out = magzpt_hr - 2.5 * np.log10(mean_flux_hr_out)
         # median_mag_hr_out = magzpt_hr - 2.5 * np.log10(median_flux_hr_out)

         # # Then calculate the mean and median magnitude bias
         # mean_magerr = (mean_mag_out - mean_mag_in
         # median_magerr = median_mag_out - median_mag_in

         # print "Average magnitude bias: {:.3f}".format(mean_magerr)
         # print "Median magnitude bias:  {:.3f}".format(median_magerr)
      else:
         print "No T-PHOT magnitudes given..."

      std_dflux = np.std(dflux_obj[sigma_filter])
      median_mag_out = np.median(mag_out)
      median_flux_out = 10.**(-0.4 * (median_mag_out - magzpt_lr))
      std_magerr = pu.sn2magerr(median_flux_out / std_dflux)
      # flux_tphot = 10.**(-0.4 * (mag_tphot - magzpt_lr))
      # std_magerr = pu.sn2magerr(flux_tphot / std_dflux)
      print "Magnitude error in color: {:.3f}".format(std_magerr)

      # mean_sn = flux_in[0] / np.abs(mean_dflux)
      # median_sn = flux_in[0] / np.abs(median_dflux)
      # std_sn = flux_in[0] / std_dflux
      # mean_magerr = pu.sn2magerr(mean_sn) * -1 * np.sign(mean_dflux)
      # median_magerr = pu.sn2magerr(median_sn) * -1 * np.sign(median_dflux)
      # std_magerr = pu.sn2magerr(std_sn)

      # print "Magnitude error from flux standard deviation: {:.3f}".format(std_magerr)

      return mag_in, mag_out, sim_filter_obj, sigma_filter


