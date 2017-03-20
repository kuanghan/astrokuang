import numpy as np
import matplotlib.pyplot as plt
import os
import PlotLePhare as plp
import eazy
from scipy.stats import mstats
from sedtools import kcorr
import pysynphot as S
from PhotomTools import photom_utils as pu
from astropy.cosmology import FlatLambdaCDM

cosmo0 = FlatLambdaCDM(H0=70.0, Om0=0.3)
arcsec_per_rad = 180. / np.pi * 3600.  # how many arcsec in a radian
##  LePhare directories, obsolete
zfix_dir = '/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/z7multi/LePhare/fix_specz'
zfree_dir = '/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/z7multi/LePhare/with_irac'

# Customize line styles
dashed = [9, 3, 9, 3]
dashdot = [3, 3, 8, 3]
dotted = [3, 3]


def plot_sedfit_one_eazy(objid, axes=[], pz_lw=2, pz_ls='-', pz_color='black'):
    """
    Plot the best-fit template at the Ly-alpha redshift and at the [OII]
    redshift.
    Also plot P(z) in a second axis.
    """
    if len(axes) == 0:
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        axes = [ax1, ax2]
    else:
        fig = axes[0].get_figure()
        ax1, ax2 = axes

    curdir = os.getcwd()
    assert objid in ['macs2129_z7a', 'macs2129_z7b', 'macs2129_z7c']
    ax1.set_yscale('log')

    # First, descend into EAZY directory (using BC03 M42 models)
    eazy_dir = '/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/z7multi/eazy/with_irac'
    os.chdir(eazy_dir)

    # The file that contains the physical properties of each template
    physfile = '/Users/khuang/eazy-1.00/templates/BC03_CSF_M42/bc2003_lr_m42_chab_csf.phys'
    
    # by default, use the M42 fits (Z = 0.02 Z_solar)
    # First, read the fixed-redshift fits
    p1 = eazy.PlotEAZY(catalog='hst_macs2129_clash_z7multi_specz_eazy.cat',
                       output='hst_macs2129_clash_z7multi_specz_eazy',
                       physfile=physfile, with_specz=True,
                       wavelims=[2000., 5e4])
    p1.plot_SED(objid + '_hi', ax=ax1, 
                plot_kwargs=dict(lw=2, ls='-', c='black'), 
                label=r'$z_{\mathrm{Ly}\alpha}=6.85$')

    # Now read free-floating photo-z solution
    p2 = eazy.PlotEAZY(catalog='hst_macs2129_clash_z7multi_eazy.cat',
                       output='hst_macs2129_clash_z7multi_eazy',
                       physfile=physfile)
    # output2 are in the following order: temp, zpeak, mass_best, 
    # log_age_best, sfr_best, tau, ebmv, MOD_BEST
    output2 = p2.read_SED(objid)
    # now plot P(z) in the second panel
    p2.plot_Pz(objid, ax=ax2, savefig=False, zmax=8.5,
        plot_kwargs=dict(label=objid, lw=pz_lw, ls=pz_ls, c=pz_color))
    ax2.set_title('')
    ax2.set_xlabel(r'$z_{\mathrm{phot}}$')

    # Now plot the [OII] solution
    os.chdir('../with_irac_m62')
    p3 = eazy.PlotEAZY(catalog='hst_macs2129_clash_z7multi_specz_m62_eazy.cat',
        output='hst_macs2129_clash_z7multi_specz_m62_eazy',
        physfile='/Users/khuang/eazy-1.00/templates/BC03_CSF_M62/bc2003_lr_m62_chab_csf.phys',
        with_specz=True, wavelims=[2000., 5e4])
    p3.plot_SED(objid + '_lo_m62', ax=ax1,
        plot_kwargs=dict(lw=1.5, ls='--', c='0.2'),
        label=r'$z_{\mathrm{[OII]}}=1.57$')

    os.chdir('../with_irac/')
    p1.plot_photom(objid + '_hi', ax=ax1, maglims=[31., 22.], #dmag_tick=2, 
        ebar_kwargs=dict(elinewidth=1, capthick=1))
    
    ax1.set_title('')
    # manually set wavelength ticks
    waveticks = [2000., 5000., 10000., 20000., 30000.]
    ax1.set_xticks(waveticks)
    ax1.set_xticklabels(['{:.1f}'.format(x / 1.e4) for x in waveticks])
    ax1.legend(loc=2, fontsize='large')
    ax1.text(0.95, 0.05, 'Image {}'.format(objid[-1].upper()),
        transform=ax1.transAxes, ha='right', va='bottom', size='x-large')

    os.chdir(curdir)

    return axes


def plot_sedfit_all():
    """
    Create a 2x2 panel plot, with three panels showing the best-fit template
    for images A to C, and the 4th panel showing the P(z) curves.
    """
    fig = plt.figure(figsize=(11, 9))
    ax1 = fig.add_subplot(2, 2, 1)  # for image A
    ax2 = fig.add_subplot(2, 2, 2)  # for image B
    ax3 = fig.add_subplot(2, 2, 3)  # for image C
    ax4 = fig.add_subplot(2, 2, 4)  # for P(z)

    # Plot image A
    ax1, ax4 = plot_sedfit_one_eazy('macs2129_z7a', axes=[ax1, ax4], pz_lw=2.5,
        pz_ls='-', pz_color='black')
    ax2, ax4 = plot_sedfit_one_eazy('macs2129_z7b', axes=[ax2, ax4], pz_lw=2.5,
        pz_ls='--', pz_color='black')
    ax4.lines[-1].set_dashes(dashed)
    ax3, ax4 = plot_sedfit_one_eazy('macs2129_z7c', axes=[ax3, ax4], pz_lw=2.5,
        pz_ls='-.', pz_color='black')
    ax4.lines[-1].set_dashes(dotted)

    # Format the P(z) panel
    ax4.plot([6.85, 6.85], [0, 2.5], ls='-', lw=1.5, c='black')
    ax4.plot([1.57, 1.57], [0, 2.5], ls='--', lw=1, c='black')
    ax4.lines[-1].set_dashes(dashed)
    ax4.legend(loc='upper center', fontsize='large')
    ax4.set_ylim(ymax=2.5)
    ax4.set_xlim(xmax=9.5)
    ax4.text(7.0, 1.5, r'$z_{\mathrm{Ly}\alpha}=6.85$', ha='left', va='top',
        size='large')
    ax4.text(1.57, 1.5, r'$z_{\mathrm{[OII]}}=1.57$', ha='left', va='top',
             size='large')

    fig.savefig('sedfit_z7abc.eps')

    return ax1, ax2, ax3, ax4


def calc_bestfit_eazy(objid, p=0.68, mu=1.0, mu_err=0.0, mass_scale=1.e8):
    """
    *** This is what we use to calculate numbers in Table 1. ***
    Calculate the best-fit stellar mass, SFR, sSFR, E(B-V), and age as well as
    the marginalized values.
    From EAZY results!!
    """
    curdir = os.getcwd()
    print "-" * 60
    print "Magnification factor = {:.2f} +/- {:.2f}".format(mu, mu_err)
    print "Confidence intervals represent {:d}% probability.".format(int(p*100))
    print "-" * 60
    os.chdir('/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/z7multi/eazy/with_irac')
    catalog = 'hst_macs2129_clash_z7multi_specz_eazy.cat'
    output = 'hst_macs2129_clash_z7multi_specz_eazy'
    physfile = '/Users/khuang/eazy-1.00/templates/BC03_CSF_M42/bc2003_lr_m42_chab_csf.phys'
    eazy.read_MCresults(objid, catalog=catalog, output=output,
        physfile=physfile, p=p, with_specz=True, mu=mu, mu_err=mu_err,
        mass_scale=mass_scale)
    os.chdir(curdir)


def calc_m1500_eazy(objid, mag, magerr, mu=1.0, mu_err=0.0, z=6.85, band='f125w', restwave=1600.):
    """
    Calculate M1500 from the observed F105W magnitude. Input magnitude errors
    are required to combine with magnification errors.
    """
    curdir = os.getcwd()
    eazy_dir = '/Users/khuang/Dropbox/Research/surfsup_dropbox/MACS2129/z7multi/eazy/with_irac'
    os.chdir(eazy_dir)

    # The file that contains the physical properties of each template
    physfile = '/Users/khuang/eazy-1.00/templates/BC03_CSF_M42/bc2003_lr_m42_chab_csf.phys'
    p = eazy.EAZY(with_specz=True)
    mod_best = p.read_mod_best(objid)
    sp = S.FileSpectrum(mod_best)
    band = S.ObsBandpass('wfc3,ir,{}'.format(band))
    k = kcorr.KCorrect(sp, S.Box(restwave, 100), band)
    absmag = k.absmag(z, mag) + 2.5 * np.log10(mu)

    # Now propogate the errors
    absflux = pu.ABmag2uJy(absmag)
    sn = pu.magerr2sn(magerr)
    absfluxerr = absflux * np.sqrt((1. / sn)**2 + (mu_err / mu)**2)
    absmagerr = pu.sn2magerr(absflux / absfluxerr)
    print "absflux = {}".format(absflux)
    print "absfluxerr = {}".format(absfluxerr)
    print "sn = {}".format(sn)
    print "absmagerr = {}".format(absmagerr)

    print "Rest-frame M_1500 of object {} is {:.3f} +/- {:.3f}.".format(objid, absmag, absmagerr)
    os.chdir(curdir)


def calc_lum_magnif(lum_input, lumerr_input, mu, mu_err=0.):
    """
    Calculate the magnification-corrected luminosity.
    """
    lum_corr = lum_input / mu
    lumerr_corr = lum_corr * np.sqrt((lumerr_input/lum_input)**2 + (mu_err/mu)**2)
    return lum_corr, lumerr_corr


def calc_bestfit_lephare(objid, p=0.68, mu=1.0, mu_err=0.0):
    """
    Calculate the best-fit stellar mass, SFR, sSFR, E(B-V), and age as well as
    the marginalized values.
    """
    curdir = os.getcwd()
    os.chdir(zfix_dir)
    q = plp.PlotLePhare(paramfile='macs2129_z7multi_m42_specz.param',
        idfile='macs2129_z7multi_m42_id.csv')
    lephare_id = q.data.LEPHARE_IDENT.loc[objid]
    print "-" * 60
    print "lephare_id = ", lephare_id
    print "Magnification factor = {:.2f} +/- {:.2f}".format(mu, mu_err)
    print "-" * 60
    data = np.genfromtxt('MonteCarlo/MCOutput_OBJ{:d}.txt'.format(int(lephare_id)),
        skip_header=1).T
    chi2 = data[2]
    age = data[3] / 1.e6   # in Myr
    ebmv = data[4]
    logmass = data[5]  # in M_solar
    logsfr = data[6]  # in M_solar / yr
    logssfr = logsfr - logmass + 9.  # in 1 / Gyr

    print "-" * 60
    print "Calculating the SED-fitting results for {}:".format(objid)
    print "Confidence intervals represent {:d}% probability.".format(int(p*100))
    p0 = (1. - p) / 2.
    p1 = 1. - p0

    # First, stellar mass in 10^8 M_solar
    logmass_best = q.data.MASS_MED.loc[objid]
    mass_best = 10.**(logmass_best - 8.)
    mass8 = 10.**(logmass - 8.)
    print "Stellar mass (in 10^8 M_solar):"
    mass_interval = confidence_interval(mass8, mass_best, p=p, scale=1.,
        mu=mu, mu_err=mu_err, verbose=False, latex=False, latex_sep=True)
    mass_marginal = np.mean(10.**logmass) / 1.e8 / mu
    print "Marginalized stellar mass (in 10^8 M_solar): $%.2f$" % mass_marginal
    print ""

    # Now SFR
    sfr = 10.**logsfr
    sfr_best = 10.**q.data.SFR_MED.loc[objid]
    print "SFR (in M_solar/yr):"
    sfr_interval = confidence_interval(sfr, sfr_best, p=p, scale=1.,
        mu=mu, mu_err=mu_err, verbose=False, latex=False, latex_sep=True)
    sfr_marginal = np.mean(10.**logsfr) / mu
    print "Marginalized SFR (in M_solar/yr): $%.2f$" % sfr_marginal
    print ""

    # Now sSFR
    ssfr = 10.**logssfr  # in Gyr^-1
    ssfr_best = sfr_best / (0.1 * mass_best)
    print "sSFR (in Gyr^{-1}):"
    ssfr_interval = confidence_interval(ssfr, ssfr_best, p=p, scale=1.,
        verbose=False, latex=False, latex_sep=True)
    ssfr_marginal = np.mean(ssfr)
    print "Marginalized sSFR (in Gyr^{-1}): $%.2f$" % ssfr_marginal
    print ""

    # Now E(B-V)
    ebmv_best = q.data.EBV_BEST.loc[objid]
    print "E(B-V):"
    ebmv_interval = confidence_interval(ebmv, ebmv_best, p=p, scale=1.,
        verbose=False, latex=False, latex_sep=True)
    ebmv_marginal = np.mean(ebmv)
    print "Marginalized E(B-V) = $%.2f$" % ebmv_marginal
    print ""

    # Now age (in Myr)
    age_best = q.data.AGE_BEST.loc[objid] / 1.e6
    print "Age (in Myr):"
    age_interval = confidence_interval(age, age_best, p=p, scale=1.,
        verbose=False, latex=False, latex_sep=True)
    age_marginal = np.mean(age)
    print "Marginalized age (in Myr) = $%.2f$" % age_marginal
    print ""

    os.chdir(curdir)
    print "\nFinished.\n"

    return data


def calc_Re_pix2kpc(R_pix, R_pix_err, z=6.85, pixscale=0.06, mu=1.0, mu_err=0.0):
    """
    Calculate the physical size in kpc from the measured angular size in
    pixels.
    Input size (R_pix) and error (R_pix_err) must be in pixels.
    """
    # Angular diameter distance in kpc
    DA_per_rad = cosmo0.angular_diameter_distance(z).value * 1.e3
    R_arcsec = R_pix * pixscale
    Rerr_arcsec = R_pix_err * pixscale
    R_kpc = R_arcsec / arcsec_per_rad * DA_per_rad
    Rerr_kpc = Rerr_arcsec / arcsec_per_rad * DA_per_rad
    # Now since R ~ sqrt(mu), let's figure out the error in sqrt(mu)
    mu_sqrt = np.sqrt(mu)
    mu_sqrt_err = 0.5 * mu_err / mu * mu_sqrt
    # Now calculate the intrinsic size in kpc
    R0_kpc = R_kpc / mu_sqrt
    R0_err_kpc = R0_kpc * np.sqrt((Rerr_kpc/R_kpc)**2 + (mu_sqrt_err/mu_sqrt)**2)

    return R0_kpc, R0_err_kpc


def calc_uv_sfr(absmag, magerr, ebmv=0., mu=1.0, mu_err=0.0, imf='salpeter'):
    """
    Calculate SFR from the observed UV flux density (given by the observed
    magnitude), redshift, and dust attenuation parameterized by E(B-V).
    For now assume Calzetti law (still to be implemented...)
    """
    tenpc_cm = 3.0857e19   # how many cm is in 10 pc
    sn = pu.magerr2sn(magerr)
    # convert from absolute magnitude to f_nu in uJy to f_nu in erg/s/cm2/Hz
    f_nu = 10.**((23.9 - absmag) / 2.5) * 1.e-29
    L_nu = f_nu * (4 * np.pi * tenpc_cm**2)  # convert to L_nu
    Lerr_nu = L_nu / sn
    sfr = 1.4e-28 * L_nu / mu
    if imf == 'chabrier':
        sfr = sfr * 0.63
    elif imf == 'kroupa':
        sfr = sfr * 0.67
    sfr_err = sfr * np.sqrt((1. / sn)**2 + (mu_err / mu)**2)

    return sfr, sfr_err

