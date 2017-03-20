import numpy as np
import os, glob
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MultipleLocator, FixedLocator
from matplotlib import font_manager as fm
from astropy.cosmology import Planck13, FlatLambdaCDM
from scipy.stats import binned_statistic
import match_gal_halo as mgh
import mass_stellar2halo as msh
import mass_size as ms
import select_galaxies as sg
import catalogs as rc
import uvj
import pandas as pd
import statsmodels.api as sm
from functools import reduce
LOWESS = sm.nonparametric.lowess


# lazy people's (just like myself) default cosmological parameters
cosmo0 = FlatLambdaCDM(H0=70., Om0=0.27)
Msolar_g = 1.98855e33   # solar mass in g
m_per_kpc = 3.0857e19   # how many meters in 1 kpc
cm_per_kpc = m_per_kpc * 1.e2  # hoe many cm in 1 kpc
arcsec_per_rad = 180. / np.pi * 3600.  # how many arcsec are in 1 radian
arcmin2_per_strad = (180. / np.pi * 60)**2  # how many arcmin^2 per steradian

# Color palette
BLUE = '#053061'
# RED = '#ef5869'
# BLUE = '#313695'
DARKBLUE = BLUE
LIGHTBLUE = 'blue'
RED = '#b2182b'
DARKRED = RED #'#d73027'
LIGHTRED = 'red'
PINK = '#de77ae'
GREEN = '#55a868'
PURPLE = '#8172b2'
YELLOW = '#ccb974'
CYAN = '#64b5cd'
DARKGRAY = '#373d3f'
GRAY = '#555f61'

# Customize line styles
dashed = [12, 3, 12, 3]
dashdot = [3, 3, 8, 3]
dotted = [3, 3]

# Define fonts used in the figures
HELVET_COND = '/Users/khuang/Dropbox/downloaded_fonts/helvetica_neue/HelveticaNeueLTStd-MdCn.otf'
AVENIRNEXT_COND = '/Users/khuang/Dropbox/downloaded_fonts/AvenirNextCondensed/AvenirNextCondensed-DemiBold.ttf'

# Read the number of galaxies in all bins/fields, as well as the number with
# n < 2.5
N_GAL_TOT = pd.read_csv('num_galaxies_tot.csv')  
N_GAL_DISK = pd.read_csv('num_galaxies_disk.csv')
# disk fraction as a function of redshift & field
DISKFRAC = N_GAL_DISK / N_GAL_TOT

# Stellar mass completeness limits, eyeball-estimated from Fig 2 of van der
# Wel et al. 2014
logM_lim_blue = np.log10(np.array([1.1e8, 2.5e8, 6.3e8, 1.3e9, 3.2e9, 7.9e9]))
logM_lim_red = np.log10(
    np.array([3.5e8, 8.9e8, 1.8e9, 4.0e9, 1.0e10, 1.8e10]))
logM_lim_udf = np.log10(
    np.array([1.e7, 5.01e7, 8.22e7, 1.70e8, 2.09e8, 3.80e8]))
    # np.array([4.79e6, 5.01e7, 8.22e7, 1.70e8, 2.09e8, 3.80e8]))

# Also read the TABULATED size-stellar mass relation from vdW+14
# (their Table 2 and Fig. 8)
RM_LATE = pd.read_csv('vdw14_fig8_late.csv')
RM_EARLY = pd.read_csv('vdw14_fig8_early.csv')


def MADN(x):
    """
    Calculate MADN (median absolute deviation).
    """
    m = np.median(np.abs(x - np.median(x)))
    return m


def sample_for_plots(c_gal_all, fields=['gds-wide', 'gds-deep', 'udf', 'gdn-deep', 'gdn-wide', 'uds', 'cosmos', 'egs'], H_maglim_wide=24.5, mass_lim=7., cosmo=cosmo0, filename='candels_rr_plots.cat'):
    """
    Does sample selection for later plots.
    This will prune samples with (1) bad GALFIT fits, and (2) photometry flag,
    (3) AGN flag, and (4) likely point sources.
    This function also allows selection of subsamples based on either Sersic
    index or sSFR, specify whether to include all galaxies or just the 10%
    tails of the distribution, and specify where the cut should be if
    including all galaxies.

    Input Arguments:
    -----------------
    c_gal_all: a dictionary storing the catalogs (as dataframes) for each of
               5 CANDELS fields. Keys are the names of each field.
    fields: a list of fields for which we perform sample selection (and will
            be written out to a text file).
    H_maglim_wide: the magnitude cut in H-band for the Wide region (the Deep
                   and UDF regions will be deeper by 0.7 and 2.2 mags,
                   respectively)
    mass_lim: impose a stellar mass limit
    cut_by: whether to split the subsample by Sersic index (sersic) or sSFR
            (ssfr)
    include: whether to include all galaxies ('all') or just the 10% tails
             ('n_10pc')
    n_cut: the Sersic index cut
    logssfr_cut: the cut in log10(ssfr) by which to cut the sample
    cosmo: an astropy.cosmology class instance for cosmological calculations
    filename: the resulting file name; if blank, use preset file name with
              the pattern 'candels_{cut_by}_{include}.dat'.
    """
    # Store the number of galaxies in each redshift bin in order to set
    # alpha weighted by number
    n_sample = []

    # Select sources in each field between 0 < z < 3
    c_out = pd.DataFrame()
    delta_N = 0
    for f in fields:
        print("Working on {}...".format(f.upper()))
        c_gal_f = sg.select_galaxies(c_gal_all, f, 0., 3.0, cosmo,
            filter_galfit=True, H_maglim_wide=H_maglim_wide)
        N1 = len(c_gal_f)
        c_gal_f = c_gal_f[c_gal_f.m_med >= mass_lim]
        N2 = len(c_gal_f)
        delta_N += (N1 - N2)
        print(N1, N2, np.sum(c_gal_f.udf_flag > 0), np.sum(c_gal_f.deep_flag > 0))
        c_gal_f = c_gal_f.reset_index()
        if f in ['gds-deep', 'gds-wide', 'udf']:
            c_gal_f['field'] = 'gds'
        elif f in ['gdn-deep', 'gdn-wide']:
            c_gal_f['field'] = 'gdn'
        else:
            c_gal_f['field'] = f
        c_out = c_out.append(c_gal_f, ignore_index=True)

    print("Stellar mass limit rejects {} galaxies.".format(delta_N))
    print("Writing out to candels_rr_plots.cat")
    c_out.to_csv(filename, header=True, index=False)

    return c_out


def func_smhm(conversion, z):
    """
    For a given SMHM relation name, return two functions, one for late-type
    and one for early-type galaxies, that calculate M200c given Mstar.
    """
    if conversion == 'b13':
        func_blue = lambda logm: msh.calc_halo_mass_b13_intp([logm], [z])[0]
        func_red = func_blue
    elif conversion == 't14':
        func_blue = lambda logm: msh.calc_halo_mass_t14_intp([logm], [z])[0]
        func_red = func_blue
    elif conversion == 'k13':
        func_blue = lambda logm: msh.calc_halo_mass_k13_intp([logm])[0]
        func_red = func_blue
    elif conversion == 'd10':
        func_blue = lambda logm: msh.calc_halo_mass_d10([logm], 1)
        func_red = lambda logm: msh.calc_halo_mass_d10([logm], 0)
    elif conversion == 'r15':
        func_blue = lambda logm: msh.calc_halo_mass_r15_intp(
            [logm], [1], [z])[0]
        func_red = lambda logm: msh.calc_halo_mass_r15_intp(
            [logm], [0], [z])[0]
    elif conversion == 'h15':
        func_blue = lambda logm: msh.calc_halo_mass_hudson15(logm, 1)
        func_red = lambda logm: msh.calc_halo_mass_hudson15(logm, 0)
    elif conversion == 'm16':
        func_blue = lambda logm: msh.calc_halo_mass_mandelbaum16(logm, 1)
        func_red = lambda logm: msh.calc_halo_mass_mandelbaum16(logm, 0)
    elif conversion == 'vmax':
        func_blue = lambda logm: msh.calc_halo_mass_vmax_intp([logm], [z])[0]
        func_red = func_blue
    return func_blue, func_red


def calc_rvir_lim(z, conversion, ltype='blue', cosmo=cosmo0):
    """
    Calculate the rough limits in R_vir.
    If ltype == 'blue', limit is for the blue galaxies from vdw14
    If ltype == 'red', limit is for the red galaxies from vdw14
    If ltype == 'udf', limit is estimated from UDF galaxies.
    """
    # Also show the halo radius limits roughly corresponding to mass
    # limits
    func_blue, func_red = func_smhm(conversion, z)

    i = np.searchsorted(np.arange(0., 3., 0.5), z) - 1
    i = np.maximum(0, i)
    if ltype == 'blue':
        logm_vir_lim = func_blue(logM_lim_blue[i])
        r_vir_lim = ms.R200(10.**logm_vir_lim, z, cosmo=cosmo)
        logr_vir_lim = np.log10(r_vir_lim)
    elif ltype == 'red':    
        logm_vir_lim = func_red(logM_lim_red[i])
        r_vir_lim = ms.R200(10.**logm_vir_lim, z, cosmo=cosmo)
        logr_vir_lim = np.log10(r_vir_lim)
    elif ltype == 'udf':
        logm_vir_lim = func_blue(logM_lim_udf[i])
        r_vir_lim = ms.R200(10.**logm_vir_lim, z, cosmo=cosmo)
        logr_vir_lim = np.log10(r_vir_lim)
    return logr_vir_lim


def plot_rr_allz(df, zbins=[0., 0.5, 1., 1.5, 2., 2.5, 3.], alpha_scatter=0.1, cut_by='sersic', include='all', logssfr_cut=-0.7, conversion='t14', suptitle=False, n_cut=2.5, add_vdw14_fit=False, add_vdw14_tab=False, fade4vdw=False, filename='', cosmo=cosmo0, lowess=False, legend_fontsize='medium', errorbars=True, legend_ncols=1):
    """
    Plot the halo size v.s. GALFIT size for the given redshift bins. The
    Reff--Rvir relations are derived using one SMHM relation.
    This function has the flexibility to do many kinds of sample selection,
    therefore it will take a while to generate the plots.

    Arguments:
    ----------
    df: a dataframes containing the galaxies that pass quality checks
    zbins: a list of redshift bin, the last value is the UPPER BOUND of the
           last bin
    alpha_scatter: the alpha value for the scatter plot showing all galaxies
                   (default=0.1)
    cut_by: whether to split sample by Sersic index (sersic) or sSFR (ssfr) or
            no cut (none; all galaxies are included)
    include: whether to include all galaxies (split by the property specified
             in "cut_by") or only include the tails of the distributions
             (n_10pc)
    logssfr_cut: the log(sSFR) value to cut the sample into star forming or
                 quiescent galaxies
    conversion: which SMHM relation to use (b13: Behroozi+13; d10: Dutton+10;
                k13: Kravtsov13; t14: candels-default; r15: RP+15)
    suptitle: whether to add a master title to the entire figure
    n_cut: where to make the Sersic cut to split the sample in 2
    nbins: how many bins in log(Rvir) to calculate median and scatter
    filename: the file name to which the figure will be saved to. If blank,
              then the figure will NOT be saved.
    """
    # Initialize dictionaries and arrays to store alpha = R_e / R_vir
    alpha = {}
    if cut_by == 'none':
        alpha['all'] = []        
    else:
        alpha['disk'] = []
        alpha['devauc'] = []

    # Define Figure parameters
    fig = plt.figure(figsize=(14, 11))
    nrows = int(np.ceil((len(zbins) - 1) / 3.))
    ncols = 3
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75),
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.05,
                     share_all=True,
                     aspect=False,
                     label_mode='L')
    xticks_major = FixedLocator([2., 3.])
    xticks_minor = FixedLocator(np.concatenate(
                                [np.log10(np.arange(20., 99., 10.)),
                                np.log10(np.arange(200., 999., 100.))]))
    yticks_major = FixedLocator([0., 1.])
    yticks_minor = FixedLocator(np.concatenate(
                                [np.log10(np.arange(0.2, 0.99, 0.1)),
                                np.log10(np.arange(2., 9.9, 1.))]))
    xlabel = r'$R_{200c}$ / kpc'
    ylabel = r'$R_{\rm{eff}}$ / kpc'
    # xlabel = r'$\log(R_{\rm{200c}}\,/\,\rm{kpc})$'
    # xbins = np.arange(1.4, 3.05, 0.15)
    # xbin_centers = (xbins[:-1] + xbins[1:]) / 2.
    xlims = [1.4, 3.1]
    ylims = [-0.6, 1.3]
    xticks = np.arange(1.5, 3., 0.5)
    # ylabel = r'$\log(R_{\rm{eff}}\,/\,\rm{kpc})$'

    # Also store the number of galaxies in each redshift bin in order to set
    # alpha weighted by number
    n_sample = np.histogram(df['zbest'].values, bins=zbins)[0]
    patches = []
    # Open files that store the average alpha == R_eff / R_vir for each
    # redshift bin and subsample
    if cut_by == 'none':
        f_alpha = open('alpha_{}.txt'.format(conversion), 'w')
        f_alpha.write('zlo,alpha_med,alpha_std,number\n')
    else:
        f_alpha1 = open(
            'alpha_late_{}_{}_{}.txt'.format(cut_by, conversion, include),
            'w')
        f_alpha2 = open(
            'alpha_early_{}_{}_{}.txt'.format(cut_by, conversion, include),
            'w')
        f_alpha1.write('zlo,alpha_med,alpha_std,number\n')
        f_alpha2.write('zlo,alpha_med,alpha_std,number\n')
    f_early = open('early_frac.txt', 'w')
    f_early.write('zlo,f_early,n_tot\n')

    # define the SMHM-related variables
    title_dic = {'t14': 'SMHM Relation 1', 'b13': 'SMHM Relation 2',
        'b13vir': 'SMHM Relation 2',
        'k13': 'SMHM Relation 3', 'd10': 'SMHM Relation 4',
        'r15': 'SMHM Relation 4', 'h15': 'Hudson+15',
        'm16': 'Mandelbaum+16', 'vmax': 'Vmax'}
    logRvir_col = {'t14': 'r200c_kpc_t14', 'b13': 'r200c_m200c_kpc_b13',
        'b13vir': 'r200c_mvir_kpc_b13', 
        'k13': 'r200c_kpc_k13', 'd10': 'r200c_kpc_d10',
        'r15': 'r200c_kpc_r15', 'h15': 'r200c_kpc_h15',
        'm16': 'r200c_kpc_m16', 'vmax': 'r200c_kpc_vmax'}

    # Now iterate over each redshift bin
    for i in range(len(zbins) - 1):
        z0 = zbins[i]
        z1 = zbins[i + 1]
        zm = (z0 + z1) / 2.
        c_bin = df[(df.zbest >= z0) & (df.zbest < z1)]
        n_bin = len(c_bin)
        # n_sample.append(n_bin)
        sersic_n_bin = c_bin['sersic_n'].values
        logssfr_bin = c_bin['logssfr_med'].values
        print("Median redshift for {:.1f} < z < {:.1f} is {:.2f}".format(
            z0, z1, c_bin['zbest'].median()))

        # Calculate halo radius limits corresponding to stellar mass limits
        logr_vir_lim_blue = calc_rvir_lim(zm, conversion, ltype='blue',
                                          cosmo=cosmo)
        logr_vir_lim_red = calc_rvir_lim(zm, conversion, ltype='red',
                                         cosmo=cosmo)
        logr_vir_lim_udf = calc_rvir_lim(zm, conversion, ltype='udf',
                                         cosmo=cosmo)
        # Figure out the halo radius bins
        xbins = np.arange(logr_vir_lim_udf, logr_vir_lim_udf+0.15*11+0.01,
                          0.15)
        xbin_centers = (xbins[:-1] + xbins[1:]) / 2.

        # Plot a (theoretically motivated) average relation between halo size
        # and galaxy disk size
        logR_halo_grid = np.linspace(0., 4., 100)
        R_eff_avg = 0.035 / np.sqrt(2.) * 10.**logR_halo_grid * 1.678
        R_eff_K13 = 0.015 * 10.**logR_halo_grid #* 1.678
        if cut_by != 'none':
            grid[i].plot(logR_halo_grid, np.log10(R_eff_avg), ls='--', lw=2,
                c=GREEN, label=r'J/M equality')
        grid[i].plot(logR_halo_grid, np.log10(R_eff_K13), ls='-', lw=2,
            c='black', label='Kravtsov13')

        # Define variables
        early_frac = float(np.sum(c_bin['sersic_n']>2.5)) / n_bin
        f_early.write('{},{},{}\n'.format(zbins[i], early_frac, n_bin))
        title = title_dic[conversion]
        # calculate log10(Rvir) in kpc (proper)
        logR_vir_bin = np.log10(c_bin[logRvir_col[conversion]])
        logR_eff_bin = np.log10(c_bin['r_eff_vrest_kpc'])

        # FIRST, make scatter plot
        # alpha_scatter_bin = alpha_scatter
        alpha_scatter_bin = alpha_scatter * (n_sample.max() / len(c_bin))
        ms_bin = 8 * (n_sample.max()) / len(c_bin)
        if cut_by == 'none':
            if alpha_scatter > 0:
                patch = grid[i].scatter(logR_vir_bin, logR_eff_bin,
                                        edgecolor=GRAY, facecolor=GRAY,
                                        marker='^', s=ms_bin,
                                        alpha=alpha_scatter,
                                        # alpha=alpha_scatter_bin,
                                        label='')
                patches.append(patch)
            # calculate the number of sources per bin
            hist = np.histogram(logR_vir_bin, bins=xbins)[0]
            # pdb.set_trace()
            med, bin_edges, binnum = binned_statistic(logR_vir_bin,
                logR_eff_bin, statistic=np.median, bins=xbins)
            lower, bin_edges, binnum = binned_statistic(logR_vir_bin,
                logR_eff_bin,
                statistic=lambda x: np.percentile(x, 16),
                bins=xbins)
            err_low = med - lower
            upper, bin_edges, binnum = binned_statistic(logR_vir_bin,
                logR_eff_bin,
                statistic=lambda x: np.percentile(x, 84),
                bins=xbins)
            err_high = upper - med

            if errorbars:
                yerr = [err_low[hist>=5], err_high[hist>=5]]
            else:
                yerr = None
            grid[i].errorbar(xbin_centers[hist>=5], med[hist>=5],
                yerr=yerr,
                fmt='s', ecolor=DARKGRAY, mfc=DARKGRAY,
                mec=DARKGRAY, ms=12, mew=2)
            # calculate alpha and write to a file
            log_alpha = logR_eff_bin - logR_vir_bin
            # Store alpha values
            alpha['all'].append(10.**log_alpha)
            log_alpha_med = np.median(log_alpha)
            log_alpha_mad = MADN(log_alpha)
            f_alpha.write('{},{},{},{}\n'.format(z0, log_alpha_med,
                          log_alpha_mad, len(log_alpha)))

        else:
            if cut_by == 'sersic':
                if include == 'all':
                    n_low = n_cut
                    n_high = n_cut
                    label1 = r'$\mathbf{n < %.1f}$' % n_cut
                    label2 = r'$\mathbf{n \geq %.1f}$' % n_cut
                elif include == 'n_10pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_10pc = int(round(n_bin * 0.1))
                    print("n_10pc = ", n_10pc)
                    n_low = sersic_n_bin_sorted[n_10pc]
                    n_high = sersic_n_bin_sorted[-n_10pc]
                    label1 = r'Lowest 10% in $\mathbf{n}$'
                    label2 = r'Highest 10% in $\mathbf{n}$'
                elif include == 'n_15pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_15pc = int(round(n_bin * 0.15))
                    print("n_15pc = ", n_15pc)
                    n_low = sersic_n_bin_sorted[n_15pc]
                    n_high = sersic_n_bin_sorted[-n_15pc]
                    label1 = r'Lowest 15% in $\mathbf{n}$'
                    label2 = r'Highest 15% in $\mathbf{n}$'
                elif include == 'n_20pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_20pc = int(round(n_bin * 0.2))
                    print("n_20pc = ", n_20pc)
                    n_low = sersic_n_bin_sorted[n_20pc]
                    n_high = sersic_n_bin_sorted[-n_20pc]
                    label1 = r'Lowest 20% in $\mathbf{n}$'
                    label2 = r'Highest 20% in $\mathbf{n}$'
                logR_vir_disk = logR_vir_bin[sersic_n_bin < n_low]
                logR_eff_disk = logR_eff_bin[sersic_n_bin < n_low]
                logR_vir_devauc = logR_vir_bin[sersic_n_bin > n_high]
                logR_eff_devauc = logR_eff_bin[sersic_n_bin > n_high]
                n_blue = np.sum(sersic_n_bin < n_low)
                n_red = np.sum(sersic_n_bin > n_high)
                print("n_low, n_high = {:.2f}, {:.2f}".format(n_low, n_high))
            elif cut_by == 'ssfr':
                if include == 'all':
                    logssfr_low = logssfr_cut
                    logssfr_high = logssfr_cut
                    label1 = r'$\log(\rm{sSFR})>%.1f$' % logssfr_cut
                    label2 = r'$\log(\rm{sSFR})<%.1f$' % logssfr_cut
                elif include == 'n_10pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_10pc = int(round(len(logssfr_bin_sorted) * 0.1))
                    print("n_10pc = ", n_10pc)
                    logssfr_low = logssfr_bin_sorted[n_10pc]
                    logssfr_high = logssfr_bin_sorted[-n_10pc]
                    # passive_bin = (logssfr_bin < ssfr_low)
                    # active_bin = (logssfr_bin > ssfr_high)
                    label1 = 'Highest 10% in sSFR'
                    label2 = 'Lowest 10% in sSFR'
                elif include == 'n_15pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_15pc = int(round(len(logssfr_bin_sorted) * 0.15))
                    print("n_15pc = ", n_15pc)
                    logssfr_low = logssfr_bin_sorted[n_15pc]
                    logssfr_high = logssfr_bin_sorted[-n_15pc]
                    label1 = 'Highest 15% in sSFR'
                    label2 = 'Lowest 15% in sSFR'
                elif include == 'n_20pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_20pc = int(round(len(logssfr_bin_sorted) * 0.2))
                    print("n_20pc = ", n_20pc)
                    logssfr_low = logssfr_bin_sorted[n_20pc]
                    logssfr_high = logssfr_bin_sorted[-n_20pc]
                    label1 = 'Highest 20% in sSFR'
                    label2 = 'Lowest 20% in sSFR'
                logR_vir_disk = logR_vir_bin[logssfr_bin > logssfr_high]
                logR_eff_disk = logR_eff_bin[logssfr_bin > logssfr_high]
                logR_vir_devauc = logR_vir_bin[logssfr_bin < logssfr_low]
                logR_eff_devauc = logR_eff_bin[logssfr_bin < logssfr_low]
                n_blue = np.sum(logssfr_bin > logssfr_high)
                n_red = np.sum(logssfr_bin < logssfr_low)
            elif cut_by == 'uvj':
                red = uvj.select_uvj_red(c_bin, (z0 + z1) / 2.)
                label1 = 'Blue'
                label2 = 'Red'
                logR_vir_disk = logR_vir_bin[~red]
                logR_eff_disk = logR_eff_bin[~red]
                logR_vir_devauc = logR_vir_bin[red]
                logR_eff_devauc = logR_eff_bin[red]
                n_red = np.sum(red)
                n_blue = len(c_bin) - n_red
            if alpha_scatter > 0:
                print("n_blue, n_red = {}, {}".format(n_blue, n_red))
                # Make scatter plots
                if n_blue >= n_red:
                    alpha_red = alpha_scatter_bin
                    alpha_blue = alpha_scatter_bin * np.sqrt(float(n_red) / n_blue)
                else:
                    alpha_blue = alpha_scatter_bin
                    alpha_red = alpha_scatter_bin * np.sqrt(float(n_blue) / n_red)
                alpha_blue = np.minimum(0.7, alpha_blue)
                # alpha_blue = np.maximum(0.02, alpha_blue)
                alpha_red = np.minimum(0.7, alpha_red)
                # alpha_red = np.maximum(0.02, alpha_red)
                print("alpha_blue, alpha_red = {:.2f}, {:.2f}".format(
                    alpha_blue, alpha_red))
                patch_disk = grid[i].scatter(logR_vir_disk, logR_eff_disk,
                                             marker='x', s=ms_bin,
                                             linewidths=2,
                                             alpha=alpha_scatter,
                                             # alpha=alpha_blue,
                                             edgecolor=LIGHTBLUE,
                                             facecolor=LIGHTBLUE,
                                             label='')
                patch_devauc = grid[i].scatter(logR_vir_devauc,
                                               logR_eff_devauc,
                                               marker='o', s=ms_bin,
                                               alpha=alpha_scatter,
                                               # alpha=alpha_red,
                                               edgecolor=LIGHTRED,
                                               facecolor=LIGHTRED,
                                               label='')
                patches.append([patch_disk, patch_devauc])

            if lowess == True:
                l_disk = LOWESS(logR_eff_disk, logR_vir_disk,
                                return_sorted=True)
                l_devauc = LOWESS(logR_eff_devauc, logR_vir_devauc,
                                  return_sorted=True)
                grid[i].plot(l_disk[:,0], l_disk[:,1], lw=3, c=DARKBLUE,
                             label='')
                grid[i].plot(l_devauc[:,0], l_devauc[:,1], lw=3, c=DARKRED,
                             label='')
            
            # Add binned medians and scatters
            # If add vdw14 stuff, lower the alpha
            if (add_vdw14_fit or add_vdw14_tab) and fade4vdw:
                alpha_medians = 0.3
            else:
                alpha_medians = 1.0
            # calculate the number of sources per bin
            hist1 = np.histogram(logR_vir_disk, bins=xbins)[0]
            hist2 = np.histogram(logR_vir_devauc, bins=xbins)[0]

            # Calculate binned statistics
            med_disk, bin_edges, binnum = binned_statistic(logR_vir_disk,
                logR_eff_disk, statistic=np.median, bins=xbins)
            lower_disk, bin_edges, binnum = binned_statistic(logR_vir_disk,
                logR_eff_disk, statistic=lambda x: np.percentile(x, 16),
                bins=xbins)
            err_low_disk = med_disk - lower_disk
            upper_disk, bin_edges, binnum = binned_statistic(logR_vir_disk,
                logR_eff_disk, statistic=lambda x: np.percentile(x, 84),
                bins=xbins)
            err_high_disk = upper_disk - med_disk
            # std_disk, bin_edges, binnum = binned_statistic(logR_vir_disk,
            #     logR_eff_disk, statistic=MADN, bins=xbins)

            if cut_by == 'uvj':
                grid[i].errorbar(xbin_centers[hist1>=5]-0.01,
                                 med_disk[hist1>=5],
                                 # yerr=[err_low_disk[hist1>=5], err_high_disk[hist1>=5]],
                                 fmt='s', ecolor=DARKBLUE, mfc=DARKBLUE,
                                 mec=DARKBLUE,
                                 ms=12, mew=2, label='Our sample (blue)',
                                 elinewidth=2) 
            else:
                if errorbars:
                    yerr = [err_low_disk[hist1>=5], err_high_disk[hist1>=5]]
                else:
                    yerr = None
                grid[i].errorbar(xbin_centers[hist1>=5]-0.01,
                                 med_disk[hist1>=5], yerr=yerr, fmt='s',
                                 ecolor=DARKBLUE, mfc=DARKBLUE, mec=DARKBLUE,
                                 ms=12, mew=2, label=label1,elinewidth=2)

            med_devauc, bin_edges, binnum = binned_statistic(
                logR_vir_devauc, logR_eff_devauc, statistic=np.median,
                bins=xbins)
            lower_devauc, bin_edges, binnum = binned_statistic(
                logR_vir_devauc, logR_eff_devauc,
                statistic=lambda x: np.percentile(x, 16),
                bins=xbins)
            err_low_devauc = med_devauc - lower_devauc
            upper_devauc, bin_edges, binnum = binned_statistic(
                logR_vir_devauc, logR_eff_devauc,
                statistic=lambda x: np.percentile(x, 84),
                bins=xbins)
            err_high_devauc = upper_devauc - med_devauc

            if cut_by == 'uvj':
                grid[i].errorbar(xbin_centers[hist2>=5]+0.01, 
                    med_devauc[hist2>=5],
                    # yerr=[err_low_devauc[hist2>=5],err_high_devauc[hist2>=5]],
                    fmt='o', ecolor=DARKRED, mfc='white', mec=DARKRED,
                    ms=12, mew=2., label='Our sample (red)',
                    elinewidth=2)
            else:
                if errorbars:
                    yerr=[err_low_devauc[hist2>=5],err_high_devauc[hist2>=5]]
                else:
                    yerr = None
                grid[i].errorbar(xbin_centers[hist2>=5]+0.01,
                                 med_devauc[hist2>=5],
                                 yerr=yerr, fmt='o', ecolor=DARKRED,
                                 mfc='white', mec=DARKRED, ms=12, mew=2.,
                                 label=label2, elinewidth=2)
                                 #, alpha=alpha_medians)

            # calculate alpha values
            log_alpha_disk = logR_eff_disk - logR_vir_disk
            log_alpha_devauc = logR_eff_devauc - logR_vir_devauc
            alpha['disk'].append(10.**log_alpha_disk)
            alpha['devauc'].append(10.**log_alpha_devauc)
            
            log_alpha_disk_med = np.median(log_alpha_disk)
            log_alpha_disk_mad = MADN(log_alpha_disk)
            f_alpha1.write('{},{},{},{}\n'.format(z0, log_alpha_disk_med,
                           log_alpha_disk_mad, len(log_alpha_disk)))
            
            log_alpha_devauc_med = np.median(log_alpha_devauc)
            log_alpha_devauc_mad = MADN(log_alpha_devauc)
            f_alpha2.write('{},{},{},{}\n'.format(z0, log_alpha_devauc_med,
                           log_alpha_devauc_mad, len(log_alpha_disk)))

        if add_vdw14_fit:
            # First, add the fitted relation
            logr_vir_vdw_b, logr_eff_vdw_b = ms.Reff_Rvir_vdw14(zm, 'blue',
                conversion=conversion)
            logr_vir_vdw_r, logr_eff_vdw_r = ms.Reff_Rvir_vdw14(zm, 'red',
                conversion=conversion)
            if add_vdw14_tab and fade4vdw:
                alpha_vdw14 = 0.3
            else:
                alpha_vdw14 = 1.0
            l1, = grid[i].plot(logr_vir_vdw_b, logr_eff_vdw_b, ls='-',
                             c=DARKBLUE, lw=5, alpha=alpha_vdw14,
                             label='vdW14 blue')
            l2, = grid[i].plot(logr_vir_vdw_r, logr_eff_vdw_r, ls='--',
                             c=DARKRED, lw=5, alpha=alpha_vdw14,
                             label='vdW14 red')
            l2.set_dashes(dashed)
            
        if add_vdw14_tab:
            # Second, add the tabulated relation
            zstr = '%.1f' % z0
            logM_star = RM_LATE['logM'].values
            func_late, func_early = func_smhm(conversion, zm)
            logM200c_late = np.array(list(map(func_late, logM_star)))
            logR200c_late = np.log10(ms.R200(
                                     10.**logM200c_late, zm, cosmo=cosmo))
            logReff_late_lo = RM_LATE[zstr+'_lo'].values
            logReff_late_med = RM_LATE[zstr+'_med'].values
            logReff_late_hi = RM_LATE[zstr+'_hi'].values
            err_late_lo = logReff_late_med - logReff_late_lo
            err_late_hi = logReff_late_hi - logReff_late_med
            grid[i].errorbar(logR200c_late-0.01, logReff_late_med,
                             # yerr=[err_late_lo, err_late_hi],
                             fmt='^', ms=12, mfc='white', mec='Navy',
                             ecolor='Navy', elinewidth=1,
                             mew=1, label='vdW14 med. (blue)')

            logM200c_early = np.array(list(map(func_early, logM_star)))
            logR200c_early = np.log10(ms.R200(
                                      10.**logM200c_early, zm, cosmo=cosmo))
            logReff_early_lo = RM_EARLY[zstr+'_lo'].values
            logReff_early_med = RM_EARLY[zstr+'_med'].values
            logReff_early_hi = RM_EARLY[zstr+'_hi'].values
            err_early_lo = logReff_early_med - logReff_early_lo
            err_early_hi = logReff_early_hi - logReff_early_med
            grid[i].errorbar(logR200c_early+0.01, logReff_early_med,
                             # yerr=[err_early_lo, err_early_hi],
                             fmt='v', ms=12, mfc='Maroon', mec='Maroon',
                             ecolor='Maroon', elinewidth=1,
                             mew=1, label='vdW14 med. (red)')

        grid[i].tick_params(axis='both', which='major', labelsize='xx-large')

        grid[i].text(0.05, 0.95,
                     r'$%.1f < z < %.1f$' % (zbins[i], zbins[i + 1]),
                     ha='left', va='top', size=22,
                     transform=grid[i].transAxes)

        if i == len(zbins) - 2:
            prop = fm.FontProperties(fname=HELVET_COND,
                                     size=legend_fontsize)
            if cut_by == 'uvj':
                legend = grid[i].legend(loc=4, markerscale=0.5,
                                        frameon=False,
                                        prop=prop,
                                        ncol=2, markerfirst=False)
            else:
                legend = grid[i].legend(loc=4, markerscale=0.5,
                                        frameon=False,
                                        prop=prop,
                                        markerfirst=False,
                                        ncol=legend_ncols)
        elif i == 1:
            grid[i].set_title(title, size=24)
        
        # Also show the halo radius limits roughly corresponding to mass
        # limits
        grid[i].plot([logr_vir_lim_udf] * 2, [ylims[0], ylims[0]+0.15], lw=4,
                      color=DARKGRAY)

        grid[i].xaxis.set_major_locator(xticks_major)
        grid[i].set_xticklabels([r'$10^2$', r'$10^3$'])
        grid[i].xaxis.set_minor_locator(xticks_minor)
        grid[i].yaxis.set_major_locator(yticks_major)
        grid[i].set_yticklabels([r'$10^0$', r'$10^1$'])
        grid[i].yaxis.set_minor_locator(yticks_minor)
        grid[i].tick_params(which='major', length=10, pad=8,
                            labelsize=18)
        grid[i].tick_params(which='minor', length=5)
        # grid[i].set_xticks(xticks)
        grid[i].set_xlim(*xlims)
        grid[i].set_ylim(*ylims)
        grid[i].set_ylabel(ylabel, size=24)
        grid[i].set_xlabel(xlabel, size=24)

    if cut_by == 'none':
        f_alpha.close()
    else:
        f_alpha1.close()
        f_alpha2.close()
    f_early.close()

    # Now figure out the proper alpha for each panel, with maximum given
    # by alpha_scatter
    # alpha_weights = float(np.max(n_sample)) / np.array(n_sample)
    # alpha_zbins = alpha_scatter * alpha_weights
    # alpha_zbins = np.maximum(alpha_scatter-0.1, alpha_zbins)
    # alpha_zbins = np.minimum(alpha_scatter+0.1, alpha_zbins)
    # print "n_sample =", n_sample
    # # print "alpha_zbins =", alpha_zbins
    # # Scale transparency proportionally in each redshift bin according to
    # # sample size
    # if alpha_scatter > 0:
    #     for i in range(len(zbins) - 1):
    #         if cut_by == 'none':
    #             patches[i].set_alpha(alpha_zbins[i])
    #         else:
    #             patches[i][0].set_alpha(alpha_zbins[i])
    #             patches[i][1].set_alpha(alpha_zbins[i])
        
    if len(filename):
        fig.savefig(filename)
    return fig, grid


def plot_rr_allmethods(df, zlo=0., dz=0.5, alpha_scatter=0.1, cut_by='none', include='all', logssfr_cut=-0.7, conversions=['t14', 'b13', 'k13', 'r15'], suptitle=False, n_cut=2.5, add_vdw14=False, filename='', nrows=1, cosmo=cosmo0, legend_fontsize='x-large'):
    """
    Plot the halo size v.s. GALFIT size using all SMHM relations in the same
    redshift bin. The Reff--Rvir relations are derived using one SMHM
    relation.

    Arguments:
    ----------
    df: a dataframes containing the galaxies that pass the quality checks
    zlo: the lower bound of the redshift bin
    dz: the redshift bin width
    alpha_scatter: the alpha value for the scatter plot showing all galaxies
                   (default=0.1)
    cut_by: whether to split sample by Sersic index (sersic) or sSFR (ssfr) or
            no cut (none; all galaxies are included)
    include: whether to include all galaxies (split by the property specified
             in "cut_by") or only include the tails of the distributions
             (n_10pc)
    logssfr_cut: the log(sSFR) value to cut the sample into star forming or
                 quiescent galaxies
    n_cut: where to cut the sample by Sersic index
    conversion: which SMHM relation to use (b13: Behroozi+13; d10: Dutton+10;
                k13: Kravtsov13)
    filename: the file name of the figure; if blank, then the figure is not
              saved.
    """
    if nrows == 1:
        fig = plt.figure(figsize=(16, 4.7))
        ncols = len(conversions)
    else:
        nrows, ncols = 2, 2
        fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75), nrows_ncols=(nrows, ncols),
        axes_pad=0.05, share_all=True, aspect=False, label_mode='L')
    # xlabel = r'$\log(R_{\rm{200c}}\,/\,\rm{kpc})$'
    # xbins = np.arange(1.4, 3.05, 0.15)
    xlims = [1.4, 3.1]
    ylims = [-0.6, 1.3]
    xticks = np.arange(1.5, 3.1, 0.5)
    xticks_major = FixedLocator([2., 3.])
    xticks_minor = FixedLocator(np.concatenate(
                                [np.log10(np.arange(20., 99., 10.)),
                                np.log10(np.arange(200., 999., 100.))]))
    yticks_major = FixedLocator([0., 1.])
    yticks_minor = FixedLocator(np.concatenate(
                                [np.log10(np.arange(0.2, 0.99, 0.1)),
                                np.log10(np.arange(2., 9.9, 1.))]))
    xlabel = r'$R_{200c}$ / kpc'
    ylabel = r'$R_{\rm{eff}}$ / kpc'
    # ylabel = r'$\log(R_{\rm{eff}}\,/\,\rm{kpc})$'
    # xbin_centers = (xbins[:-1] + xbins[1:]) / 2.

    z0 = zlo
    z1 = zlo + dz
    zm = (z0 + z1) / 2.

    # define the SMHM-related variables
    title_dic = {'t14': 'SMHM Relation 1', 'b13': 'SMHM Relation 2',
        'b13vir': 'SMHM Relation 2',
        'k13': 'SMHM Relation 3', 'd10': 'SMHM Relation 4',
        'r15': 'SMHM Relation 4'}
    logRvir_col = {'t14': 'r200c_kpc_t14', 'b13': 'r200c_m200c_kpc_b13',
        'b13vir': 'r200c_mvir_kpc_b13', 
        'k13': 'r200c_kpc_k13', 'd10': 'r200c_kpc_d10',
        'r15': 'r200c_kpc_r15'}

    c_bin = df[(df.zbest >= z0) & (df.zbest < z1)]
    logR_eff_bin = np.log10(c_bin['r_eff_vrest_kpc'])
    sersic_n_bin = c_bin['sersic_n'].values
    logssfr_bin = c_bin['logssfr_med'].values

    # write the binned median values into a dataframe
    if cut_by in ['sersic', 'ssfr']:
        alpha_med_late = pd.DataFrame()
        alpha_med_early = pd.DataFrame()
    else:
        alpha_med = pd.DataFrame()

    for i in range(len(conversions)):
        logR_vir_bin = np.log10(c_bin[logRvir_col[conversions[i]]])
        # Calculate the halo radius lower limits corresponding to stellar
        # mass limits
        zm = zlo + dz / 2.
        logr_vir_lim_blue = calc_rvir_lim(zm, conversions[i], ltype='blue',
                                          cosmo=cosmo)
        logr_vir_lim_red = calc_rvir_lim(zm, conversions[i], ltype='red',
                                         cosmo=cosmo)
        logr_vir_lim_udf = calc_rvir_lim(zm, conversions[i], ltype='udf',
                                         cosmo=cosmo)
        # Figure out the halo radius bins
        xbins = np.arange(logr_vir_lim_udf, logr_vir_lim_udf+0.15*11+0.01,
                          0.15)
        xbin_centers = (xbins[:-1] + xbins[1:]) / 2.
        
        # FIRST, make scatter plot
        if cut_by == 'none':
            grid[i].scatter(logR_vir_bin, logR_eff_bin, edgecolor=GRAY,
                facecolor=GRAY, marker='^', s=8, alpha=alpha_scatter,
                label='')
            log_alpha = logR_eff_bin - logR_vir_bin
            print("Median alpha for all galaxies in {:.1f}<z<{:.1f} is {:.4f}".format(zlo, zlo+dz, 10.**np.median(log_alpha)))
            # calculate the number of sources per bin
            hist = np.histogram(logR_vir_bin, bins=xbins)[0]
            
            med, bin_edges, binnum = binned_statistic(logR_vir_bin,
                logR_eff_bin, statistic=np.median, bins=xbins)
            lower, bin_edges, binnum = binned_statistic(logR_vir_bin,
                logR_eff_bin, statistic=lambda x: np.percentile(x, 16),
                bins=xbins)
            err_low = med - lower
            upper, bin_edges, binnum = binned_statistic(logR_vir_bin,
                logR_eff_bin, statistic=lambda x: np.percentile(x, 84),
                bins=xbins)
            err_high = upper - med
            
            alpha_med['smhm-{}'.format(i+1)] = med
            alpha_med.index = bin_edges[:-1]

            grid[i].errorbar(xbin_centers[hist>=5], med[hist>=5],
                yerr=[err_low[hist>=5], err_high[hist>=5]],
                fmt='p', ecolor=DARKGRAY, mfc=DARKGRAY,
                mec=DARKGRAY, ms=12, mew=2)
            grid[i].set_title(title_dic[conversions[i]], size=22)

        else:
            if cut_by == 'sersic':
                if include == 'all':
                    n_low = n_cut
                    n_high = n_cut
                    label1 = r'$\mathbf{n < %.1f}$' % n_cut
                    label2 = r'$\mathbf{n \geq %.1f}$' % n_cut
                elif include == 'n_10pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_10pc = int(round(len(c_bin) * 0.1))
                    print("n_10pc = ", n_10pc)
                    n_low = sersic_n_bin_sorted[n_10pc]
                    n_high = sersic_n_bin_sorted[-n_10pc]
                    label1 = r'Lowest 10% in $\mathbf{n}$'
                    label2 = r'Highest 10% in $\mathbf{n}$'
                elif include == 'n_15pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_15pc = int(round(len(c_bin) * 0.15))
                    print("n_15pc = ", n_15pc)
                    n_low = sersic_n_bin_sorted[n_15pc]
                    n_high = sersic_n_bin_sorted[-n_15pc]
                    label1 = r'Lowest 15% in $\mathbf{n}$'
                    label2 = r'Highest 15% in $\mathbf{n}$'
                elif include == 'n_20pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_20pc = int(round(len(c_bin) * 0.2))
                    print("n_20pc = ", n_20pc)
                    n_low = sersic_n_bin_sorted[n_20pc]
                    n_high = sersic_n_bin_sorted[-n_20pc]
                    label1 = r'Lowest 20% in $\mathbf{n}$'
                    label2 = r'Highest 20% in $\mathbf{n}$'
                logR_vir_bin_disk = logR_vir_bin[sersic_n_bin < n_low]
                logR_eff_bin_disk = logR_eff_bin[sersic_n_bin < n_low]
                logR_vir_bin_devauc = logR_vir_bin[sersic_n_bin > n_high]
                logR_eff_bin_devauc = logR_eff_bin[sersic_n_bin > n_high]
            elif cut_by == 'ssfr':
                if include == 'all':
                    logssfr_low = logssfr_cut
                    logssfr_high = logssfr_cut
                    label1 = 'log(sSFR)>{:.1f}'.format(logssfr_cut)
                    label2 = 'log(sSFR)<{:.1f}'.format(logssfr_cut)
                elif include == 'n_10pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_10pc = int(round(len(c_bin) * 0.1))
                    print("n_10pc = ", n_10pc)
                    logssfr_low = logssfr_bin_sorted[n_10pc]
                    logssfr_high = logssfr_bin_sorted[-n_10pc]
                    label1 = 'Highest 10% in sSFR'
                    label2 = 'Lowest 10% in sSFR'
                elif include == 'n_15pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_15pc = int(round(len(c_bin) * 0.15))
                    print("n_15pc = ", n_15pc)
                    logssfr_low = logssfr_bin_sorted[n_15pc]
                    logssfr_high = logssfr_bin_sorted[-n_15pc]
                    label1 = 'Highest 15% in sSFR'
                    label2 = 'Lowest 15% in sSFR'
                elif include == 'n_20pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_20pc = int(round(len(c_bin) * 0.2))
                    print("n_20pc = ", n_20pc)
                    logssfr_low = logssfr_bin_sorted[n_20pc]
                    logssfr_high = logssfr_bin_sorted[-n_20pc]
                    label1 = 'Highest 20% in sSFR'
                    label2 = 'Lowest 20% in sSFR'
                logR_vir_bin_disk = logR_vir_bin[logssfr_bin > logssfr_high]
                logR_eff_bin_disk = logR_eff_bin[logssfr_bin > logssfr_high]
                logR_vir_bin_devauc = logR_vir_bin[logssfr_bin < logssfr_low]
                logR_eff_bin_devauc = logR_eff_bin[logssfr_bin < logssfr_low]
            
            # Print the median alpha (the proportionality constant between
            # Reff and R200c)
            print("For SMHM relation {}:".format(conversions[i].upper()))
            print("median(alpha) = {:.4f} for late-types".format(
                  np.median(logR_eff_bin_disk - logR_vir_bin_disk)))
            print("median(alpha) = {:.4f} for early-types".format(
                  np.median(logR_eff_bin_devauc - logR_vir_bin_devauc)))

            # Make scatter plots
            grid[i].scatter(logR_vir_bin_disk, logR_eff_bin_disk, marker='x',
                s=8, alpha=alpha_scatter, edgecolor=LIGHTBLUE,
                facecolor=LIGHTBLUE, label='')
            grid[i].scatter(logR_vir_bin_devauc, logR_eff_bin_devauc,
                            marker='o', s=8, alpha=alpha_scatter,
                            edgecolor=LIGHTRED, facecolor=LIGHTRED,
                            label='')

            # calculate the number of sources per bin
            hist1 = np.histogram(logR_vir_bin_disk, bins=xbins)[0]
            hist2 = np.histogram(logR_vir_bin_devauc, bins=xbins)[0]
            
            # Calculate binned statistics
            med_disk, bin_edges, binnum = binned_statistic(logR_vir_bin_disk,
                logR_eff_bin_disk, statistic=np.median, bins=xbins)
            lower_disk, bin_edges, binnum = binned_statistic(
                logR_vir_bin_disk, logR_eff_bin_disk,
                statistic=lambda x: np.percentile(x, 16),
                bins=xbins)
            err_low_disk = med_disk - lower_disk
            upper_disk, bin_edges, binnum = binned_statistic(
                logR_vir_bin_disk, logR_eff_bin_disk,
                statistic=lambda x: np.percentile(x, 84),
                bins=xbins)
            err_high_disk = upper_disk - med_disk
            grid[i].errorbar(xbin_centers[hist1>=5]-0.01, med_disk[hist1>=5],
                yerr=[err_low_disk[hist1>=5], err_high_disk[hist1>=5]],
                fmt='s', ecolor=DARKBLUE,
                mfc=DARKBLUE, mec=DARKBLUE, ms=12, mew=2, label=label1)
            alpha_med_late['smhm{}-{}'.format(i+1, cut_by)] = med_disk
            alpha_med_late.index = bin_edges[:-1]

            med_devauc, bin_edges, binnum = binned_statistic(
                logR_vir_bin_devauc, logR_eff_bin_devauc, statistic=np.median,
                bins=xbins)
            lower_devauc, bin_edges, binnum = binned_statistic(
                logR_vir_bin_devauc, logR_eff_bin_devauc,
                statistic=lambda x: np.percentile(x, 16),
                bins=xbins)
            err_low_devauc = med_devauc - lower_devauc
            upper_devauc, bin_edges, binnum = binned_statistic(
                logR_vir_bin_devauc, logR_eff_bin_devauc,
                statistic=lambda x: np.percentile(x, 84),
                bins=xbins)
            err_high_devauc = upper_devauc - med_devauc
            # std_devauc, bin_edges, binnum = binned_statistic(
            #     logR_vir_bin_devauc, logR_eff_bin_devauc, statistic=MADN,
            #     bins=xbins)
            
            # grid[i].plot(xbin_centers[hist2>=5], med_devauc[hist2>=5],
            #              marker='o', ls='none', mfc='none', mec=DARKRED,
            #              ms=12, label=label2, mew=2)
            # grid[i].plot(xbin_centers[hist2>=5], lower_devauc[hist2>=5],
            #              marker='', ls='-', color=DARKRED, lw=1.5,
            #              label='')
            # grid[i].plot(xbin_centers[hist2>=5], upper_devauc[hist2>=5],
            #              marker='', ls='-', color=DARKRED, lw=1.5,
            #              label='')
            grid[i].errorbar(xbin_centers[hist2>=5]+0.01,
                             med_devauc[hist2>=5],
                             yerr=[err_low_devauc[hist2>=5], err_high_devauc[hist2>=5]],
                             fmt='o', ecolor=DARKRED, mfc='white',
                             mec=DARKRED, ms=12, mew=2, label=label2)

            alpha_med_early['smhm{}-{}'.format(i+1, cut_by)] = med_devauc
            alpha_med_early.index = bin_edges[:-1]

        # Plot a (theoretically moticated) average relation between halo size
        # and galaxy disk size
        logR_halo_grid = np.linspace(0., 4., 100)
        R_eff_avg = 0.035 / np.sqrt(2.) * 10.**logR_halo_grid * 1.678
        R_eff_K13 = 0.015 * 10.**logR_halo_grid #* 1.678

        if cut_by != 'none':
            grid[i].plot(logR_halo_grid, np.log10(R_eff_avg), ls='--', lw=2,
                c=GREEN, label=r'J/M equality')
        grid[i].plot(logR_halo_grid, np.log10(R_eff_K13), ls='-', lw=2,
            c='black', label='Kravtsov13')
        
        grid[i].tick_params(axis='both', which='major', labelsize='xx-large')

        if i == len(conversions) - 1:
            prop = fm.FontProperties(fname=HELVET_COND,
                                     size=legend_fontsize)
            legend = grid[i].legend(loc=4, markerscale=0.5, prop=prop,
                                    frameon=False, markerfirst=False)
        if nrows == 1:
            grid[i].text(0.05, 0.95, r'$%.1f < z < %.1f$' % (z0, z1),
                ha='left', va='top', size=22,
                transform=grid[i].transAxes)
        else:
            grid[i].text(0.05, 0.95,
                title_dic[conversions[i]] + '\n' + r'$%.1f < z < %.1f$' % (z0, z1),
                ha='left', va='top', size=22,
                transform=grid[i].transAxes)
        if (cut_by != 'none') and (nrows == 1):
            grid[i].set_title('SMHM relation {}'.format(i+1), size=22)

        # Also show the halo radius limits roughly corresponding to mass
        # limits
        ymin = -0.6
        grid[i].plot([logr_vir_lim_udf] * 2, [ymin, ymin+0.15], lw=4,
                      color=DARKGRAY)

        if add_vdw14:
            logr_vir_vdw_b, logr_eff_vdw_b = ms.Reff_Rvir_vdw14(zm, 'blue',
                conversion=conversions[i])
            logr_vir_vdw_r, logr_eff_vdw_r = ms.Reff_Rvir_vdw14(zm, 'red',
                conversion=conversions[i])
            l1, = grid[i].plot(logr_vir_vdw_b, logr_eff_vdw_b, ls='-.',
                             c='blue', lw=4)
            l2, = grid[i].plot(logr_vir_vdw_r, logr_eff_vdw_r, ls='-.',
                             c='red', lw=4)
            l1.set_dashes(dashdot)
            l2.set_dashes(dashdot)

        grid[i].xaxis.set_major_locator(xticks_major)
        grid[i].set_xticklabels([r'$10^2$', r'$10^3$'])
        grid[i].xaxis.set_minor_locator(xticks_minor)
        grid[i].yaxis.set_major_locator(yticks_major)
        grid[i].set_yticklabels([r'$10^0$', r'$10^1$'])
        grid[i].yaxis.set_minor_locator(yticks_minor)
        grid[i].tick_params(which='major', length=10, pad=8,
                            labelsize='xx-large')
        grid[i].tick_params(which='minor', length=5)
        # grid[i].set_xticks(xticks)
        grid[i].set_xlim(xlims)
        grid[i].set_ylim(ylims)
        grid[i].set_ylabel(ylabel, size=24)
        grid[i].set_xlabel(xlabel, size=24)
        # grid[i].tick_params(axis='both', labelsize='x-large')
                     
    if cut_by != 'none':
        alpha_med_late.to_csv(
            'alpha_med_allmethods_late_{}.csv'.format(cut_by), index=True,
            header=True)
        alpha_med_early.to_csv(
            'alpha_med_allmethods_early_{}.csv'.format(cut_by), index=True,
            header=True)
    else:
        alpha_med.to_csv('alpha_med_allmethods.csv', index=True, header=True)
    # plt.tight_layout()
    if len(filename):
        fig.savefig(filename)
    return grid


def calc_alpha_stat(cut_by='none', method='b13'):
    """
    Calculate global statistics of alpha from the median alpha in each redshift
    bin.
    """
    if cut_by == 'none':
        c = pd.read_csv('alpha_{}.txt'.format(method))
        avg_wht = 10.**np.average(c['alpha_med'], weights=c['number'])
        print("Weighted average alpha from all redshift bins is {:.3f}".format(avg_wht))
        med_allz = 10.**np.median(c['alpha_med'])
        print("Median alpha from all redshift bins is {:.3f}".format(med_allz))
    else:
        c1 = pd.read_csv('alpha_late_{}_{}.txt'.format(cut_by, method))
        c2 = pd.read_csv('alpha_early_{}_{}.txt'.format(cut_by, method))
        avg_wht1 = 10.**np.average(c1['alpha_med'], weights=c1['number'])
        med_allz1 = 10.**np.median(c1['alpha_med'])
        avg_wht2 = 10.**np.average(c2['alpha_med'], weights=c2['number'])
        med_allz2 = 10.**np.median(c2['alpha_med'])
        print("Weighted average alpha for LATE-TYPES from all redshift bins is {:.3f}".format(avg_wht1))
        print("Weighted average alpha for EARLY-TYPES from all redshift bins is {:.3f}".format(avg_wht2))
        print("Median alpha for LATE-TYPES from all redshift bins is {:.3f}".format(med_allz1))
        print("Median alpha for EARLY-TYPES from all redshift bins is {:.3f}".format(med_allz2))





# ----------------------------------
# OLD STUFF USING MDPL2 SIMULATIONS
# ----------------------------------


def plot_rr_allz_v0(c_gal_all, zarr=[0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], fields=['gds-wide', 'gds-deep', 'udf', 'gdn-deep', 'gdn-wide', 'uds', 'cosmos', 'egs'], H_maglim_wide=24.5, alpha_scatter=0.1, cut_by='sersic', include='all', logssfr_cut=-0.7, conversion='b13', suptitle=False, cosmo=cosmo0, n_cut=2.5, d10_cut_by='sersic', equal_bins=False, nbins=8, xaxis='rvir', yaxis='reff', filename=''):
    """
    Plot the halo size v.s. GALFIT size for the given redshift bins. The
    Reff--Rvir relations are derived using one SMHM relation.
    This function has the flexibility to do many kinds of sample selection,
    therefore it will take a while to generate the plots.

    Arguments:
    ----------
    c_gal_all: a dictionary of dataframes representing the galaxy catalog in
               each field
    zarr: a list of redshift bin LOWER BOUNDS
    dz: the redshift bin width
    fields: a list of field names
    H_maglim_wide: H-band magnitude limit in CANDELS Wide regions 
                   (default=24.5)
    alpha_scatter: the alpha value for the scatter plot showing all galaxies
                   (default=0.1)
    cut_by: whether to split sample by Sersic index (sersic) or sSFR (ssfr) or
            no cut (none; all galaxies are included)
    include: whether to include all galaxies (split by the property specified
             in "cut_by") or only include the tails of the distributions
             (n_10pc)
    logssfr_cut: the log(sSFR) value to cut the sample into star forming or
                 quiescent galaxies
    conversion: which SMHM relation to use (b13: Behroozi+13; d10: Dutton+10;
                k13: Kravtsov13)
    """
    # Initialize dictionaries and arrays to store alpha = R_e / R_vir
    alpha = {}
    if cut_by == 'none':
        alpha['all'] = []        
    else:
        alpha['disk'] = []
        alpha['devauc'] = []
    rcorr_all = {}
    for i in range(len(zarr)):
        rcorr_all[i] = []
    sersic_n_all = {}
    smass_all = {}
    hmag_all = {}
    ssfr_all = {}

    # Define Figure parameters
    fig = plt.figure(figsize=(14, 11))
    nrows = int(np.ceil((len(zarr) - 1) / 3.))
    ncols = 3
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75),
                     nrows_ncols=(nrows, ncols),
                     axes_pad=0.2,
                     share_all=True,
                     aspect=True,
                     label_mode='L')

    # Also store the number of galaxies in each redshift bin in order to set
    # alpha weighted by number
    n_sample = []
    patches = []
    # Open files that store the average alpha == R_eff / R_vir for each
    # redshift bin and subsample
    if cut_by == 'none':
        f_alpha = open('alpha_{}.txt'.format(conversion), 'w')
        f_alpha.write('zlo,alpha_med,alpha_std,number\n')
    else:
        f_alpha1 = open(
            'alpha_late_{}_{}.txt'.format(cut_by, conversion), 'w')
        f_alpha2 = open(
            'alpha_early_{}_{}.txt'.format(cut_by, conversion), 'w')
        f_alpha1.write('zlo,alpha_med,alpha_std,number\n')
        f_alpha2.write('zlo,alpha_med,alpha_std,number\n')
    f_early = open('early_frac.txt', 'w')
    f_early.write('zlo,f_early,n_tot\n')

    # Now iterate over each redshift bin
    for i in range(len(zarr) - 1):
        z0 = zarr[i]
        z1 = zarr[i + 1]
        if z0 < 1.5:
            band = 'j'
            zp = 1.5
        else:
            band = 'h'
            zp = 2.2
        print("Working on {:.1f} < z < {:.1f}".format(z0, z1)) 
        c_bin = pd.DataFrame()
        for f in fields:
            print("Working on {}...".format(f.upper()))
            c_gal_f = sg.select_galaxies(c_gal_all, f, z0, z1, cosmo,
                filter_galfit=True, H_maglim_wide=H_maglim_wide)
            c_bin = c_bin.append(c_gal_f, ignore_index=True)

        # Define variables
        sersic_n_bin = c_bin['n_'+band.lower()].values
        early_frac = float(np.sum(sersic_n_bin>2.5)) / float(len(c_bin))
        sersic_n_all['{:.1f}'.format(zarr[i])] = sersic_n_bin
        smass_all['{:.1f}'.format(zarr[i])] = c_bin['m_med'].values
        f_early.write('{},{},{}\n'.format(zarr[i], early_frac, len(c_bin)))
        logssfr_bin = c_bin['logssfr_med'].values
        if conversion == 'b13':
            M_halo = 10.**c_bin.m200c_b13.values  # in M_solar
            title = 'SMHM Relation 2'
        elif conversion == 'k13':
            M_halo = 10.**c_bin.m_halo_k13.values  # in M_solar
            title = 'SMHM Relation 3'
        elif conversion == 't14':
            M_halo = 10.**c_bin.m_halo_t14.values  # in M_solar
            title = 'SMHM Relation 1'
        elif conversion == 'r15':
            M_halo = 10.**c_bin.m200c_r15.values  # in M_solar
            title = ''
        elif conversion == 'd10':
            print("Calculating halo mass using Dutton 2010 relation...")
            title = 'SMHM Relation 4'
            # Calculate here b/c we might be using different cuts for galaxy
            # types
            # gtype = 1 --> late-type
            # gtype = 0 --> early-type
            # gtype = 2 --> undefined
            if cut_by == 'none':
                if d10_cut_by == 'sersic':
                    # decide which SMHM relation to use using Sersic index
                    gtype = np.where(sersic_n_bin > n_cut, 0, 1)
                else:
                    print("Using log(sSFR) to determine whether to use the late-type SMHM relation or not...")
                    gtype = np.where(logssfr_bin > logssfr_cut, 1, 0)
            elif cut_by == 'sersic':
                if include == 'all':
                    n_low = n_cut
                    n_high = n_cut
                elif include == 'n_10pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_10pc = len(sersic_n_bin) / 10
                    print("n_10pc = ", n_10pc)
                    n_low = sersic_n_bin_sorted[n_10pc]
                    n_high = sersic_n_bin_sorted[-n_10pc]
                    print("n_low = {:.1f}, n_high = {:.1f}".format(n_low, n_high))
                gtype = np.where(sersic_n_bin > n_high, 0, 2)
                gtype = np.where(sersic_n_bin < n_low, 1, gtype)
            elif cut_by == 'ssfr':
                if include == 'all':
                    logssfr_low = logssfr_cut
                    logssfr_high = logssfr_cut
                elif include == 'n_10pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_10pc = len(logssfr_bin_sorted) / 10
                    print("n_10pc = ", n_10pc)
                    logssfr_low = logssfr_bin_sorted[n_10pc]
                    logssfr_high = logssfr_bin_sorted[-n_10pc]
                    print("ssfr_low = {:.2f}, ssfr_high = {:.2f}".format(logssfr_low, logssfr_high))
                gtype = np.where(logssfr_bin > logssfr_high, 0, 2)
                gtype = np.where(logssfr_bin < logssfr_low, 1, gtype)
            logM_halo = msh.calc_halo_mass_d10(c_bin.m_med.values, gtype)
            c_bin['logm_vir_d10'] = logM_halo
            M_halo = 10 ** logM_halo
        zbest = c_bin.zbest.values
        R_vir_bin = np.array(list(map(ms.R200, M_halo, zbest)))  # in kpc        
        logR_vir_bin = np.log10(R_vir_bin)
        R_eff_bin = c_bin['re_' + band.lower()].values  # in arcsec
            

        # Also apply galaxy color gradient correction, if this information
        # does not already exist in the catalogs
        rcorr_disk = -0.35 + 0.12 * c_bin.zbest - 0.25 * (c_bin.m_med - 10.)
        # cap the maximum correction factor at 0
        rcorr_disk = np.minimum(rcorr_disk.values, 0.)
        rcorr_devauc = -0.25
        rcorr = np.where(sersic_n_bin < n_cut, rcorr_disk, rcorr_devauc)
        rcorr_factor = np.power((1. + zbest) / (1. + zp), rcorr)
        # R_eff_bin still in arcsec
        R_eff_bin = R_eff_bin * rcorr_factor
        rcorr_all[i] = np.concatenate([rcorr_all[i], rcorr_factor])
        # Calculate angular diameter distance in kpc / arcsec
        dist_a = np.array([cosmo.angular_diameter_distance(z).value * 1.e3 / arcsec_per_rad for z in zbest])
        R_eff_kpc = R_eff_bin * dist_a
        logR_eff_bin = np.log10(R_eff_kpc)

        # Decide which quantity to plot in both axes
        if xaxis == 'rvir':
            xdata = logR_vir_bin
            xlabel = r'$\log R_{\rm{vir}}$ [kpc]'
            bins = np.arange(1.0, 3.2, 0.2)
            xlims = [1.0, 3.5]
            xticks = np.arange(1.5, 3.5, 0.5)
        else:
            xdata = np.log10(0.015) + logR_vir_bin
            xlabel = r'$\log(0.015\times R_{\rm{vir}})$ [kpc]'
            bins = np.arange(-0.8, 1.4, 0.2)
            xlims = [-0.8, 1.5]
            xticks = np.arange(-0.5, 1.5, 0.5)
        if yaxis == 'reff':
            ydata = logR_eff_bin
            ylabel = r'$\log R_{\rm{eff}}$ [kpc]'
        else:
            ydata = np.where(sersic_n_bin > 2.5, np.log10(1.34) + logR_eff_bin,
                             logR_eff_bin)
            ylabel = r'$\log R_{1/2}$ [kpc]'
        n_sample.append(len(xdata))
        
        bin_centers = (bins[:-1] + bins[1:]) / 2.
        # FIRST, make scatter plot
        if cut_by == 'none':
            patch = grid[i].scatter(xdata, ydata, edgecolor=GRAY,
                facecolor=GRAY, marker='^', s=8, alpha=alpha_scatter)
            patches.append(patch)
            log_alpha = logR_eff_bin - logR_vir_bin
            # Store alpha values
            alpha['all'].append(10.**log_alpha)
            # calculate the number of sources per bin
            hist = np.histogram(xdata, bins=bins)[0]
            # pdb.set_trace()
            med, bin_edges, binnum = binned_statistic(xdata,
                ydata, statistic=np.median, bins=bins)
            std, bin_edges, binnum = binned_statistic(xdata,
                ydata, statistic=np.std, bins=bins)
            
            grid[i].errorbar(bin_centers[hist>=5], med[hist>=5],
                yerr=std[hist>=5], fmt='s', ecolor=DARKGRAY, mfc=DARKGRAY,
                mec='none', ms=10, mew=1.5)
            # Write alpha to a file
            log_alpha = logR_eff_bin - logR_vir_bin
            log_alpha_med = np.median(log_alpha)
            log_alpha_std = np.std(log_alpha)
            f_alpha.write('{},{},{},{}\n'.format(z0, log_alpha_med, log_alpha_std, len(log_alpha)))

        else:
            if cut_by == 'sersic':
                if include == 'all':
                    n_low = n_cut
                    n_high = n_cut
                    label1 = r'$n < %.1f$' % n_cut
                    label2 = r'$n \geq %.1f$' % n_cut
                elif include == 'n_10pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_10pc = len(sersic_n_bin) / 10
                    print("n_10pc = ", n_10pc)
                    n_low = sersic_n_bin_sorted[n_10pc]
                    n_high = sersic_n_bin_sorted[-n_10pc]
                    label1 = r'Lowest 10% in $n$'
                    label2 = r'Highest 10% in $n$'
                xdata_disk = xdata[sersic_n_bin < n_low]
                ydata_disk = ydata[sersic_n_bin < n_low]
                xdata_devauc = xdata[sersic_n_bin > n_high]
                ydata_devauc = ydata[sersic_n_bin > n_high]
                print("N_disk, N_devauc = {}, {}".format(
                    np.sum(sersic_n_bin < n_cut), np.sum(sersic_n_bin > n_cut)))
            elif cut_by == 'ssfr':
                if include == 'all':
                    logssfr_low = logssfr_cut
                    logssfr_high = logssfr_cut
                    # passive_bin = (logssfr_bin <= logssfr_cut)
                    # active_bin = np.logical_not(passive_bin)
                    label1 = 'log(sSFR)>{:.1f}'.format(logssfr_cut)
                    label2 = 'log(sSFR)<{:.1f}'.format(logssfr_cut)
                elif include == 'n_10pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_10pc = len(logssfr_bin_sorted) / 10
                    print("n_10pc = ", n_10pc)
                    logssfr_low = logssfr_bin_sorted[n_10pc]
                    logssfr_high = logssfr_bin_sorted[-n_10pc]
                    # passive_bin = (logssfr_bin < ssfr_low)
                    # active_bin = (logssfr_bin > ssfr_high)
                    label1 = 'Highest 10% in sSFR'
                    label2 = 'Lowest 10% in sSFR'
                xdata_disk = xdata[logssfr_bin > logssfr_high]
                ydata_disk = ydata[logssfr_bin > logssfr_high]
                xdata_devauc = xdata[logssfr_bin < logssfr_low]
                ydata_devauc = ydata[logssfr_bin < logssfr_low]
            # Make scatter plots
            patch_disk = grid[i].scatter(xdata_disk, ydata_disk, marker='x',
                s=8, alpha=alpha_scatter, edgecolor=LIGHTBLUE,
                facecolor=LIGHTBLUE)
            patch_devauc = grid[i].scatter(xdata_devauc, ydata_devauc,
                marker='o', s=8, alpha=alpha_scatter, edgecolor=LIGHTRED,
                facecolor=LIGHTRED)
            patches.append([patch_disk, patch_devauc])
            log_alpha_disk = ydata_disk - xdata_disk
            log_alpha_devauc = ydata_devauc - xdata_devauc
            # store alpha values
            alpha['disk'].append(10.**log_alpha_disk)
            alpha['devauc'].append(10.**log_alpha_devauc)
            # calculate the number of sources per bin
            hist1 = np.histogram(xdata_disk, bins=bins)[0]
            hist2 = np.histogram(xdata_devauc, bins=bins)[0]
            # Calculate binned statistics
            med_disk, bin_edges, binnum = binned_statistic(xdata_disk,
                ydata_disk, statistic=np.median, bins=bins)
            std_disk, bin_edges, binnum = binned_statistic(xdata_disk,
                ydata_disk, statistic=np.std, bins=bins)
            grid[i].errorbar(bin_centers[hist1>=5], med_disk[hist1>=5],
                yerr=std_disk[hist1>=5], fmt='s', ecolor=DARKBLUE,
                mfc=DARKBLUE, mec='none', ms=10, mew=1.2, label=label1)
            med_devauc, bin_edges, binnum = binned_statistic(
                xdata_devauc, ydata_devauc, statistic=np.median,
                bins=bins)
            std_devauc, bin_edges, binnum = binned_statistic(
                xdata_devauc, ydata_devauc, statistic=np.std,
                bins=bins)
            grid[i].errorbar(bin_centers[hist2>=5], med_devauc[hist2>=5],
                yerr=std_devauc[hist2>=5], fmt='o', ecolor=DARKRED, mfc='none',
                mec=DARKRED, ms=10, mew=1.5, label=label2)
            # Write alpha info
            log_alpha_disk = ydata_disk - xdata_disk
            log_alpha_disk_med = np.median(log_alpha_disk)
            log_alpha_disk_std = np.std(log_alpha_disk)
            f_alpha1.write('{},{},{},{}\n'.format(z0, log_alpha_disk_med, log_alpha_disk_std, len(log_alpha_disk)))
            log_alpha_devauc = ydata_devauc - xdata_devauc
            log_alpha_devauc_med = np.median(log_alpha_devauc)
            log_alpha_devauc_std = np.std(log_alpha_devauc)
            f_alpha2.write('{},{},{},{}\n'.format(z0, log_alpha_devauc_med, log_alpha_devauc_std, len(log_alpha_disk)))
        # Plot a (theoretically moticated) average relation between halo size
        # and galaxy disk size
        logR_halo_grid = np.linspace(0., 4., 100)
        R_eff_avg = 0.035 / np.sqrt(2.) * 10.**logR_halo_grid * 1.678
        R_eff_K13 = 0.015 * 10.**logR_halo_grid #* 1.678
        if xaxis == 'rvir':
            xcons = logR_halo_grid
        else:
            xcons = np.log10(0.015) + logR_halo_grid

        if cut_by != 'none':
            grid[i].plot(xcons, np.log10(R_eff_avg), ls='--', lw=2,
                c=GREEN, label=r'$J$/$M$ equality')
        grid[i].plot(xcons, np.log10(R_eff_K13), ls='-', lw=1,
            c='black', label='Kravtsov13')
        grid[i].tick_params(axis='both', which='major', labelsize='xx-large')
        if cut_by == 'none':
            fontsize = 'x-large'
        else:
            fontsize = 'large'
        legend = grid[i].legend(loc=4, markerscale=1, fontsize=fontsize,
                                frameon=False, markerfirst=False)
        grid[i].text(0.05, 0.95, r'$%.1f < z < %.1f$' % (zarr[i], zarr[i + 1]),
            ha='left', va='top', size='xx-large', transform=grid[i].transAxes)

        if i == 1:
            grid[i].set_title(title, size=28)
        else:
            grid[i].set_title('')
        grid[i].set_xticks(xticks)
        grid[i].set_xlim(xlims)
        grid[i].set_ylim(-1, 1.5)
        grid[i].set_ylabel(ylabel, size=28)
        grid[i].set_xlabel(xlabel, size=28)
                     
    if cut_by == 'none':
        f_alpha.close()
    else:
        f_alpha1.close()
        f_alpha2.close()
    f_early.close()

    # plt.suptitle(title, fontsize=20)
    # Now figure out the proper alpha for each panel, with maximum given
    # by alpha_scatter
    alpha_weights = float(np.max(n_sample)) / np.array(n_sample)
    alpha_zbins = alpha_scatter * alpha_weights
    alpha_zbins = np.maximum(alpha_scatter-0.1, alpha_zbins)
    alpha_zbins = np.minimum(alpha_scatter+0.1, alpha_zbins)
    print("n_sample =", n_sample)
    print("alpha_zbins =", alpha_zbins)
    for i in range(len(zarr) - 1):
        if cut_by == 'none':
            patches[i].set_alpha(alpha_zbins[i])
        else:
            patches[i][0].set_alpha(alpha_zbins[i])
            patches[i][1].set_alpha(alpha_zbins[i])
        
    if len(filename):
        fig.savefig(filename)
    # rcorr_all = reduce(lambda x, y: np.concatenate([x, y]), rcorr_all)
    return grid, alpha, rcorr_all, sersic_n_all, smass_all


def plot_rr_allmethods_v0(c_gal_all, zlo=0, dz=0.5, fields=['gds-wide', 'gds-deep', 'udf', 'gdn-deep', 'gdn-wide', 'uds', 'cosmos', 'egs'], H_maglim_wide=24.5, zspec=False, alpha_scatter=0.2, cut_by='sersic', include='all', logssfr_cut=-0.7, conversions=['t14', 'b13', 'k13', 'd10'], suptitle=False, cosmo=cosmo0, n_cut=2.5, d10_cut_by='sersic', equal_bins=False, nbins=8, xaxis='rvir', yaxis='reff', filename=''):
    """
    Plot the halo size v.s. GALFIT size using all SMHM relations in the same
    redshift bin. The Reff--Rvir relations are derived using one SMHM relation.

    Arguments:
    ----------
    c_gal_all: a dictionary of dataframes representing the galaxy catalog in
               each field
    zlo: the lower bound of the redshift bin
    dz: the redshift bin width
    fields: a list of field names
    H_maglim_wide: H-band magnitude limit in CANDELS Wide regions 
                   (default=24.5)
    zspec: whether to only use objects with spectroscopic redshifts or not
    alpha_scatter: the alpha value for the scatter plot showing all galaxies
                   (default=0.1)
    cut_by: whether to split sample by Sersic index (sersic) or sSFR (ssfr) or
            no cut (none; all galaxies are included)
    include: whether to include all galaxies (split by the property specified
             in "cut_by") or only include the tails of the distributions
             (n_10pc)
    logssfr_cut: the log(sSFR) value to cut the sample into star forming or
                 quiescent galaxies
    conversion: which SMHM relation to use (b13: Behroozi+13; d10: Dutton+10;
                k13: Kravtsov13)
    """    
    # Define Figure parameters
    fig = plt.figure(figsize=(16, 6))
    nrows = 1
    ncols = len(conversions)
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75), nrows_ncols=(nrows, ncols), 
        axes_pad=0.2, share_all=True, aspect=True, label_mode='L')

    z0 = zlo
    z1 = zlo + dz
    if z0 < 1.5:
        band = 'j'
        zp = 1.5
    else:
        band = 'h'
        zp = 2.2
    print("Working on {:.1f} < z < {:.1f}".format(z0, z1)) 
    c_bin = pd.DataFrame()
    for f in fields:
        print("Working on {}...".format(f.upper()))
        # To test how much UDF GALFIT filtering changes results
        if f == 'udf':
            c_gal_f = sg.select_galaxies(c_gal_all, f, z0, z1, cosmo,
                filter_galfit=True, H_maglim_wide=H_maglim_wide)
        else:
            c_gal_f = sg.select_galaxies(c_gal_all, f, z0, z1, cosmo,
                filter_galfit=True, H_maglim_wide=H_maglim_wide)
        c_bin = c_bin.append(c_gal_f, ignore_index=True)

    # Define variables
    sersic_n_bin = c_bin['n_'+band.lower()].values
    logssfr_bin = c_bin['logssfr_med'].values
    zbest = c_bin.zbest.values    
    # Also apply galaxy color gradient correction
    rcorr_disk = -0.35 + 0.12 * c_bin.zbest - 0.25 * (c_bin.m_med - 10.)
    # cap the maximum correction factor at 0
    rcorr_disk = np.minimum(rcorr_disk.values, 0.)
    rcorr_devauc = -0.25
    rcorr = np.where(sersic_n_bin < n_cut, rcorr_disk, rcorr_devauc)
    rcorr_factor = np.power((1. + zbest) / (1. + zp), rcorr)
    # R_eff_bin still in arcsec
    R_eff_bin = c_bin['re_' + band.lower()].values  # in arcsec
    R_eff_bin = R_eff_bin * rcorr_factor
    # Calculate angular diameter distance in kpc / arcsec
    dist_a = np.array([cosmo.angular_diameter_distance(z).value * 1.e3 / arcsec_per_rad for z in zbest])
    R_eff_kpc = R_eff_bin * dist_a
    logR_eff_bin = np.log10(R_eff_kpc)

    # write the binned median values into a dataframe
    if cut_by in ['sersic', 'ssfr']:
        alpha_med_late = pd.DataFrame()
        alpha_med_early = pd.DataFrame()
    else:
        alpha_med = pd.DataFrame()

    for i in range(len(conversions)):
        if conversions[i] == 't14':
            M_halo = 10.**c_bin.m_halo_t14.values  # in M_solar
        elif conversions[i] == 'b13':
            M_halo = 10.**c_bin.m200c_b13.values  # in M_solar
        elif conversions[i] == 'k13':
            M_halo = 10.**c_bin.m_halo_k13.values  # in M_solar
        elif conversions[i] == 'd10':
            print("Calculating halo mass using Dutton 2010 relation...")
            # Calculate here b/c we might be using different cuts for galaxy
            # types
            # gtype = 1 --> late-type
            # gtype = 0 --> early-type
            # gtype = 2 --> undefined
            if cut_by == 'none':
                if d10_cut_by == 'sersic':
                    # decide which SMHM relation to use using Sersic index
                    gtype = np.where(sersic_n_bin > n_cut, 0, 1)
                else:
                    print("Using log(sSFR) to determine whether to use the late-type SMHM relation or not...")
                    gtype = np.where(logssfr_bin > logssfr_cut, 1, 0)
            elif cut_by == 'sersic':
                if include == 'all':
                    n_low = n_cut
                    n_high = n_cut
                elif include == 'n_10pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_10pc = len(sersic_n_bin) / 10
                    print("n_10pc = ", n_10pc)
                    n_low = sersic_n_bin_sorted[n_10pc]
                    n_high = sersic_n_bin_sorted[-n_10pc]
                    print("n_low = {:.1f}, n_high = {:.1f}".format(n_low, n_high))
                gtype = np.where(sersic_n_bin > n_high, 0, 2)
                gtype = np.where(sersic_n_bin < n_low, 1, gtype)
            elif cut_by == 'ssfr':
                if include == 'all':
                    logssfr_low = logssfr_cut
                    logssfr_high = logssfr_cut
                elif include == 'n_10pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_10pc = len(logssfr_bin_sorted) / 10
                    print("n_10pc = ", n_10pc)
                    logssfr_low = logssfr_bin_sorted[n_10pc]
                    logssfr_high = logssfr_bin_sorted[-n_10pc]
                    print("ssfr_low = {:.2f}, ssfr_high = {:.2f}".format(logssfr_low, logssfr_high))
                gtype = np.where(logssfr_bin > logssfr_high, 0, 2)
                gtype = np.where(logssfr_bin < logssfr_low, 1, gtype)
            logM_halo = msh.calc_halo_mass_d10(c_bin.m_med.values, gtype)
            c_bin['logm_vir_d10'] = logM_halo
            M_halo = 10 ** logM_halo
        
        R_vir_bin = np.array(list(map(ms.R200, M_halo, zbest)))  # in kpc        
        logR_vir_bin = np.log10(R_vir_bin)
        # R_eff_bin = c_bin['re_' + band.lower()].values  # in arcsec
        # Decide which quantity to plot in both axes
        if xaxis == 'rvir':
            xdata = logR_vir_bin
            xlabel = r'$\log R_{\rm{vir}}$ [kpc]'
            bins = np.arange(1.0, 3.2, 0.2)
            xlims = [1.0, 3.5]
            xticks = np.arange(1.5, 3.5, 0.5)
        else:
            xdata = np.log10(0.015) + logR_vir_bin
            xlabel = r'$\log(0.015\times R_{\rm{vir}})$ [kpc]'
            bins = np.arange(-0.8, 1.4, 0.2)
            xlims = [-0.8, 1.5]
            xticks = np.arange(-0.5, 1.5, 0.5)
        if yaxis == 'reff':
            ydata = logR_eff_bin
            ylabel = r'$\log R_{\rm{eff}}$ [kpc]'
        else:
            ydata = np.where(sersic_n_bin > 2.5, np.log10(1.34) + logR_eff_bin,
                         logR_eff_bin)
            ylabel = r'$\log R_{1/2}$ [kpc]'
        if conversions[i] == 'b13':
            xdata0 = xdata.copy()
            ydata0 = ydata.copy()
            
        # if equal_bins:
        #     logR_vir_bin_sorted = np.sort(logR_vir_bin)
        #     binsize = len(logR_vir_bin) // nbins
        #     bins = logR_vir_bin_sorted[::binsize]
        #     if logR_vir_bin_sorted[-1] > bins[-1]:
        #         bins = np.concatenate([bins, [logR_vir_bin_sorted[-1]]])
        #     bin_centers = binned_statistic(logR_vir_bin, logR_vir_bin,
        #         statistic=np.median, bins=bins)[0]
        # else:
        #     bins = np.arange(1.0, 3.2, 0.2)
        bin_centers = (bins[:-1] + bins[1:]) / 2.
        # FIRST, make scatter plot
        if cut_by == 'none':
            filt = np.isnan(ydata)
            grid[i].scatter(xdata, ydata, edgecolor=GRAY,
                facecolor=GRAY, marker='^', s=8, alpha=alpha_scatter)
            log_alpha = logR_eff_bin - logR_vir_bin
            # calculate the number of sources per bin
            hist = np.histogram(xdata, bins=bins)[0]
            med, bin_edges, binnum = binned_statistic(xdata[~filt],
                ydata[~filt], statistic=np.median, bins=bins)
            std, bin_edges, binnum = binned_statistic(xdata[~filt],
                ydata[~filt], statistic=np.std, bins=bins)
            alpha_med['smhm-{}'.format(i+1)] = med
            alpha_med.index = bin_edges[:-1]
            grid[i].errorbar(bin_centers[hist>=5], med[hist>=5],
                yerr=std[hist>=5], fmt='s', ecolor=DARKGRAY, mfc=DARKGRAY,
                mec='none', ms=10, mew=1.5)
            grid[i].set_title('SMHM relation {}'.format(i+1))

        else:
            if cut_by == 'sersic':
                if include == 'all':
                    n_low = n_cut
                    n_high = n_cut
                    label1 = r'$n < %.1f$' % n_cut
                    label2 = r'$n \geq %.1f$' % n_cut
                elif include == 'n_10pc':
                    # Inlcud just the top and bottom 10% of the distribution
                    sersic_n_bin_sorted = np.sort(sersic_n_bin)
                    n_10pc = len(sersic_n_bin) / 10
                    print("n_10pc = ", n_10pc)
                    n_low = sersic_n_bin_sorted[n_10pc]
                    n_high = sersic_n_bin_sorted[-n_10pc]
                    label1 = r'Lowest 10% in $n$'
                    label2 = r'Highest 10% in $n$'
                xdata_disk = xdata[sersic_n_bin < n_low]
                ydata_disk = ydata[sersic_n_bin < n_low]
                xdata_devauc = xdata[sersic_n_bin > n_high]
                ydata_devauc = ydata[sersic_n_bin > n_high]
                print("N_disk, N_devauc = {}, {}".format(
                    np.sum(sersic_n_bin < n_cut), np.sum(sersic_n_bin > n_cut)))
            elif cut_by == 'ssfr':
                if include == 'all':
                    logssfr_low = logssfr_cut
                    logssfr_high = logssfr_cut
                    # passive_bin = (logssfr_bin <= logssfr_cut)
                    # active_bin = np.logical_not(passive_bin)
                    label1 = 'log(sSFR)>{:.1f}'.format(logssfr_cut)
                    label2 = 'log(sSFR)<{:.1f}'.format(logssfr_cut)
                elif include == 'n_10pc':
                    logssfr_bin_sorted = np.sort(logssfr_bin)
                    n_10pc = len(logssfr_bin_sorted) / 10
                    print("n_10pc = ", n_10pc)
                    logssfr_low = logssfr_bin_sorted[n_10pc]
                    logssfr_high = logssfr_bin_sorted[-n_10pc]
                    # passive_bin = (logssfr_bin < ssfr_low)
                    # active_bin = (logssfr_bin > ssfr_high)
                    label1 = 'Highest 10% in sSFR'
                    label2 = 'Lowest 10% in sSFR'
                xdata_disk = xdata[logssfr_bin > logssfr_high]
                ydata_disk = ydata[logssfr_bin > logssfr_high]
                xdata_devauc = xdata[logssfr_bin < logssfr_low]
                ydata_devauc = ydata[logssfr_bin < logssfr_low]
            # Make scatter plots
            grid[i].scatter(xdata_disk, ydata_disk, marker='x',
                s=8, alpha=alpha_scatter+0.05, edgecolor=LIGHTBLUE,
                facecolor=LIGHTBLUE)
            grid[i].scatter(xdata_devauc, ydata_devauc, marker='o',
                s=8, alpha=alpha_scatter+0.05, edgecolor=LIGHTRED,
                facecolor=LIGHTRED)
            # calculate the number of sources per bin
            hist1 = np.histogram(xdata_disk, bins=bins)[0]
            hist2 = np.histogram(xdata_devauc, bins=bins)[0]
            # Calculate binned statistics
            med_disk, bin_edges, binnum = binned_statistic(xdata_disk,
                ydata_disk, statistic=np.median, bins=bins)
            std_disk, bin_edges, binnum = binned_statistic(xdata_disk,
                ydata_disk, statistic=np.std, bins=bins)
            grid[i].errorbar(bin_centers[hist1>=5], med_disk[hist1>=5],
                yerr=std_disk[hist1>=5], fmt='s', ecolor=DARKBLUE,
                mfc=DARKBLUE, mec='none', ms=10, mew=1.2, label=label1)
            alpha_med_late['smhm{}-{}'.format(i+1, cut_by)] = med_disk
            alpha_med_late.index = bin_edges[:-1]
            med_devauc, bin_edges, binnum = binned_statistic(
                xdata_devauc, ydata_devauc, statistic=np.median,
                bins=bins)
            std_devauc, bin_edges, binnum = binned_statistic(
                xdata_devauc, ydata_devauc, statistic=np.std,
                bins=bins)
            grid[i].errorbar(bin_centers[hist2>=5], med_devauc[hist2>=5],
                yerr=std_devauc[hist2>=5], fmt='o', ecolor=DARKRED, mfc='none',
                mec=DARKRED, ms=10, mew=1.5, label=label2)
            alpha_med_early['smhm{}-{}'.format(i+1, cut_by)] = med_devauc
            alpha_med_early.index = bin_edges[:-1]
        # Plot a (theoretically moticated) average relation between halo size
        # and galaxy disk size
        logR_halo_grid = np.linspace(0., 4., 100)
        R_eff_avg = 0.035 / np.sqrt(2.) * 10.**logR_halo_grid * 1.678
        R_eff_K13 = 0.015 * 10.**logR_halo_grid #* 1.678
        if xaxis == 'rvir':
            xcons = logR_halo_grid
        else:
            xcons = np.log10(0.015) + logR_halo_grid

        if cut_by != 'none':
            grid[i].plot(xcons, np.log10(R_eff_avg), ls='--', lw=2,
                c=GREEN, label=r'$J$/$M$ equality')
        grid[i].plot(xcons, np.log10(R_eff_K13), ls='-', lw=1,
            c='black', label='Kravtsov13')
        
        # Plot the +/- 0.5 dex dotted line like Kravtsov did
        # grid[i].plot(xcons, np.log10(R_eff_K13) + 0.5, ls=':', lw=1, c='black')
        # grid[i].plot(xcons, np.log10(R_eff_K13) - 0.5, ls=':', lw=1, c='black')

        grid[i].tick_params(axis='both', which='major', labelsize='xx-large')
        if cut_by == 'none':
            fontsize = 'x-large'
        else:
            fontsize = 'large'
        legend = grid[i].legend(loc=4, markerscale=1, fontsize=fontsize,
                                frameon=False, markerfirst=False)
        grid[i].text(0.05, 0.95, r'$%.1f < z < %.1f$' % (z0, z1),
            ha='left', va='top', size='xx-large', transform=grid[i].transAxes)
        # grid[i].plot([2.0, 2.0], [-1.1, 1.6], c='Indigo')
        # grid[i].set_title(r'${}\leq z < {}$'.format(zlo[i], zhi[i]))
        if cut_by != 'none':
            grid[i].set_title('SMHM relation {}'.format(i+1))
        grid[i].set_xticks(xticks)
        grid[i].set_xlim(xlims)
        grid[i].set_ylim(-1, 1.5)
        grid[i].set_ylabel(ylabel, size=28)
        grid[i].set_xlabel(xlabel, size=28)
                     
    if cut_by != 'none':
        alpha_med_late.to_csv(
            'alpha_med_allmethods_late_{}.csv'.format(cut_by), index=True,
            header=True)
        alpha_med_early.to_csv(
            'alpha_med_allmethods_early_{}.csv'.format(cut_by), index=True,
            header=True)
    else:
        alpha_med.to_csv('alpha_med_allmethods.csv', index=True, header=True)
    # plt.tight_layout()
    if len(filename):
        fig.savefig(filename)
    return grid, xdata0, ydata0


def read_gal_all():
    """
    Read all galaxy catalogs.
    """
    c_gal_all = {}
    for field in ['gds', 'gdn', 'uds', 'cosmos', 'egs']:
        print("Reading galaxy catalog in {}...".format(field.upper()))
        c_gal_all[field] = rc.read_candels_cat(field + '_mass.cat')

    return c_gal_all


def read_halo_all(fields=['gds', 'gdn', 'uds', 'cosmos', 'egs', 'udf'], sim='MDPL2', zlo=['0p5', '1', '2']):
    """
    Read all DM halo catalogs.
    For now, read the MDPL2 halo catalogs.
    """
    c_halo_all = {}
    for f in fields:
        f = f.lower()
        if sim == 'MDPL2':
            if len(glob.glob('MDPL2/MDPL2_Rockstar_{}*.csv'.format(f))):
                c_halo_all[f] = []
            for i in range(len(zlo)):
                catalogs = glob.glob('MDPL2/MDPL2_Rockstar_{}_z_{}_*.csv'.format(f, zlo[i]))
                if len(catalogs) > 0:
                    # if there are more than one halo catalog in this field and
                    # redshift bin, randomly chooses one
                    cat = np.random.choice(catalogs)
                    print("Reading halo catalog in {} for zlo = {}...".format(f, zlo[i]))
                    c_halo_all[f].append(pd.read_csv(cat))
                else:
                    # append an empty data frame if can't find a halo catalog
                    # within this field in the given redshift range
                    c_halo_all[f].append(pd.DataFrame({}))
        else:
            raise NotImplementedError
    return c_halo_all


def select_galfit_sources(c_match, z_mid, cosmo):
    """
    Use GALFIT measurement quality check and return a boolean array that
    selects the galaxies with good GALFIT measurements within the redshift bin.
    """
    # Calculate angular diameter distance in kpc / arcsec
    dist_a = np.array([cosmo.angular_diameter_distance(z).value * 1.e3 / arcsec_per_rad for z in c_match.zbest])

    if z_mid >= 2.0 and z_mid < 3.0:
        band = 'H'
        gf_flag = reduce(np.logical_and, [c_match.M_med > 0,
            (c_match.dre_h / c_match.re_h) <= 0.3, c_match.f_h == 0, 
            c_match.n_h >= 0.1, c_match.n_h < 8,
            c_match.re_h >= 0.01])
        R_eff_kpc = (c_match.re_h * dist_a)[gf_flag].values
        sersic_n = c_match.n_h[gf_flag].values

    elif z_mid >= 1.0 and z_mid < 2.0:
        band = 'J'
        gf_flag = reduce(np.logical_and, [c_match.M_med > 0,
            (c_match.dre_j / c_match.re_j) <= 0.3, c_match.n_j >= 0.1,
            c_match.f_j == 0, c_match.n_j < 8, c_match.re_j >= 0.01])
        R_eff_kpc = (c_match.re_j * dist_a)[gf_flag].values
        sersic_n = c_match.n_j[gf_flag].values
    else:
        band = 'Y'
        gf_flag = reduce(np.logical_and, [c_match.M_med > 0,
            (c_match.dre_y / c_match.re_y) <= 0.3, c_match.n_y >= 0.1,
            c_match.f_y == 0, c_match.n_y < 8, c_match.re_y >= 0.01])
        R_eff_kpc = (c_match.re_y * dist_a)[gf_flag].values
        sersic_n = c_match.n_y[gf_flag].values
    
    return gf_flag, R_eff_kpc, sersic_n, band


def plot_halo_size_gal_size_mdpl2(c_gal, c_halo, zlo, zhi, field='gds-wide', ax=None, H_maglim=24.5, cosmo=Planck13, contour=True, plot_reln=True, sersic_cut=False, legend=True, scatter_kwargs={}):
    """
    Plot the halo virial radius vs. galaxy effective radius. Both are converted
    into kpc.
    The matching is done directly between galaxy catalog and N-body simulation
    catalogs. Galaxies are sorted by stellar mass and halos are sorted by halo
    (virial) mass.

    Arguments:
    ----------
    c_gal: a data frame containing the galaxies and their measured properties
    c_halo: a data frame containing the DM halos pulled from N-body simulations
    zlo: redshift lower bound
    zhi: redshift upper bound
    field: the galaxy survey field
    ax: the matplotlib axes instance; if None, create a new figure and axes
        (default: None)
    H_maglim: the H-band magnitude limit down to which galaxies are matched to
              DM halos
    cosmo: the astropy cosmology instance used for calculating distances
    contour: whether to calculate and plot galaxy number density contours
             (default: True)
    plot_reln: whether to plot the theoretically expected R_vir vs. R_disk
               scaling relation (default: True)

    Returns:
    --------
    R_vir_kpc: the sorted and matched halo virial radius in kpc (physical)
    R_eff_kpc: the sorted and matched galaxy effective radius in kpc (physical)
    """
    scatter_kwargs_def = dict(s=12, facecolor='DarkSeaGreen', marker='o',
                              edgecolor='none')
    for k in scatter_kwargs:
        scatter_kwargs_def[k] = scatter_kwargs[k]

    # First, match the galaxies with halos
    if field == 'gds-wide':
        c_gal = c_gal[np.logical_and(c_gal.udf_flag==0, c_gal.deep_flag==0)]
    elif field == 'gds-deep':
        c_gal = c_gal[c_gal.deep_flag==1]
    elif field == 'udf':
        c_gal = c_gal[c_gal.udf_flag==1]
    c_match = mgh.match_galaxy_halo(c_gal, c_halo, zlo, zhi, H_maglim=H_maglim)

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    bandnames = {'H': 'F160W', 'J': 'F125W', 'Y': 'F105W'}


    z_mid = (zlo + zhi) / 2.
    gf_flag, R_eff_kpc, sersic_n, band = select_galfit_sources(c_match,
        z_mid, cosmo)

    if np.sum(gf_flag) == 0:
        print("-" * 60)
        print("Something is wrong with the GALFIT measurements in {} for {} <= z < {}!!".format(field.upper(), zlo, zhi))
        print("-" * 60)
        return np.array([]), np.array([]), np.array([])


    logR_eff_kpc = np.log10(R_eff_kpc)    
    R_eff_min = logR_eff_kpc.min() - 0.5
    R_eff_max = logR_eff_kpc.max() + 0.5
    
    # convert halo virial radius into kpc (comoving --> proper??)
    # dividing by h gives the comoving size, and divide further by 1+z gives
    # the physical size
    R_vir_kpc = c_match.Rvir[gf_flag].values / (cosmo.h * (1. + c_match.z_halo[gf_flag].values))
    logR_halo = np.log10(R_vir_kpc)
    R_halo_min = logR_halo.min() - 0.5
    R_halo_max = logR_halo.max() + 0.5

    if sersic_cut:
        logR_halo_disk = logR_halo[sersic_n < 2.5]
        logR_eff_disk = logR_eff_kpc[sersic_n < 2.5]
        logR_halo_devauc = logR_halo[sersic_n >= 2.5]
        logR_eff_devauc = logR_eff_kpc[sersic_n >= 2.5]
        # Plot the disk-like galaxies
        scatter_kwargs_def['edgecolor'] = 'DodgerBlue'
        scatter_kwargs_def['marker'] = '^'
        scatter_kwargs_def['facecolor'] = 'none'
        if legend:
            scatter_kwargs_def['label'] = r'$n < 2.5$'
        ax.scatter(logR_halo_disk, logR_eff_disk, **scatter_kwargs_def)
        # Plot the elliptical-like galaxies
        scatter_kwargs_def['facecolor'] = 'Crimson'
        scatter_kwargs_def['edgecolor'] = 'none'
        scatter_kwargs_def['marker'] = 'o'
        if legend:
            scatter_kwargs_def['label'] = r'$ n \geq 2.5$'
        ax.scatter(logR_halo_devauc, logR_eff_devauc, **scatter_kwargs_def)
    else:
        ax.scatter(logR_halo, logR_eff_kpc, label=field.upper(),
                   **scatter_kwargs_def)
    ax.set_ylabel(r'%s $\log R_{\mathrm{eff}}\ [\mathrm{kpc}]$' % bandnames[band])
    ax.set_xlabel(r'$\log R_{\mathrm{vir}}\ [\mathrm{kpc}]$')
    ax.set_title(r'%s; $%.1f \leq z < %.1f$' % (field, zlo, zhi))

    if contour:
        # Now plot contours on top of the points
        xi = np.linspace(R_halo_min, R_halo_max, 30)
        yi = np.linspace(R_eff_min, R_eff_max, 30)
        # Bin the data
        num, xedges, yedges = np.histogram2d(logR_halo, logR_eff_kpc,
            bins=[xi, yi], normed=False)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
        # draw the contours
        if field == "udf":
            CS = ax.contour(num.T, 5, origin='lower', linewidths=1.0,
                        colors='darkred', extent=extent)
        else:
            CS = ax.contour(num.T, 5, origin='lower', linewidths=2.0,
                            colors='black', extent=extent)

    if plot_reln:
        # Plot a (theoretically moticated) average relation between halo size
        # and galaxy disk size
        R_halo_grid = np.linspace(0., 4., 100)
        # also remember that R_eff = 1.678 * R_d, where R_d is the exponential
        # disk scale length
        R_eff_avg = 0.032 * 10.**R_halo_grid * 1.678
        R_eff_K13 = 0.015 * 10.**R_halo_grid * 1.678
        ax.plot(R_halo_grid, np.log10(R_eff_avg), ls='--', lw=2, c='red',
                label='theory')
        ax.plot(R_halo_grid, np.log10(R_eff_avg) - 0.5, ls=':', lw=2,
            c='purple')
        ax.plot(R_halo_grid, np.log10(R_eff_avg) + 0.5, ls=':', lw=2,
            c='purple')
        ax.plot(R_halo_grid, np.log10(R_eff_K13), ls='-', lw=1, c='Indigo',
            label='Kravtsov13')
        ax.set_ylim(R_eff_min, R_eff_max)
        ax.set_xlim(R_halo_min, R_halo_max)

    # plt.xlim(9.5, 15.0)

    return R_vir_kpc, R_eff_kpc, sersic_n


def plot_Rvir_Reff_allz(c_gal_all, c_halo_all, zlo=[0.5, 1.0, 2.0], zhi=[1.0, 2.0, 3.0], fields=['gds-wide', 'gds-deep', 'udf', 'gdn', 'uds', 'cosmos', 'egs'], H_maglim=24.5, cosmo=Planck13, contour=False, figsize=(14, 6), sersic_cut=False, contour_alpha=0.1, contour_levels=3):
    """
    Plot R_vir vs. R_eff for all redshift bins in one given field.
    Note that if field == 'GDS', will also include HUDF galaxies (but the
    abundance matching between galaxies and halos for HUDF is done separately
    from the rest of GOODS-S).

    Arguments:
    ----------
    c_gal_all: a dictionary of data frames containing galaxies and their
               measured properties
    c_halo_all: a dictionary of lists of N data frames containing DM halos from
                simulations in each redshift bin, where N should be the number
                of redshift bins
    zlo: redshift lower bounds of each redshift bin (length = N)
    zhi: redshift upper bounds of each redshift bin (length = N)
    field: the survey field being considered (default: 'GDS')
    H_maglims: H-band magnitude limit for the given field. Note that for
               GOODS-S (GDS), the H-band magnitude limit in HUDF is
               H_maglim + 2.2 (hard coded).
    cosmo: the astropy cosmology instance for computing distances
    contour: whether to add number density contours on the plot. 
             (default: True)
    plot_reln: whether to plot the theoretically expected scaling relation
               between galaxy size and halo size (default: True)
    fig_size: figure size
    """
    nbins = len(zlo)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75), nrows_ncols=(1, nbins), 
        axes_pad=1.0, share_all=True, aspect=False, label_mode='all')
    print("fields = ", fields)
    # When showing contours only (when the data points have very low alpha),
    # don't put the survey fields in the legend

    data = {}
    for f in [x.lower() for x in fields]:
        data[f] = []
    print(list(data.keys()))

    for i in range(nbins):
        if (contour == True) and (contour_alpha <= 0.1):
            _legend = False
        else:
            _legend = True
        zlo_bin = zlo[i]
        zhi_bin = zhi[i]
        plot_reln = True
        for f in fields:
            print("Working on {}...".format(f.upper()))
            f2 = f.split('-')[0]
            if f == 'udf':
                scatter_kwargs = dict(marker='v', facecolor='LightCoral')
                if contour:
                    scatter_kwargs['alpha'] = contour_alpha
                c_gal = c_gal_all['gds']
                R_vir, R_eff, sersic_n = plot_halo_size_gal_size(c_gal,
                    c_halo_all['udf'][i], zlo_bin, zhi_bin, field="udf",
                    H_maglim=H_maglim+2.2, cosmo=cosmo, 
                    contour=False, ax=grid[i],
                    scatter_kwargs=scatter_kwargs, sersic_cut=sersic_cut,
                    legend=_legend, plot_reln=plot_reln)
                plot_reln = False
                if (sersic_cut == True) and (_legend == True):
                    _legend = False
            else:
                c_gal = c_gal_all[f2]
                if f.startswith('gds'):
                    scatter_kwargs = dict(marker='o', facecolor='DarkSeaGreen')
                elif f.startswith('gdn'):
                    scatter_kwargs = dict(marker='x', facecolor='DodgerBlue')
                elif f == 'cosmos':
                    # No Y-band data for COSMOS, EGS, and UDS
                    if i == 0:
                        data[f].append([[], [], []])
                        continue
                    scatter_kwargs = dict(marker='o', facecolor='none',
                        edgecolor='DarkCyan')
                elif f == 'uds':
                    if i == 0:
                        data[f].append([[], [], []])
                        continue
                    scatter_kwargs = dict(marker='^', facecolor='Orchid')
                elif f == 'egs':
                    if i == 0:
                        data[f].append([[], [], []])
                        continue
                    scatter_kwargs = dict(marker='s', edgecolor='Turquoise',
                        facecolor='none')
                if contour:
                    scatter_kwargs['alpha'] = contour_alpha
                print("Now plotting {} in bin {}...".format(f.upper(), i))
                if f.endswith('deep'):
                    R_vir, R_eff, sersic_n = plot_halo_size_gal_size(c_gal,
                        c_halo_all[f2][i], zlo_bin, zhi_bin, 
                        field=f, H_maglim=H_maglim+0.7, cosmo=cosmo,
                        contour=False, ax=grid[i],
                        scatter_kwargs=scatter_kwargs, sersic_cut=sersic_cut,
                        legend=_legend, plot_reln=plot_reln)
                else:
                    R_vir, R_eff, sersic_n = plot_halo_size_gal_size(c_gal,
                        c_halo_all[f2][i], zlo_bin, zhi_bin,
                        field=f, H_maglim=H_maglim, cosmo=cosmo, contour=False,
                        ax=grid[i], scatter_kwargs=scatter_kwargs,
                        sersic_cut=sersic_cut, legend=_legend,
                        plot_reln=plot_reln)
                plot_reln = False
                if (sersic_cut == True) and (_legend == True):
                    _legend = False
            data[f].append([R_vir, R_eff, sersic_n])
            
        if contour:
            # for f in fields:
            #     print f, len(data[f][-1])
            R_vir = reduce(lambda a, b: np.concatenate([a, b]),
                [data[f][-1][0] for f in fields])
            logR_vir = np.log10(R_vir)
            R_eff = reduce(lambda a, b: np.concatenate([a, b]),
                [data[f][-1][1] for f in fields])
            sersic_n_bin = reduce(lambda a, b: np.concatenate([a, b]),
                [data[f][-1][2] for f in fields])
            logR_eff = np.log10(R_eff)
            logR_vir_min = np.min(logR_vir)
            logR_vir_max = np.max(logR_vir)
            logR_eff_min = np.min(logR_eff)
            logR_eff_max = np.max(logR_eff)
            # derive the overall contour
            xi = np.linspace(logR_vir_min, logR_vir_max, 30)
            yi = np.linspace(logR_eff_min, logR_eff_max, 30)
            if sersic_cut:
                # Calculate and plot the contours for disks and devaucs
                # separately
                logR_vir_bin_disk = logR_vir[sersic_n_bin < 2.5]
                logR_eff_bin_disk = logR_eff[sersic_n_bin < 2.5]
                logR_vir_bin_devauc = logR_vir[sersic_n_bin >= 2.5]
                logR_eff_bin_devauc = logR_eff[sersic_n_bin >= 2.5]
                num1, xedges1, yedges1 = np.histogram2d(
                    logR_vir_bin_disk, logR_eff_bin_disk,
                    bins=[xi, yi], normed=False)
                extent1 = [xedges1[0], xedges1[-1], yedges1[0], yedges1[-1]]
                CS1 = grid[i].contour(num1.T, contour_levels, origin='lower',
                    linewidths=2.0, linestyles='solid', colors='black',
                    extent=extent1, label=r'$n < 2.5$')
                grid[i].plot([0., 0.], [0., 0.], ls='-', c='black', lw=2,
                    label=r'$n < 2.5$')
                num2, xedges2, yedges2 = np.histogram2d(
                    logR_vir_bin_devauc, logR_eff_bin_devauc,
                    bins=[xi, yi], normed=False)
                extent2 = [xedges2[0], xedges2[-1], yedges2[0], yedges2[-1]]
                CS2 = grid[i].contour(num2.T, contour_levels, origin='lower',
                    linewidths=1.0, linestyles='solid', colors='Crimson',
                    extent=extent2)
                grid[i].plot([0., 0.], [0, 0], ls='-', c='Crimson', lw=1,
                    label=r'$n \geq 2.5$')
            else:
                # Bin the data
                num, xedges, yedges = np.histogram2d(logR_vir, logR_eff,
                    bins=[xi, yi], normed=False)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            
                # draw the contours
                CS = grid[i].contour(num.T, 5, origin='lower', linewidths=2.0,
                    colors='black', extent=extent)
    
        grid[i].tick_params(axis='both', which='major', labelsize='large')
        grid[i].legend(loc=4, markerscale=2, fontsize='medium')
        grid[i].plot([2.0, 2.0], [-1.1, 1.6], c='Indigo')
        grid[i].set_title(r'${}\leq z < {}$'.format(zlo[i], zhi[i]))
        grid[i].set_xlim(1.3, 3.5)
        grid[i].set_ylim(-1.1, 1.6)

    grid[0].text(0.05, 0.95, 'Direct matching to MDPL2 sims', ha='left',
        va='top', transform=grid[0].transAxes, size='large')

    return data


def plot_Mvir_Mstar(c_gal, c_halo, zlo, zhi, field='gds-wide', ax=None, H_maglim=24.5, cosmo=Planck13, legend=True, scatter_kwargs={}):
    """
    Plot the halo virial mass vs. galaxy stellar mass, for diagnostic purposes.
    """
    scatter_kwargs_def = dict(s=12, facecolor='DarkSeaGreen', marker='o',
                              edgecolor='none')
    for k in scatter_kwargs:
        scatter_kwargs_def[k] = scatter_kwargs[k]

    # First, match the galaxies with halos
    if field == 'gds-wide':
        c_gal = c_gal[np.logical_and(c_gal.udf_flag==0, c_gal.deep_flag==0)]
    elif field == 'gds-deep':
        c_gal = c_gal[c_gal.deep_flag==1]
    elif field == 'udf':
        c_gal = c_gal[c_gal.udf_flag==1]
    c_match = mgh.match_galaxy_halo(c_gal, c_halo, zlo, zhi, H_maglim=H_maglim)

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    bandnames = {'H': 'F160W', 'J': 'F125W', 'Y': 'F105W'}

    z_mid = (zlo + zhi) / 2.
    gf_flag, R_eff_kpc, sersic_n, band = select_galfit_sources(c_match,
        z_mid, cosmo)

    if np.sum(gf_flag) == 0:
        print("-" * 60)
        print("Something is wrong with the GALFIT measurements in {} for {} <= z < {}!!".format(field.upper(), zlo, zhi))
        print("-" * 60)
        return np.array([]), np.array([]), np.array([])

    Mvir = c_match.Mvir[gf_flag]
    if field == 'uds':
        Mstar = c_match.M_med[gf_flag]
    else:
        Mstar = 10.**(c_match.M_med[gf_flag])

    Mvir_min = Mvir.min() / 2
    Mvir_max = Mvir.max() * 2
    Mstar_min = Mstar.min() / 2
    Mstar_max = Mstar.max() * 2
    
    ax.scatter(Mvir, Mstar, label=field.upper(), **scatter_kwargs_def)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_ylabel(r'$M_{\mathrm{stellar}}\ [M_{\odot}]$')
    ax.set_title(r'%s; $%.1f \leq z < %.1f$' % (field, zlo, zhi))
    ax.set_xlim(Mvir_min, Mvir_max)
    ax.set_ylim(Mstar_min, Mstar_max)

    return Mvir, Mstar, sersic_n


def plot_Mvir_Mstar_allz(c_gal_all, c_halo_all, zlo=[0.5, 1.0, 2.0], zhi=[1.0, 2.0, 3.0], fields=['gds-wide', 'gds-deep', 'udf', 'gdn', 'uds', 'cosmos', 'egs'], H_maglim=24.5, cosmo=Planck13, figsize=(14, 6)):
    """
    Plot virial mass vs. galaxy stellar mass for all redshift bins.
    """
    nbins = len(zlo)
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75), nrows_ncols=(1, nbins), 
        axes_pad=1.0, share_all=True, aspect=False, label_mode='all')
    print("fields = ", fields)

    data = {}
    for f in [x.lower() for x in fields]:
        data[f] = []
    Mvir_min = 1.e15
    Mvir_max = -1
    Mstar_min = 1.e15
    Mstar_max = -1

    for i in range(nbins):
        zlo_bin = zlo[i]
        zhi_bin = zhi[i]
        for f in fields:
            print("Working on {}...".format(f.upper()))
            f2 = f.split('-')[0]
            if f == 'udf':
                scatter_kwargs = dict(marker='v', facecolor='LightCoral')
                c_gal = c_gal_all['gds']
                Mvir, Mstar, sersic_n = plot_Mvir_Mstar(c_gal,
                    c_halo_all['udf'][i], zlo_bin, zhi_bin, field="udf",
                    H_maglim=H_maglim+2.2, cosmo=cosmo, ax=grid[i],
                    scatter_kwargs=scatter_kwargs, legend=True)

            else:
                c_gal = c_gal_all[f2]
                if f.startswith('gds'):
                    scatter_kwargs = dict(marker='o', facecolor='DarkSeaGreen')
                elif f.startswith('gdn'):
                    scatter_kwargs = dict(marker='x', facecolor='DodgerBlue')
                elif f == 'cosmos':
                    # No Y-band data for COSMOS, EGS, and UDS
                    if i == 0:
                        data[f].append([[], [], []])
                        continue
                    scatter_kwargs = dict(marker='o', facecolor='none',
                        edgecolor='DarkCyan')
                elif f == 'uds':
                    if i == 0:
                        data[f].append([[], [], []])
                        continue
                    scatter_kwargs = dict(marker='^', facecolor='Orchid')
                elif f == 'egs':
                    if i == 0:
                        data[f].append([[], [], []])
                        continue
                    scatter_kwargs = dict(marker='s', edgecolor='Turquoise',
                        facecolor='none')
                print("Now plotting {} in bin {}...".format(f.upper(), i))
                if f.endswith('deep'):
                    Mvir, Mstar, sersic_n = plot_Mvir_Mstar(c_gal,
                        c_halo_all[f2][i], zlo_bin, zhi_bin, 
                        field=f, H_maglim=H_maglim+0.7, cosmo=cosmo,
                        ax=grid[i], scatter_kwargs=scatter_kwargs,
                        legend=True)
                else:
                    Mvir, Mstar, sersic_n = plot_Mvir_Mstar(c_gal,
                        c_halo_all[f2][i], zlo_bin, zhi_bin,
                        field=f, H_maglim=H_maglim, cosmo=cosmo,
                        ax=grid[i], scatter_kwargs=scatter_kwargs,
                        legend=True)
            data[f].append([Mvir, Mstar, sersic_n])
            if len(Mvir) > 0:
                Mvir_min = np.minimum(Mvir.min(), Mvir_min)
                Mvir_max = np.maximum(Mvir.max(), Mvir_max)
                Mstar_min = np.minimum(Mstar.min(),
                    Mstar_min)
                Mstar_max = np.maximum(Mstar.max(),
                    Mstar_max)
    
    # Set plot limits
    for i in range(nbins):
        grid[i].plot([3.e10, 3.e10], [Mstar_min/1.5, Mstar_max*1.5], c='black')
        grid[i].set_xlim(Mvir_min / 1.5, Mvir_max * 1.5)
        grid[i].set_ylim(Mstar_min / 1.5, Mstar_max * 1.5)
        grid[i].legend(loc=4, fontsize='medium')
        grid[i].set_title(r'${:.1f} \leq z < {:.1f}$'.format(zlo[i], zhi[i]))


def plot_Mvir_hist(c_gal, c_halo, zlo, zhi, ax=None, H_maglim=24.5, bw=0.2, hist_kwargs={}):
    """
    Plot the M_vir histogram for the halos matched to the galaxies.
    """
    c_match = mgh.match_galaxy_halo(c_gal, c_halo, zlo, zhi, H_maglim=H_maglim)
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    Mvir_min = np.min(c_match.Mvir)
    Mvir_max = np.max(c_match.Mvir)
    logMvir_bins = np.arange(np.floor(np.log10(Mvir_min)),
        np.ceil(np.log10(Mvir_max)), bw)
    logMvir = np.log10(c_match.Mvir)

    ax.hist(logMvir, bins=logMvir_bins, histtype='step', **hist_kwargs)
    ax.set_xlabel(r'$M_{\mathrm{vir}}$')
    ax.set_ylabel('Number')

    return ax


def plot_Mvir_hist_all(c_gal, c_halo_list, field='GOODS-S', zlo=[0.5, 1.0, 2.0], zhi=[1.0, 2.0, 3.0], H_maglim=24.5, bw=0.2, hist_kwargs={}):
    """
    Plot the virial mass histograms for all redshift bins.
    """
    fig = plt.figure(figsize=(8, 12))
    nrows = len(zlo)
    assert len(c_halo_list) == len(zlo)
    axes = []

    for i in range(len(zlo)):
        ax = fig.add_subplot(nrows, 1, i+1)
        hist_kwargs['label'] = r'${}\leq z < {}$'.format(zlo[i], zhi[i])
        ax = plot_Mvir_hist(c_gal, c_halo_list[i], zlo[i], zhi[i], ax=ax,
                            H_maglim=H_maglim, bw=bw, 
                            hist_kwargs=hist_kwargs)
        ax.set_ylabel('Number')
        ax.legend(loc = 1)
        ax.set_xlabel('')
        if i == 0:
            ax.text(0.5, 0.95, field, ha='center', va='top', size='xx-large',
                    transform=ax.transAxes)

    ax.set_xlabel(r'$M_{\mathrm{vir}}$')
    plt.tight_layout()

    return axes