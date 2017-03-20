import numpy as np
import pandas as pd
import catalogs as rc
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import Planck13
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.mlab import griddata
import select_galaxies as sg
import pdb
import mass_stellar2halo as msh
from functools import reduce
# import halo_mass as hm


cosmo0 = FlatLambdaCDM(H0=70., Om0=0.27)
Msolar_g = 1.98855e33   # solar mass in g
m_per_kpc = 3.0857e19   # how many meters in 1 kpc
cm_per_kpc = m_per_kpc * 1.e2  # hoe many cm in 1 kpc
arcsec_per_rad = 180. / np.pi * 3600.  # how many arcsec are in 1 radian
arcmin2_per_strad = (180. / np.pi * 60)**2  # how many arcmin^2 per steradian

BLUE = 'MediumBlue'
LIGHTBLUE = 'Blue'
DARKBLUE = 'DarkSlateBlue'
RED = 'Crimson'
LIGHTRED = 'HotPink'
DARKRED = 'DarkRed'
GREEN = 'ForestGreen'
LIGHTGREEN = 'LimeGreen'
GRAY = 'SlateGray'
DARKGRAY = 'DarkSlateGray'

# Read the number of galaxies in all bins/fields, as well as the number with
# n < 2.5
N_GAL_TOT = pd.read_csv('num_galaxies_tot.csv')  
N_GAL_DISK = pd.read_csv('num_galaxies_disk.csv')
# disk fraction as a function of redshift & field
DISKFRAC = N_GAL_DISK / N_GAL_TOT


def mass_size_vdw14(z, gtype='blue'):
    """
    Return two arrays that follow the parameterized fit to the stellar mass-
    size relation from van der Wel et al. 2014.
    The first returned array will be log(stellar mass) in M_solar, and the
    second array will be effective radius in kpc.
    Note that only the mean relation is implemented here, so not taking into
    account the scatter.
    The range of stellar mass in each redshift bin is eye-balled from Fig 8.
    The parameterization is 
    log(R_eff) = log(A) + alpha * (log10(M_stellar) - log10(5e10))

    Input Arguments:
    ----------------
    z: redshift
    gtype: galaxy type ('blue' or 'red')

    Returns:
    ----------------
    logm: log(stellar mass) [M_solar]
    logr: log(R_eff) [kpc]

    """
    assert gtype.lower() in ['red', 'blue', 'r', 'b']
    assert z < 3
    if gtype == 'blue':
        logm0, logm1 = 9.5, 11.5
    else:
        logm0, logm1 = 10.25, 11.5

    if 0 <= z < 0.5:
        if gtype == 'blue':
            # logm0, logm1 = 9.25, 11.5
            logA, alpha = 0.86, 0.25
        else:
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = 0.60, 0.75
    elif 0.5 <= z < 1:
        if gtype == 'blue':
            # logm0, logm1 = 9.25, 11.25
            logA, alpha = 0.78, 0.22
        else:
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = 0.42, 0.71
    elif 1 <= z < 1.5:
        if gtype == 'blue':
            # logm0, logm1 = 9.25, 11.25
            logA, alpha = 0.70, 0.22
        else:
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = 0.22, 0.76
    elif 1.5 <= z < 2:
        if gtype == 'blue':
            # logm0, logm1 = 9.25, 11.25
            logA, alpha = 0.65, 0.23
        else:
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = 0.09, 0.76
    elif 2 <= z < 2.5:
        if gtype == 'blue':
            # logm0, logm1 = 9.75, 11.25
            logA, alpha = 0.55, 0.22
        else:
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = -0.05, 0.76
    elif 2.5 <= z < 3:
        if gtype == 'blue':
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = 0.51, 0.18
        else:
            # logm0, logm1 = 10.25, 11.25
            logA, alpha = -0.06, 0.79
    logm = np.arange(logm0, logm1 + 0.25, 0.25)
    logr = logA + alpha * (logm - np.log10(5.e10))
    return logm, logr


def Reff_Rvir_vdw14(z, gtype='blue', conversion='t14'):
    """
    Converts the van der Wel+14 mass-size relation into Reff--Rvir relation.
    """
    logm_star, logr_eff = mass_size_vdw14(z, gtype=gtype)
    if conversion == 't14':
        logm_vir = [msh.calc_halo_mass_t14(m, z) for m in logm_star]
    elif conversion == 'b13':
        logm_vir = [msh.calc_halo_mass_b13(m, z)[0] for m in logm_star]
    elif conversion == 'k13':
        logm_vir = list(map(msh.calc_halo_mass_k13, logm_star))
    elif conversion == 'd10':
        gt = 1 if gtype == 'blue' else 0
        logm_vir = msh.calc_halo_mass_d10(logm_star, [gt] * len(logm_star))
    elif conversion == 'r15':
        gt = 1 if gtype == 'blue' else 0
        logm_vir = [msh.calc_halo_mass_r15(m, gt, z) for m in logm_star]
    logm_vir = np.array(logm_vir)
    logr_vir = np.log10(R200(10.**logm_vir, z))
    return logr_vir, logr_eff


def comoving_volume_zbin(z0, z1, area, cosmo=cosmo0):
    """
    Calculate the comoving volumn inside an area (in sq. arcmin) from redshifts
    z0 to z1.
    Return comoving volume in Mpc^3
    """
    volume_per_arcmin2 = (cosmo.comoving_volume(z1).value - cosmo.comoving_volume(z0).value) / (4. * np.pi * arcmin2_per_strad)
    return volume_per_arcmin2 * area


def read_mass_size(catalog='gds_mass.cat'):
    return rc.read_candels_cat(catalog, index_col=0)


def calc_H_maglim(field, maglim):
    """
    Calculate the H-band limiting magnitude given the field name and the
    limiting magnitude in the Wide portion of CANDELS.
    """
    if field.lower() == 'udf':
        return maglim + 2.2
    elif field.lower().endswith('deep'):
        return maglim + 0.7
    else:
        return maglim


def plot_galaxy_mass_size_all(c_gal_all, zlo=[0.5, 1.0, 2.0], zhi=[1.0, 2.0, 3.0], fields=['gds-wide', 'gds-deep', 'udf', 'gdn-wide', 'uds', 'cosmos', 'egs'], H_maglim=24.5, alpha=0.5):
    """
    Plot galaxy stellar mass vs. galaxy effective radii, to test if galaxy 
    matching is done properly.
    """
    fig = plt.figure(figsize=(14, 6))
    grid = ImageGrid(fig, (0.08, 0.15, 0.9, 0.75), nrows_ncols=(1, 3), 
        axes_pad=1.0, share_all=True, aspect=False, label_mode='all')
    data = {}
    # data[key] = [R_vir, R_eff, sersic_n], all in kpc
    data['0.5'] = [[], [], []]
    data['1.0'] = [[], [], []]
    data['2.0'] = [[], [], []]
    zbins = ['0.5', '1.0', '2.0']
    scatter_kwargs = dict(alpha=alpha)
    has_label = False
    plotted_vdw14 = False
    for f in fields:
        print("Working on {}...".format(f.upper()))

        if f.lower() == 'udf':
            df = c_gal_all['gds']
        else:
            df = c_gal_all[f.lower().split('-')[0]]
        H_maglim_f = H_maglim
        H_maglim_f = calc_H_maglim(f, H_maglim)
        for i in range(len(zlo)):
            zflag = sg.select_redshift(df, zlo[i], zhi[i])
            z_mid = (zlo[i] + zhi[i]) / 2.
            gfflag, R_eff_kpc, sersic_n, band = sg.select_galfit_sources(df,
                z_mid, cosmo0)
            flag = np.logical_and(zflag, gfflag)
            # Pick sources within sub-fields within GOODS-S (will also do this
            # for GOODS-N, but not for now)
            sflag = sg.select_subfield(df, f)
            flag = np.logical_and(flag, sflag)
            print("{} points selected for plotting between z = {:.1f} and {:.1f}.".format(np.sum(flag), zlo[i], zhi[i]))
    
            # late-type
            flag_disk = np.logical_and(flag,
                df['n_'+band.lower()] < 2.5).values
            flag_devauc = np.logical_and(flag,
                df['n_'+band.lower()] >= 2.5).values
            print("N_disk, N_devauc = {}, {}".format(np.sum(flag_disk), np.sum(flag_devauc)))
            zbest = df.zbest.values
            R_eff = np.maximum(df['re_'+band.lower()].values, 1.e-5)
            dist_a = np.array([cosmo0.angular_diameter_distance(z).value * 1.e3 / arcsec_per_rad for z in zbest])
            R_eff_kpc = R_eff * dist_a
            # effective radius in kpc
            logR_eff_kpc = np.log10(R_eff_kpc)
            if f == 'uds':
                smass = df.m_med.values
            else:
                smass = 10.**df.m_med.values
            # late-type:
            if not has_label:
                scatter_kwargs['label'] = 'n < 2.5'
            else:
                scatter_kwargs['label'] = ''
            scatter_kwargs['marker'] = 's'
            scatter_kwargs['facecolor'] = BLUE
            scatter_kwargs['edgecolor'] = 'none'
            grid[i].scatter(smass[flag_disk], R_eff_kpc[flag_disk],
                **scatter_kwargs)
            # pdb.set_trace()
            # early-type:
            if not has_label:
                scatter_kwargs['label'] = 'n > 2.5'
            else:
                scatter_kwargs['label'] = ''
            scatter_kwargs['marker'] = 'o'
            scatter_kwargs['facecolor'] = 'none'
            scatter_kwargs['edgecolor'] = 'Crimson'
            grid[i].scatter(smass[flag_devauc], R_eff_kpc[flag_devauc],
                **scatter_kwargs)
            if not has_label:
                has_label = True

            if not plotted_vdw14:
                # Plot the van der Well+2014 mass-size relations,
                # parameterized as
                # Reff[kpc] = A * (smass[m_solar] / 5e10) ^ alpha
                logsmass_grid = np.linspace(7, 12, 10)
                smass_grid = 10 ** logsmass_grid
                if 0.5 < z_mid < 1.0:
                    logA_disk = 0.78
                    alpha_disk = 0.22
                    logA_devauc = 0.42
                    alpha_devauc = 0.71
                elif 1.0 < z_mid < 2.0:
                    logA_disk = (0.70 + 0.65) / 2.
                    alpha_disk = (0.22 + 0.23) / 2.
                    logA_devauc = (0.22 + 0.09) / 2.
                    alpha_devauc = (0.76 + 0.76) / 2.
                else:
                    logA_disk = (0.55 + 0.51) / 2.
                    alpha_disk = (0.22 + 0.18) / 2.
                    logA_devauc = (-0.05 - 0.06) / 2.
                    alpha_devauc = (0.76 + 0.79) / 2.
                logR_disk = logA_disk + alpha_disk * (logsmass_grid - np.log10(5.e10))
                R_disk = 10. ** logR_disk
                logR_devauc = logA_devauc + alpha_devauc * (logsmass_grid - np.log10(5.e10))
                R_devauc = 10. ** logR_devauc
                grid[i].plot(smass_grid, R_disk, lw=2, ls='-', c='blue')
                grid[i].plot(smass_grid, R_devauc, lw=2, ls='--', c=GREEN)
        plotted_vdw14 = True

    for i in range(len(zlo)):
        grid[i].set_xscale('log')
        grid[i].set_yscale('log')
        grid[i].set_xlabel(r'Stellar Mass $[M_\odot]$')
        grid[i].set_ylabel('Effective Radius [kpc]')
        grid[i].set_title(r'$%.1f \leq z < %.1f$' % (zlo[i], zhi[i]))
        # grid[i].plot([2.0, 2.0], [-1.1, 1.6], c='Indigo')
        grid[i].set_xlim(5.e7, 1.e12)
        grid[i].set_ylim(0.1, 20)
        # grid[i].set_xticks(np.arange(1.5, 3.9, 0.5))
        grid[i].tick_params(axis='both', which='major', labelsize='large')
        grid[i].legend(loc=4, fontsize='medium')
    return grid


def plot_halo_mass_size(df, zlo, zhi, ax=None, field='GDS', sn_lolim=10., cosmo=cosmo0, zspec=False):
    """
    Plot the halo mass v.s. GALFIT size within a redshift range dz around z.
    """
    zflag = select_redshift(df, zlo, zhi, zspec=zspec, H_SNR_lim=5.0)
    flag = reduce(np.logical_and, [zflag, df.m_med > 0, df.sn_j >= sn_lolim,
        (df.dre_j / df.re_j) <= 0.3, df.n_j >= 0.1, df.n_j < 8,
        df.re_j >= 0.01])
    print("{} points selected for plotting.".format(np.sum(flag)))

    zbest = df.zbest[flag].values
    M_halo = 10.**df.M_halo_b13[flag].values
    logM_halo = np.log10(M_halo)
    R_eff = df.re_j[flag].values
    dist_a = np.array([cosmo.angular_diameter_distance(z).value * 1.e3 / arcsec_per_rad for z in zbest])
    R_eff_kpc = R_eff * dist_a  # effective radius in kpc
    logR_eff_kpc = np.log10(R_eff_kpc)
    # logRe = np.log10(R_eff)

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.scatter(logM_halo, logR_eff_kpc, s=6,
        edgecolor='none', facecolor='DarkSeaGreen')
    # ax.scatter(M_halo, logRe, s=8)

    # Now plot contours on top of the points
    xi = np.linspace(9., 15., 30)
    yi = np.linspace(-1., 2.5, 30)
    # Bin the data
    num, xedges, yedges = np.histogram2d(logM_halo, logR_eff_kpc,
        bins=[xi, yi], normed=False)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    # draw the contours
    CS = ax.contour(num.T, 5, origin='lower', linewidths=2.0, colors='black',
        extent=extent)

    if zspec:
        ax.text(0.05, 0.95, "spec-z only", transform=ax.transAxes,
            ha='left', va='top', size='medium')

    # Plot two lines to guide the eyes
    ax.plot([11., 11.], ax.get_ylim(), ls='--', c=GREEN, lw=1.5)
    ax.plot(ax.get_xlim(), [0., 0.], ls='--', c=GREEN, lw=1.5)

    ax.set_ylabel(r'$\log R_{\mathrm{eff}}\ [\mathrm{kpc}]$')
    ax.set_xlabel(r'$\log M_{\mathrm{vir}}\ [M_{\odot}]$')
    ax.set_title(r'%s; $%.1f \leq z < %.1f$' % (field, zlo, zhi))
    ax.set_xticks(np.arange(9, 16))
    # ax.set_xticks(10.**np.array([9., 11., 13., 15.]))
    ax.set_xlim(9., 15.0)

    return num, xedges, yedges


def plot_halo_mass_size_all(df, field='GDS', sn_lolim=10, zspec=False):
    """
    Plot the halo mass v.s. GALFIT size for three redshift bins:
    - 0.5 < z < 1.0
    - 1.0 < z < 2.0
    - 2.0 < z < 3.0
    """
    fig = plt.figure(figsize=(12, 6))
    grid = ImageGrid(fig, (0.1, 0.15, 0.85, 0.75), nrows_ncols=(1, 3), 
        axes_pad=0.1, share_all=True, aspect=False)

    flg1 = plot_halo_mass_size(df, 0.5, 1.0, ax=grid[0], field=field,
        sn_lolim=sn_lolim, zspec=zspec)
    flg2 = plot_halo_mass_size(df, 1.0, 2.0, ax=grid[1], field=field,
        sn_lolim=sn_lolim, zspec=zspec)
    flg3 = plot_halo_mass_size(df, 2.0, 3.0, ax=grid[2], field=field,
        sn_lolim=sn_lolim, zspec=zspec)
    grid[0].set_xticks(np.arange(9, 16))
    grid[0].set_xlim(9., 15.5)
    grid[0].set_ylim(-1.2, 2.5)

    return flg1, flg2, flg3


def plot_stellar_mass_size(df, zlo, zhi, field='GDS', sn_lolim=10., zspec=False):
    """
    Plot the stellar mass v.s. GALFIT size within a redshift range dz around z.
    """
    zflag = select_redshift(df, zlo, zhi, zspec=zspec, H_SNR_lim=5.0)
    flag = reduce(np.logical_and, [zflag, df.m_med > 0, df.sn_j >= sn_lolim,
        (df.dre_j / df.re_j) <= 0.3, df.n_j >= 0.1, df.n_j < 8,
        df.re_j >= 0.01])
    print("{} points selected for plotting.".format(np.sum(flag)))

    M_stell = df.m_med[flag].values
    logRe = np.log10(df.re_j[flag].values)

    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    ax.scatter(M_stell, logRe, s=8)
    ax.set_ylabel('log(Re) [arcsec]')
    ax.set_xlabel('log(M_stellar) [M_solar]')
    ax.set_title(r'%s; $%.1f \leq z < %.1f$' % (field, zlo, zhi))
    ax.set_xlim(7.5, 12.0)

    return flag


def M200(R200, z, cosmo=cosmo0):
    """
    Calculate the halo mass M_200 from the halo size R_200 (in kpc) and a
    given Flat LCDM cosmology model (cosmo).
    Input R200 in kpc and return M200 in solar mass.
    Converts everything in cgs unit before doing calculations. Note that
    astropy returns critical density at a given redshift in g / cm3.
    Use the equation
    M200 = 200 * rho_crit * (4*pi/3) * R200**3
    """
    R200_cm = R200 * cm_per_kpc

    # calculate critical density of the universe in g / cm3
    rho_crit = cosmo.critical_density(z)
    M200_g = 200 * rho_crit.value * (4 * np.pi / 3.) * R200_cm**3

    # convert M200 from g to M_solar
    M200 = M200_g / Msolar_g

    return M200


def R200(Mvir, z, cosmo=cosmo0, mass_def='200c'):
    """
    Calculate the halo size R200 in kpc given a halo mass M200 in M_solar
    given a flat Lambda CDM model.
    This is the inverse function of M200.
    Note that depending on different definition of Mvir, R200 will be
    different! If 
    mass_def = '200c': mean density within radius = 200 * critical density
    mass_def = 'vir': mean density within radius = delta_vir * critical density
    mass_def = '200m': mean density within radius = 200 * rho_mean_matter
    """
    Mvir_g = Mvir * Msolar_g
    rho_crit = cosmo.critical_density(z).value
    if mass_def == '200c':
        R200c_cm = (Mvir_g * 3 / (200 * 4 * np.pi * rho_crit))**(1./3.)
    R200c_kpc = R200c_cm / cm_per_kpc
    # return R200 in kpc; this is PHYSICAL SIZE
    return R200c_kpc


# def select_halo_size_gal_size(df, zlo, zhi, ax=None, field='gds-wide', sn_lolim=10., cosmo=cosmo0, zspec=False, H_maglim=24.5, plot_reln=True, contour=True, cut_by='sersic', include='all', logssfr_cut=-0.7, legend=True, conversion='b13', rcorr=True, scatter_kwargs={}, plot=True, alpha=0.1):
#     """
#     Plot halo size v.s. galaxy size by using the halo mass--stellar mass
#     relation derived from abundance matching (and not from directly matching
#     to DM simulations).
#     Convert halo mass to halo size using the redshift of each galaxy.
#     conversion: either 'b13' (Behroozi+2013) or 'd10' (Dutton+2010)
#     """
#     assert conversion in ['b13', 'd10']
#     scatter_kwargs_def = dict(s=2, facecolor='DarkSeaGreen', marker='o',
#                               edgecolor='none')
#     for k in scatter_kwargs:
#         scatter_kwargs_def[k] = scatter_kwargs[k]
#     bandnames = {'H': 'F160W', 'J': 'F125W', 'Y': 'F105W'}

#     zflag = sg.select_redshift(df, zlo, zhi, zspec=zspec, H_maglim=H_maglim)
#     output = sg.select_galfit_sources(df, (zlo + zhi) / 2., cosmo)
#     gf_flag = output[0]
#     band = output[3]
#     sub_flag = sg.select_subfield(df, field)
#     flag = np.array(reduce(np.logical_and, [zflag, gf_flag, sub_flag]))
#     print "{} points selected for plotting.".format(np.sum(flag))

#     if conversion == 'b13':
#         M_halo = 10.**df.m_halo_b13[flag].values
#     elif conversion == 'd10':
#         M_halo = 10.**df.m_halo_d10[flag].values
#     zbest = df.zbest[flag].values
#     sersic_n = df['n_'+band.lower()][flag].values
#     R_halo = np.array(map(R200, M_halo, zbest))  # in kpc
#     logR_halo = np.log10(R_halo)
#     R_eff = df['re_' + band.lower()][flag].values  # in arcsec
#     logssfr = df['logssfr_med'][flag].values

#     # Calculate angular diameter distance in kpc / arcsec
#     dist_a = np.array(map(lambda z: cosmo.angular_diameter_distance(z).value * 1.e3 / arcsec_per_rad, zbest))
    
#     # Convert galaxy effective radius into kpc
#     R_eff_kpc = R_eff * dist_a
#     if zlo < 2.0:
#         # for redshift bins below z=2, use J-band (F125W) sizes
#         band = 'J'
#         zp = 1.5   # the pivot redshift for color gradient correction
#     else:
#         band = 'H'
#         zp = 2.2
#     if rcorr:
#         # Also correct for color gradient -- follow the procedure outlined by
#         # van der Wel+2014, Section 2
#         rcorr_disk = -0.35 + 0.12 * zbest - 0.25 * (df.m_med[flag] - 10.)
#         # cap the maximum correction factor at 0
#         rcorr_disk = np.minimum(rcorr_disk.values, 0.)
#         rcorr_devauc = -0.25
#         R_eff_kpc = np.where(sersic_n < 2.5,
#             R_eff_kpc * np.power((1. + zbest) / (1 + zp), rcorr_disk),
#             R_eff_kpc * np.power((1. + zbest) / (1 + zp), rcorr_devauc))
#         # pdb.set_trace()
#     logR_eff_kpc = np.log10(R_eff_kpc)
#     logR_eff_min = -1.0
#     logR_eff_max = 1.5
#     if len(R_eff_kpc) > 0:
#         logR_eff_min = logR_eff_kpc.min() - 0.5
#         logR_eff_max = logR_eff_kpc.max() + 0.5
#     else:
#         return [[], [], []]

#     # if ax == None:
#     #     fig = plt.figure()
#     #     ax = fig.add_subplot(111)

#     if cut_by == 'none':
#         ax.scatter(logR_halo, logR_eff_kpc, edgecolor=GRAY, facecolor=GRAY,
#             marker='^', s=1, alpha=alpha)
#     else:
#         if cut_by == 'sersic':
#             if include == 'all':
#                 n_low = 2.5
#                 n_high = 2.5
#                 label1 = r'$n < 2.5$'
#                 label2 = r'$n \geq 2.5$'
#             elif include == 'n_10pc':
#                 # Inlcud just the top and bottom 10% of the distribution
#                 sersic_n_bin_sorted = np.sort(sersic_n_bin)
#                 n_10pc = len(sersic_n) / 10
#                 print "n_10pc = ", n_10pc
#                 n_low = sersic_n_bin_sorted[n_10pc]
#                 n_high = sersic_n_bin_sorted[-n_10pc]
#                 print "n_low = {:.1f}, n_high = {:.1f}".format(n_low, n_high)
#                 label1 = r'Lowest 10% in $n$'
#                 label2 = r'Highest 10% in $n$'
#             logR_vir_disk = logR_halo[sersic_n < n_low]
#             logR_eff_disk = logR_eff_kpc[sersic_n < n_low]
#             logR_vir_devauc = logR_halo[sersic_n >= n_high]
#             logR_eff_devauc = logR_eff_kpc[sersic_n >= n_high]
#             print "N_disk, N_devauc = {}, {}".format(np.sum(sersic_n < 2.5), np.sum(sersic_n > 2.5))

#         elif cut_by == 'ssfr':
#             if include == 'all':
#                 passive_bin = (logssfr_bin <= logssfr_cut)
#                 active_bin = np.logical_not(passive_bin)
#                 label1 = 'log(sSFR)>{:.1f}'.format(logssfr_cut)
#                 label2 = 'log(sSFR)<{:.1f}'.format(logssfr_cut)
#             elif include == 'n_10pc':
#                 logssfr_bin_sorted = np.sort(logssfr_bin)
#                 n_10pc = len(logssfr_bin_sorted) / 10
#                 print "n_10pc = ", n_10pc
#                 ssfr_low = logssfr_bin_sorted[n_10pc]
#                 ssfr_high = logssfr_bin_sorted[-n_10pc]
#                 print "ssfr_low = {:.2f}, ssfr_high = {:.2f}".format(ssfr_low, ssfr_high)
#                 passive_bin = (logssfr_bin < ssfr_low)
#                 active_bin = (logssfr_bin > ssfr_high)
#                 label1 = 'Highest 10% in sSFR'
#                 label2 = 'Lowest 10% in sSFR'
#             logR_vir_disk = logR_halo[logssfr >= logssfr_cut]
#             logR_eff_disk = logR_eff_kpc[logssfr >= logssfr_cut]
#             logR_vir_devauc = logR_halo[logssfr < logssfr_cut]
#             logR_eff_devauc = logR_eff_kpc[logssfr < logssfr_cut]
#         # Plot the disk-like galaxies
#         scatter_kwargs_def['edgecolor'] = BLUE
#         scatter_kwargs_def['marker'] = '^'
#         scatter_kwargs_def['facecolor'] = BLUE
#         scatter_kwargs_def['s'] = 1
#         scatter_kwargs_def['alpha'] = alpha
#         # if legend:
#         #     if cut_by == 'sersic':
#         #         scatter_kwargs_def['label'] = r'$n < 2.5$'
#         #     elif cut_by == 'ssfr':
#         #         scatter_kwargs_def['label'] = 'log(sSFR)>{:.1f}'.format(logssfr_cut)
#         ax.scatter(logR_vir_disk, logR_eff_disk, **scatter_kwargs_def)
#         # Plot the elliptical-like galaxies
#         scatter_kwargs_def['facecolor'] = RED
#         scatter_kwargs_def['edgecolor'] = RED
#         scatter_kwargs_def['marker'] = 'o'
#         scatter_kwargs_def['s'] = 2
#         # if legend:
#         #     if cut_by == 'sersic':
#         #         scatter_kwargs_def['label'] = r'$ n \geq 2.5$'
#         #     elif cut_by == 'ssfr':
#         #         scatter_kwargs_def['label'] = 'log(sSFR)<{:.1f}'.format(logssfr_cut)
#         ax.scatter(logR_vir_devauc, logR_eff_devauc, **scatter_kwargs_def)

#     # else:
#     #     ax.scatter(logR_halo, logR_eff_kpc, label=field.upper(), 
#     #         **scatter_kwargs_def)
#     ax.set_ylabel(r'$\log R_{\mathrm{eff}}\ [\mathrm{kpc}]$', size=28)
#     ax.set_xlabel(r'$\log R_{\mathrm{vir}}\ [\mathrm{kpc}]$', size=28)
#     ax.set_title(r'$%.1f \leq z < %.1f$' % (zlo, zhi))

#     # Now plot contours on top of the points
#     xi = np.linspace(1., 3.0, 30)
#     yi = np.linspace(-1., 2.5, 30)
#     # Bin the data
#     num, xedges, yedges = np.histogram2d(logR_halo, logR_eff_kpc,
#         bins=[xi, yi], normed=False)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

#     # draw the contours
#     if contour:
#         CS = ax.contour(num.T, 5, origin='lower', linewidths=2.0,
#             colors='black', extent=extent)

#     if zspec:
#         ax.text(0.05, 0.95, "spec-z only", transform=ax.transAxes,
#             ha='left', va='top', size='medium')

#     if plot_reln:
#         # Plot a (theoretically moticated) average relation between halo size
#         # and galaxy disk size
#         R_halo_grid = np.linspace(0., 4., 100)
#         R_eff_avg = 0.035 * 10.**R_halo_grid * 1.678 / np.sqrt(2.)
#         R_eff_K13 = 0.015 * 10.**R_halo_grid #* 1.678
#         if cut_by != 'none':
#             ax.plot(R_halo_grid, np.log10(R_eff_avg), ls='--', lw=2, c=GREEN,
#                 label=r'$J/M$ conserved')
#         # ax.plot(R_halo_grid, np.log10(R_eff_avg) - 0.5, ls=':', lw=2,
#         #     c=LIGHTGREEN)
#         # ax.plot(R_halo_grid, np.log10(R_eff_avg) + 0.5, ls=':', lw=2,
#         #     c=LIGHTGREEN)
#         ax.plot(R_halo_grid, np.log10(R_eff_K13), ls='-', lw=1, c='black',
#             label='Kravtsov13')

#     ax.set_xlim(1.0, 3.5)
#     ax.set_ylim(-1.1, 1.6)
#     ax.legend(loc=4, markerscale=2, fontsize='x-large')
#     plt.draw()
#     # plt.xlim(9.5, 15.0)

#     return R_halo, R_eff_kpc, sersic_n, logssfr


