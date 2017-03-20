import os, sys, glob, pdb
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.cosmology import Planck13, FlatLambdaCDM
import sexdf
import mass_stellar2halo as msh
import halo_mass as hm

"""
Read the CANDELS catalogs of stellar mass, photometry, and GALFIT results.
"""

ARCMIN_PER_RAD = 3437.74677   # how many arcminutes in one radian
ARCSEC_PER_RAD = 180. / np.pi * 3600.  # how many arcsec are in 1 radian
MSOLAR_G = 1.98855e33   # solar mass in g
m_per_kpc = 3.0857e19   # how many meters in 1 kpc
cm_per_kpc = m_per_kpc * 1.e2  # hoe many cm in 1 kpc
CATDIR = '/Users/khuang/Dropbox/Research/CANDELS/catalogs'
HALO_DIR = '/Users/khuang/Dropbox/Research/2016b/Milli2'
area = {}
area['udf'] = 123. * 136. / 3600.  # a single WFC3/IR field of view
area['gds-deep'] = 68. - area['udf']
# area['gds-ers'] = 44.7
area['gds-wide'] = 147.51 - area['udf'] - area['gds-deep'] # - area['gds-ers']
area['gdn-deep'] = 68.
area['gdn-wide'] = 142.68 - area['gdn-deep']
area['uds'] = 9.4 * 22.0
area['cosmos'] = 9.0 * 22.4
area['egs'] = 6.4 * 30.5
area['gdn'] = 142.68
area['candels'] = 0.
for fld in ['udf', 'gds-deep', 'gds-wide', 'gdn-deep', 'gdn-wide', 'uds', 'cosmos', 'egs']:
    area['candels'] += area[fld]
cosmo0 = FlatLambdaCDM(H0=70., Om0=0.27)


def R200(M200c, z, cosmo=cosmo0):
    """
    Calculate the halo size R200 in kpc given a halo mass M200 in M_solar
    given a flat Lambda CDM model.
    """
    M200c_g = M200c * MSOLAR_G
    rho_crit = cosmo.critical_density(z).value
    R200c_cm = (M200c_g * 3 / (200 * 4 * np.pi * rho_crit))**(1./3.)
    R200c_kpc = R200c_cm / cm_per_kpc
    # return R200 in kpc; this is PHYSICAL SIZE
    return R200c_kpc


def Rvir(Mvir, z, cosmo=cosmo0):
    """
    Calculate the virial radius defined by the virial overdensity factor.
    Input Mvir is in units of solar mass (in LINEAR scale).
    """
    x = cosmo.Om(z) - 1.
    Delta_vir = 18 * np.pi**2 + 82. * x - 39. * x**2
    # convert virial mass to grams
    Mvir_g = Mvir * MSOLAR_G
    Rvir_cm = (3 * Mvir_g / (4. * np.pi * Delta_vir * cosmo.critical_density(z).value))**(1./3.)
    Rvir_kpc = Rvir_cm / cm_per_kpc

    return Rvir_kpc


def read_candels_cat(catalog, index_col=0):
    """
    Read the CANDELS catalogs.
    Assume that the first line of the catalog contains all the column names
    and the line starts with a # character.
    """
    f = open(catalog, 'r')
    line = f.readline()
    l = line.split()
    f.close()
    
    if l[0] != '#':
        print("Make sure the first line contains column names!")
        sys.exit()

    columns = l[1:]
    df = pd.read_table(catalog, sep=r' +', comment='#', names=columns,
        engine='python', index_col=index_col)

    return df


def read_stellar_mass(field='gds', index_col=0):
    """
    Read the stellar mass catalog.
    """
    if field == 'gds':
        catalog = CATDIR + '/goodss/CANDELS.GOODSS.F160W.v1.mass.cat'
    elif field == 'gdn':
        catalog = CATDIR + '/goodsn/CANDELS.GOODSN.F160W.v1.mass.cat'
    elif field == 'uds':
        catalog = CATDIR + '/uds/CANDELS.UDS.F160W.v1.mass.cat'
    elif field == 'cosmos':
        catalog = CATDIR + '/cosmos/CANDELS.COSMOS.F160W.v1_1.mass.cat'
    elif field == 'egs':
        catalog = CATDIR + '/egs/CANDELS.EGS.F160W.v1.mass.cat'
    else:
        raise NotImplementedError("Stellar mass catalog in {} not yet available.".format(field))

    return read_candels_cat(catalog, index_col=index_col)


def read_photometry(field='gds', index_col=0):
    """
    Read the photometric catalog.
    """
    # candels_dir = '/Volumes/raid0_kuang/candels/2016b/galaxy_halo_size'
    candels_dir = '/Users/khuang/Dropbox/Research/CANDELS/catalogs'
    if field == 'gds':
        catalog = candels_dir + '/goodss/gds.hdf5'
    elif field == 'gdn':
        catalog = candels_dir + '/goodsn/gdn.hdf5'
    elif field == 'uds':
        catalog = candels_dir + '/uds/uds.hdf5'
    elif field == 'egs':
        catalog = candels_dir + '/egs/egs.hdf5'
    elif field == 'cosmos':
        catalog = candels_dir + '/cosmos/cos.hdf5'
    # catalog = os.path.join(candels_dir, catalog)
    print("Reading {}".format(catalog))
    c  = pd.read_hdf(catalog, 'table')
    return c


def read_galfit(field='gds', band='f125w', index_col=0):
    """
    Read the GALFIT results.
    """
    if field == 'gds':
        catalog = CATDIR + '/goodss/CANDELS.GOODSS.F160W.v1.galfit.{}.cat'.format(band.lower())
    elif field == 'gdn':
        catalog = CATDIR + '/goodsn/CANDELS.GOODSN.F160W.v1.galfit.{}.cat'.format(band.lower())
    elif field == 'uds':
        catalog = CATDIR + '/uds/CANDELS.UDS.F160W.v1.galfit.{}.cat'.format(band.lower())
    elif field == 'cosmos':
        catalog = CATDIR + '/cosmos/CANDELS.COSMOS.F160W.v1.galfit.{}.cat'.format(band.lower())
    elif field == 'egs':
        catalog = CATDIR + '/egs/CANDELS.EGS.F160W.v1.galfit.{}.cat'.format(band.lower())
    else:
        raise NotImplementedError("GALFIT catalog in {} not yet available.".format(field))

    return read_candels_cat(catalog, index_col=index_col)


def merge_galfit_mass_catalogs(field='gds', cosmo=cosmo0, n_cut=2.5, zmax=3.0):
    """
    Merge the GALFIT catalog with stellar mass catalog.
    For GOODS-S, also flag the UDF objects.
    """
    print("Reading catalogs...")
    c_mass = read_photometry(field)
    # c_mass = read_stellar_mass(field=field)
    if field in ['gdn', 'gds']:
        bands = ['f105w', 'f125w', 'f160w']
    else:
        bands = ['f125w', 'f160w']
        
    # Columns to include from the stellar mass catalog
    mass_cols = ['ID', 'RAdeg', 'DECdeg', 'Hmag', 'PhotFlag', 'zbest',
        # 'M_med_lin', 's_med_lin', 'M_neb_med_lin', 's_neb_med_lin',
        'M_med', 's_med', 'M_neb_med', 's_neb_med', 'restUjohnson',
        'restVjohnson', 'restJ2mass', 'logsfr_med', 'zspec', 'q_zspec']
    if field in ['gds', 'uds']:
        mass_cols += ['H_SNR', 'CLASS_STAR', 'AGNFlag', 'zphot',
            'zphot_l68', 'zphot_u68', 'zphot_l95', 'zphot_u95']
        if field == 'gds':
            mass_cols.append('FLUX_RADIUS_2')
            mass_cols.append('udf_flag')
            mass_cols.append('deep_flag')
            mass_cols.append('WFC3_F160W_WEIGHT')
        else:
            mass_cols.append('flux_radius2_f160w')
    elif field == 'egs':
        mass_cols += ['CLASS_STAR', 'zphot', 'zphot_l68', 'zphot_u68',
            'zphot_l95', 'zphot_u95', 'FLUX_RADIUS_2_F160W']
    elif field == 'cosmos':
        mass_cols.append('FLUX_RADIUS_2_F160W')
    elif field == 'gdn':
        mass_cols.append('FLUX_RADIUS_2_F160W')
        mass_cols.append('Weight_F160W')
        mass_cols.append('deep_flag')
    if field == 'gds':
        # Also read the photometry catalog and figure out which objects are in
        # UDF
        print("Selecting UDF sources...")
        c_mass['udf_flag'] = np.where(c_mass['WFC3_F160W_WEIGHT'] >= 10.**4.4, 1, 0)
        c_mass['deep_flag'] = np.where((c_mass['WFC3_F160W_WEIGHT'] >= 10.**3.9) & (c_mass['WFC3_F160W_WEIGHT'] < 10.**4.4), 1, 0)
        # c_merged = c_merged.merge(c[['ID', 'udf_flag', 'deep_flag']], 
        #     left_index=True, right_on='ID')
    elif field == 'gdn':
        c_mass['H_SNR'] = c_mass['WFC3_F160W_FLUX'] / c_mass['WFC3_F160W_FLUXERR']
        c_mass['deep_flag'] = np.where((c_mass['Weight_F160W'] >= 10.**4.4),
            1, 0)
    elif field == 'cosmos':
        c_mass['H_SNR'] = c_mass['WFC3_F160W_FLUX'] / c_mass['WFC3_F160W_FLUXERR']
    elif field == 'egs':
        c_mass['H_SNR'] = c_mass['WFC3_F160W_FLUX'] / c_mass['WFC3_F160W_FLUXERR']
    c_merged = c_mass[mass_cols]
    if field == 'uds':
        # The original M_med column for UDS is in linear unit, not in log unit
        c_merged['M_med'] = np.log10(c_merged['M_med'])
        c_merged = c_merged.rename(columns={'flux_radius2_f160w': 'flux_radius_2'})
    elif field in ['gdn', 'cosmos', 'egs']:
        c_merged = c_merged.rename(columns={'FLUX_RADIUS_2_F160W': 'flux_radius_2'})
    # pdb.set_trace()

    c_gf = {}
    for b in bands:
        c_gf[b] = read_galfit(field=field, band=b)
    
    print("Merging catalogs...")
    for b in bands:
        c_merged = c_merged.merge(c_gf[b], right_index=True,
            left_on='ID')

    # Now impose the maximum redshift so we don't need to work on the sources
    # we don't use in this paper

    c_merged = c_merged[c_merged['zbest'] <= zmax]

    # Use the stellar mass-halo mass relation from Behroozi et al. 2013
    print("Calculating halo masses using Behroozi+2013 SMHM relation...")
    logMvir_b13, Mvir_hi_b13, Mvir_lo_b13 = msh.calc_halo_mass_b13_all(
        c_merged['M_med'].values, c_merged['zbest'].values, mass_def='vir')
    c_merged['Mvir_b13'] = logMvir_b13
    # c_merged['Mvir_b13_hi'] = Mvir_hi_b13
    # c_merged['Mvir_b13_lo'] = Mvir_lo_b13

    print("Also convert M_vir into M_200c")
    # M200c_b13, M200c_hi_b13, M200c_lo_b13 = msh.calc_halo_mass_b13_all(
    #     c_merged['M_med'].values, c_merged['zbest'].values, mass_def='200c')
    logM200c_b13 = msh.calc_halo_mass_b13_intp(c_merged['M_med'].values,
                                               c_merged['zbest'].values,
                                               mass_def='200c')
    logMvir_b13 = msh.calc_halo_mass_b13_intp(c_merged['M_med'].values,
                                               c_merged['zbest'].values,
                                               mass_def='vir')
    c_merged['M200c_b13'] = logM200c_b13
    c_merged['Mvir_b13'] = logMvir_b13
    # c_merged['M200c_b13_hi'] = M200c_hi_b13
    # c_merged['M200c_b13_lo'] = M200c_lo_b13

    # Calculate R200 from halo mass, but note that the Behroozi SMHM relation
    # returns VIRIAL MASS, not M200
    c_merged['R200c_Mvir_kpc_b13'] = R200(10.**logM200c_b13,
                                          c_merged['zbest'].values,
                                          cosmo=cosmo)
    # Calculate R200 from first converting Mvir into M200 assuming a halo
    # concentration c_vir=5., and then calculate R200 using M200
    c_merged['R200c_M200c_kpc_b13'] = R200(10.**logM200c_b13,
        c_merged['zbest'].values, cosmo=cosmo)
    
    # Calculate Rvir as used in Somerville+2017
    c_merged['Rvir_Mvir_kpc_b13'] = Rvir(10.**logMvir_b13,
                                         c_merged['zbest'].values,
                                         cosmo=cosmo)

    print("")

    print("Calculating halo masses using Kravtsov2013 SMHM relation...")
    # Also calculate halo mass based on Kravtsov 2013's SMHM relation
    logM200c_k13 = msh.calc_halo_mass_k13_intp(c_merged['M_med'].values)
    c_merged['M200c_k13'] = logM200c_k13
    c_merged['R200c_kpc_k13'] = R200(10.**logM200c_k13, c_merged.zbest.values,
                                     cosmo=cosmo)
    print("")

    # Also return the stellar mass-halo mass relation from Dutton et al. 2010,
    # MNRAS, 407, 2
    # Use F160W GALFIT Sersic index for classification
    # remember to scale by Hubble parameter h before input to halo mass
    # calculations (WHY???)
    print("Calculating halo masses using Dutton2010 SMHM relation...")
    gtypes = np.where(c_merged['n_h'] >= n_cut, 0, 1)
    # if Sersic index < 0 or > 10, set to a bad type (2)
    gtypes = np.where(
        np.logical_or(c_merged['n_h'] <= 0, c_merged['n_h'] > 10), 2, gtypes)
    logM_halo_d10 = msh.calc_halo_mass_d10(c_merged['M_med'] + np.log10(0.7),
        gtypes)
    c_merged['M_halo_d10'] = logM_halo_d10 - np.log10(0.7)
    c_merged['R200c_kpc_d10'] = R200(10.**logM_halo_d10,
                                     c_merged.zbest.values,
                                     cosmo=cosmo)

    print("Calculating halo masses using Rodriguez-Puebla+2015 SMHM relation...")
    # M200c_r15 = msh.calc_halo_mass_r15_all(c_merged['M_med'].values, gtypes,
    #     c_merged.zbest.values, mass_def='200c')
    logM200c_r15 = msh.calc_halo_mass_r15_intp(c_merged['M_med'].values,
                                               gtypes,
                                               c_merged['zbest'].values,
                                               mass_def='200c')
    c_merged['M200c_r15'] = logM200c_r15
    c_merged['R200c_kpc_r15'] = R200(10.**logM200c_r15, c_merged.zbest.values,
                                     cosmo=cosmo)
    print("")

    # Also calculate the halo mass from matching Millennium-II and SMF from
    # Tomczak et al. 2014
    print("Calculating halo masses using HMF from Millennium-II and SMF")
    print("from Tomczak et al. 2014...")
    logM200c_t14 = msh.calc_halo_mass_t14_intp(c_merged['M_med'].values,
        c_merged['zbest'].values)
    c_merged['M200c_t14'] = logM200c_t14
    c_merged['R200c_kpc_t14'] = R200(10.**logM200c_t14,
                                     c_merged['zbest'].values,
                                     cosmo=cosmo)
    print("")

    print("Calculating halo masses using HMF from Millennium-II and SMF")
    print("from our own Vmax calculation...")
    logM200c_vmax = msh.calc_halo_mass_vmax_intp(c_merged['M_med'].values,
        c_merged['zbest'].values)
    c_merged['M200c_vmax'] = logM200c_vmax
    c_merged['R200c_kpc_vmax'] = R200(10.**logM200c_vmax,
                                     c_merged['zbest'].values,
                                     cosmo=cosmo)
    print("")

    # Calculate halo mass using Hudson+2015 relation
    print("Calculating halo masses using Hudson+15 relations...")
    M_halo_h15 = msh.calc_halo_mass_hudson15_all(c_merged['M_med'].values,
                                                 gtypes)
    c_merged['M_halo_h15'] = M_halo_h15
    c_merged['R200c_kpc_h15'] = R200(10.**M_halo_h15, c_merged.zbest.values,
                                     cosmo=cosmo)

    # Calculate halo mass using Mandelbaum+2016 relation
    print("Calculating halo masses using Mandelbaum+16 relations...")
    M_halo_m16 = msh.calc_halo_mass_mandelbaum16_all(c_merged['M_med'].values,
                                                     gtypes,
                                                     c_merged['zbest'].values)
    c_merged['M_halo_m16'] = M_halo_m16
    c_merged['R200c_kpc_m16'] = R200(10.**M_halo_m16,
                                     c_merged['zbest'].values,
                                     cosmo=cosmo)

    # Now convert galaxy size onto rest-frame V-band using the color gradient
    # correction
    rcorr_disk = -0.35 + 0.12 * c_merged.zbest - 0.25 * (c_merged.M_med - 10.)
    rcorr_disk = np.where(c_merged.zbest > 3, -1.0, rcorr_disk)
    # cap the maximum correction factor at 0
    rcorr_disk = np.minimum(rcorr_disk, 0.)
    rcorr_devauc = -0.25
    sersic_n = np.where(c_merged.zbest < 1.5, c_merged.n_j, c_merged.n_h)
    c_merged['sersic_n'] = sersic_n
    rcorr = np.where(sersic_n < n_cut, rcorr_disk, rcorr_devauc)
    zp_all = np.where(c_merged.zbest < 1.5, 1.5, 2.2)
    rcorr_factor = np.power((1. + c_merged.zbest) / (1. + zp_all), rcorr)
    rcorr_factor = np.where(c_merged.zbest > 3.0, 0., rcorr_factor)
    # R_eff_arcsec still in arcsec
    R_eff_arcsec = np.where(c_merged.zbest < 1.5, c_merged.re_j, c_merged.re_h)
    R_eff_arcsec = R_eff_arcsec * rcorr_factor
    # Calculate angular diameter distance in kpc / arcsec
    dist_a = np.array([cosmo.angular_diameter_distance(z).value * 1.e3 / ARCSEC_PER_RAD for z in c_merged.zbest])
    R_eff_kpc = R_eff_arcsec * dist_a
    c_merged['R_eff_Vrest_kpc'] = R_eff_kpc

    # Calculate log(sSFR) in Gyr^-1
    c_merged['logssfr_med'] = c_merged['logsfr_med'] - c_merged['M_med'] + 9.

    # rename all columns to lower case!
    new_cols = {}
    for col in c_merged.columns:
        new_cols[col] = col.lower()
    c_merged = c_merged.rename(columns=new_cols)

    print("Writing to output...")
    if 'id' in c_merged.columns:
        # move id to the first column
        new_cols = ['id'] + [col for col in c_merged.columns if col != 'id']
        c_merged[new_cols].to_csv('{}_mass.cat'.format(field), sep=' ',
            index=False)
    else:
        c_merged = c_merged.rename({'Seq': 'id'})
        c_merged.to_csv('{}_mass.cat'.format(field), sep=' ', index=False)
        pass

    # put a hashtag sign in front of the first line
    with open('{}_mass.cat'.format(field)) as f:
        lines = f.readlines()
        lines[0] = '# ' + lines[0]
    f2 = open('{}_mass.cat'.format(field), 'w')
    for l in lines:
        f2.write(l)
    f2.close()

    print("Done!")


def read_gal_all():
    """
    Read all galaxy catalogs.
    """
    master_columns = ['radeg', 'decdeg', 'hmag', 'photflag', 'zbest', 'm_med',
        's_med', 'restujohnson', 'restvjohnson', 'restj2mass', 'logsfr_med',
        'flux_radius_2', 'ra_gfit_j', 'dec_gfit_j', 'f_j', 'mag_j', 'dmag_j',
        're_j', 'dre_j', 'n_j', 'dn_j', 'q_j', 'dq_j', 'pa_j', 'dpa_j', 'sn_j',
        'ra_gfit_h', 'dec_gfit_h', 'f_h', 'mag_h', 'dmag_h', 're_h',
        'dre_h', 'n_h', 'dn_h', 'q_h', 'dq_h', 'pa_h', 'dpa_h', 'sn_h',
        'mvir_b13', 'm200c_b13', 'logssfr_med',
        'udf_flag', 'deep_flag', 'm200c_k13', 'zspec', 'q_zspec',
        'm200c_t14', 'm200c_r15', 'r200c_m200c_kpc_b13', 'r200c_mvir_kpc_b13',
        'rvir_mvir_kpc_b13',
        'r200c_kpc_k13', 'r200c_kpc_d10', 'r200c_kpc_t14', 'r_eff_vrest_kpc',
        'sersic_n', 'm_halo_h15', 'r200c_kpc_h15', 'r200c_kpc_r15',
        'm_halo_m16', 'r200c_kpc_m16', 'm200c_vmax', 'r200c_kpc_vmax']
    c_gal_all = {}
    for field in ['gds', 'gdn', 'uds', 'cosmos', 'egs']:
        print("Reading galaxy catalog in {}...".format(field.upper()))
        c_field = read_candels_cat(field + '_mass.cat')
        if field != 'gds':
            c_field['udf_flag'] = 0
        if field not in ['gds', 'gdn']:
            c_field['deep_flag'] = 0
        c_gal_all[field] = c_field[master_columns]
        if 'agnflag' in c_field.columns:
            c_gal_all[field]['agnflag'] = c_field['agnflag']

    return c_gal_all


def read_halo_all(fields=['gds-deep', 'gds-wide', 'gdn', 'uds', 'cosmos', 'egs', 'udf'], zlo=['0.5', '1.0', '2.0']):
    """
    Read all DM halo catalogs from Millennium II.
    """
    sim_dir = '/Users/khuang/CANDELS/2016b/milli2'
    dtypes = {'haloId': np.int64, 'type': np.int8, 'snapnum': np.int16,
        'x': np.float32, 'y': np.float32, 'z': np.float32, 'mvir': np.float32,
        'rvir': np.float32, 'vmax': np.float32, 'stellarMass': np.float32,
        'redshift': np.float32}
    c_halo_all = {}
    for f in fields:
        f = f.lower()
        if f not in c_halo_all:
            c_halo_all[f] = []
        # if len(glob.glob(os.path.join(sim_dir, simcat))) == 0:
        #     c_halo_all[f] = []
        for i in range(len(zlo)):
            simcat = 'MRIIscPlanck1_{}_z_{}*.csv'.format(f, zlo[i])
            catalogs = glob.glob(os.path.join(sim_dir, simcat))
            if len(catalogs) > 0:
                # if there are more than one halo catalog in this field and
                # redshift bin, randomly chooses one
                cat = np.random.choice(catalogs)
                print("Reading {}...".format(os.path.split(cat)[-1]))
                c_halo_all[f].append(pd.read_csv(cat, dtype=dtypes, comment='#'))
            else:
                # append an empty data frame if can't find a halo catalog
                # within this field in the given redshift range
                c_halo_all[f].append(pd.DataFrame(columns=list(dtypes.keys())))
    return c_halo_all


def read_mi2_field(field, zlo, zhi):
    """
    Read all the halo catalogs from Millennium 2 simulation, for a given field
    within a given redshift bin.
    """
    c_halo_list = glob.glob(HALO_DIR + '/MRIIscPlanck1_{}_rand*_z_{:.1f}_{:.1f}.csv'.format(field.lower(), zlo, zhi))
    assert len(c_halo_list) > 0, "No halo catalogs found!!"
    return c_halo_list


def read_master_halo_catalog(zlo, zhi, ext='hdf5'):
    """
    Read the master halo catalog for a given redshift bin.
    """
    if ext == 'csv':
        catalog = 'MRIIscPlanck1_all_z_{:.1f}_{:.1f}.csv'.format(zlo, zhi)
        catalog = os.path.join('/Volumes/raid0_kuang/candels/2016b/milli2', catalog)
    elif ext == 'hdf5':
        catalog = 'MRIIscPlanck1_all_z_{:.1f}_{:.1f}.hdf5'.format(zlo, zhi)
    catalog = os.path.join('/Volumes/raid0_kuang/candels/2016b/milli2', catalog)
    assert os.path.exists(catalog), "Catalog {} does not exist!".format(catalog)
    if ext == 'csv':
        c_halo = pd.read_csv(catalog, comment='#')
    elif ext == 'hdf5':
        c_halo = pd.read_hdf(catalog, 'table')
    newcols = {}
    # pdb.set_trace()
    for col in c_halo.columns:
        newcols[col] = col.lower()
    c_halo = c_halo.rename(columns=newcols)
    return c_halo, catalog


def calc_spin_parameter(zlo, zhi, coosmo=Planck13):
    """
    For a given halo catalog, calculate the spin parameter lambda', where
    we use the alternative definition of lambda:

    lambda' = J / (sqrt(2) * M * V * R)

    or if specific angular momentum j is used, then 

    lambda' = j / (sqrt(2) * V * R)

    Also in Millennium II, both spin[xyz] and rvir are scaled by h, so h
    cancels out.
    """
    df, csv = read_master_halo_catalog(zlo, zhi, ext='csv')
    hdf5 = os.path.splitext(csv)[0] + '.hdf5'
    if os.path.exists(hdf5):
        print("Warining: file {} already exists!!".format(hdf5))
    df['j'] = np.sqrt(df['spinx']**2 + df['spiny']**2 + df['spinz']**2)
    df['lambda_p'] = df['j'] / (np.sqrt(2.) * df['vvir'] * df['rvir'])
    df.to_hdf(hdf5, 'table', format='table')

    return df


def select_halos(c_halo, fields, redshift_low, redshift_high, cosmo=Planck13, N=50000, verbose=False):
    """
    Select halos from the halo catalog with the same comoving volume as the
    the supplied survey field.
    """
    # calculate the transverse comoving distance at the midpoint redshift for
    # the given survey field geometry
    total_area = 0.
    for fld in fields:
        total_area += area[fld]
    volume = (cosmo.comoving_volume(redshift_high).value - cosmo.comoving_volume(redshift_low).value) / (4 * np.pi * ARCMIN_PER_RAD**2) * total_area
    if verbose:
        print("*" * 60)
        print("For {}...".format(fields))
        print("*" * 60)
        print("comoving volume = {:.2f} Mpc^3".format(volume))
    vfrac = volume / (100. / cosmo.h)**3
    if verbose:
        print("vfrac = {:.1f}% for {} between redshifts {:.1f} and {:.1f}\n".format(vfrac * 100, fields, redshift_low, redshift_high))

    # calculate the side of a square whose volume is the same as the total
    # volume of the surveyed field, in true comoving Mpc, when one side of
    # the rectangle is the same as the simulation box
    x = ((100 / cosmo.h)**2 * vfrac) ** (1./2)
    x_scaled = x * cosmo.h  # scaled by 
    # calculate the upper/lower limits of the center of the field, because
    # we need to leave enough room around the field center in order to include
    # enough comoving volume
    x0 = x_scaled / 2.
    x1 = 100. - x_scaled / 2.
    
    # randomly select the center of the volume, assuming it's rectangular
    xc = np.random.uniform(x0, x1)
    x_low = xc - x_scaled / 2.
    x_high = xc + x_scaled / 2.
    yc = np.random.uniform(x0, x1)
    y_low = yc - x_scaled / 2.
    y_high = yc + x_scaled / 2.
    # check the sampled comoving volumen within the (x, y) limits
    volume_sampled = (x_high - x_low) * (y_high - y_low) * 100 / cosmo.h**3
    if verbose:
        print("Sampled comoving volume within the limits is {:.2f} Mpc3".format(volume_sampled))
        print("Sampled volume agrees with the calculated volume to within {:.1f}% error.".format(100 * (volume_sampled - volume) / volume))

    # Now select halos
    select = (c_halo.x >= x_low) & (c_halo.x < x_high) & (c_halo.y >= y_low) & (c_halo.y < y_high)
    c_halo_out = c_halo[select].sort_values(by='mvir', ascending=False).iloc[:N]

    return c_halo_out
