"""A collection of methods to generating astrophysical populations"""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from cosmic-cats import utils

__all__ = ['get_star_3d_positions', 'get_sfh_stars', 'metallicity_dependent_binary_fraction',
           'metallicity_dependent_single_fraction', 'match_metallicities', 'get_simulated_data_stats',
           'get_formation_efficiency', 'get_simulated_matches', 'filter_sim_set',
           'get_evolved_systesm', 'sample_stars', 'connect_simulations_to_stars']

def get_star_3d_positions(stars):
    """Uses astropy to get cartesian Galactocentric coordinates and
    distances to stars of interest

    Parameters
    ----------
    stars : `DataFrame`
        Dataset containing radii

    Returns
    -------
    stars : `DataFrame`
        Input dataset with added columns for cartesion Galactocentric
        coordinates and distance
    """

    phi = np.random.uniform(0, 2*np.pi, len(stars))
    theta = theta = np.pi - np.arccos(np.random.uniform(-1, 1, len(stars)))
    
    stars['X'] = stars['R'] * np.cos(phi) * np.sin(theta)
    stars['Y'] = stars['R'] * np.sin(phi) * np.sin(theta)
    stars['Z'] = stars['R'] * np.cos(theta)
    c = SkyCoord(x=np.array(stars.X) * u.kpc,
                 y=np.array(stars.Y) * u.kpc, 
                 z=np.array(stars.Z) * u.kpc,
                 frame=coord.Galactocentric)
    stars['dist'] = c.transform_to(coord.ICRS).distance.to(u.kpc)
    
    return stars

def get_sfh_stars(sfh_model):
    """Generates a dataset of stars with positions, ages, and metallicities
    according to your model of choice

    Parameters
    ----------
    sfh_model : `str`
        model assumed for stellar ages and positions as a funciton of metallicity
        current models include:
        'Frankel19' : positions and ages from Frankel+2019

    Returns
    -------
    star_sample : `DataFrame`
        dataset of stars with positions, ages, and metallicities
        according to specified model
    """

    if sfh_model == 'Frankel19':
        sfh_read = './2021-02-16_Breivik_mockmw.fits'
        star_sample = Table.read(sfh_read).to_pandas()
        star_sample = get_star_3d_positions(stars=star_sample)
    
        star_sample['met_stars'] = utils.get_Z_from_FeH(star_sample['FeH'], Z_sun=0.014)
    else:
        raise ValueError("We only support sfh_model='Frankel19' at this time. Sorry bout it!")

    return star_sample

def metallicity_dependent_binary_fraction(met):
    """Computes the binary, and single by way of only having 
    single stars and binary systems, fraction of a population
    with Z=met following Moe+2019
    
    Parameters
    ----------
    met : `float`
        metallicity of the population
        

    Returns
    -------
    f_b : `float`
        binary fraction

    f_s : `float`
        single fraction
    """
    Fe_H = get_FeH_from_Z(met, Z_sun=0.014)
    if type(met) == float:
        if Fe_H <= -1.0:
            f_b = -0.0648 * Fe_H + 0.3356
        else:
            f_b = -0.1977 * Fe_H + 0.2025
    else:
        f_b = np.zeros(len(Fe_H))
        ind_lo, = np.where(Fe_H <= -1.0)
        ind_hi, = np.where(Fe_H > -1.0)

        if len(ind_lo) > 0:
            f_b[ind_lo] = -0.0648 * Fe_H[ind_lo] + 0.3356
        if len(ind_hi) > 0:
            f_b[ind_hi] = -0.1977 * Fe_H[ind_hi] + 0.2025
            
    f_s = 1 - f_b
    
    return f_b
    
def metallicity_dependent_single_fraction(met):
    """Computes the binary, and single by way of only having 
    single stars and binary systems, fraction of a population
    with Z=met following Moe+2019
    
    Parameters
    ----------
    met : `float`
        metallicity of the population
        

    Returns
    -------
    f_b : `float`
        binary fraction

    f_s : `float`
        single fraction
    """
    Fe_H = get_FeH_from_Z(met, Z_sun=0.014)
    if type(met) == float:
        if Fe_H <= -1.0:
            f_b = -0.0648 * Fe_H + 0.3356
        else:
            f_b = -0.1977 * Fe_H + 0.2025
    else:
        f_b = np.zeros(len(Fe_H))
        ind_lo, = np.where(Fe_H <= -1.0)
        ind_hi, = np.where(Fe_H > -1.0)

        if len(ind_lo) > 0:
            f_b[ind_lo] = -0.0648 * Fe_H[ind_lo] + 0.3356
        if len(ind_hi) > 0:
            f_b[ind_hi] = -0.1977 * Fe_H[ind_hi] + 0.2025
            
    f_s = 1 - f_b
    
    return f_s
    

def match_metallicities(met_list, met_stars):
    """Matches the metallicities of Neige's star samples to
    the metallcity bins of Katie's simulated binary populations
    such that every stellar metallicity is assigned one-to-one
    to the closest metallicity bin

    Parameters
    ----------
    met_list : list
        List of metallicity bins for simulated binary population

    met_stars : array
        Array of metallicities from stars sampled from Frankel disk model

    Returns
    -------
    inds : array
        Array giving the index of met_list for each of met_stars to make a
        one-to-one match between the two.
    """
    diff = []
    for met in met_list:
        diff.append(np.abs(np.array(met_stars) - met))

    diff = np.vstack(diff)  
    inds = np.argmin(diff, axis=0)

    return inds

def get_simulated_data_stats(path, metallicity, qmin):
    """Gathers the number of simulated systems and the total simulated ZAMS
    mass, including companions if appropriate, to compute the formation number
    of a given stellar system type per unit mass
    
    Parameters
    ----------
    path : `str`
        path to where the data is stored

    metallicity : `float`
        metallicity of simulated systems along metallicity grid

    qmin : `int`
        integer which specifies the qmin model
        
    Returns
    -------
    xi : `float`
        formation number per unit ZAMS mass of the stellar system
    """
    if 'single' in path:
        filename = 'singles.h5'
        bcm = pd.read_hdf(path+str(metallicity)+'/'+filename, key='bcm')
        n_sys = np.max(pd.read_hdf(path+str(metallicity)+'/'+filename, key='n_sim'))[0]
    
    elif 'binary' in path:
        filename = 'binaries_qmin_{}.h5'.format(qmin)
        bcm = pd.read_hdf(path+str(metallicity)+'/'+filename, key='bcm')
        n_sys = np.max(pd.read_hdf(path+str(metallicity)+'/'+filename, key='n_sim'))[0]
    
    elif 'BH' in path:
        ### NOTE ### : BH_LCs are assumed to have a binary fraction of 70%
        ############   and the n_stars contains all of the stars, from single
        ############   stars and binary systems, to produce the BH-LC population
        filename = 'dat_kstar1_14_kstar2_0_9_SFstart_13700.0_SFduration_0.0_metallicity_'+str(metallicity)+'.h5'
        bcm = pd.read_hdf(path+'/'+filename, key='bcm')
        n_sys = np.max(pd.read_hdf(path+'/'+filename, key='n_stars'))[0]

    # xi is the number of unique systems divided by the total
    # amount of stars simulated stars
    xi = len(bcm.bin_num.unique())/n_sys
    
    return xi

def get_formation_efficiency(mets, path, qmin, sys_type, f_b=None):
    """Reads in saved data containing relative formation number 
    per number of samples form initial conditions for each metallicity
    it met grid and system type: sys_type 
    if not available, reads in cosmic data to get this statistic and
    saves that info for future use to cut on computation time
    
    Parameters
    ----------
    mets : `list of lists`
        list of metallicities in simulation grid for passed system type

    path : `list`
        list containing path to simulated data for passed system type

    qmin : `int`
        integer which specifies the qmin model

    sys_type : `int`
        singles = 0; binaries = 1; bh binaries = 2;

    f_b : `float/list of lists`
        binary fraction to weight single stars against binary stars
        
        
    Returns
    -------
    xi : `ndarray`
        relative formation number per total population size for each 
        metallicity in simulation grid for each system type
    
    """
    try:
        xi = np.load('formation_efficiency_{}_{}.npy'.format(sys_type, qmin), allow_pickle=True)
    except:
        # get relative formation number per unit solar mass for each metallicity
        xi = []
        
        if sys_type == 0:
            if f_b == None:
                weights = metallicity_dependent_single_fraction(mets)
            else:
                f_s = 1 - f_b
                weights = len(mets) * [[f_s]]
        if sys_type == 1:
            if f_b == None:
                weights = metallicity_dependent_binary_fraction(mets)
            else:
                weights = len(mets) * [[f_b]]
        if sys_type == 2:
            # This is already taken care of in the simulation which
            # assumes close massive binaries have a binary fraction of 0.7
            weights = np.ones_like(mets)

        for met, weight in zip(mets, weights):
            xi.append(weight * get_simulated_data_stats(path=path, metallicity=met, qmin=qmin))
                
        xi = np.array(xi, dtype=object)
        np.save('formation_efficiency_{}_{}.npy'.format(sys_type, qmin), xi)
    return xi


def get_simulated_matches(path, met, sample_to_match, pop_var):
    """Selects initial conditions from cosmic data to match to star sample

    Parameters
    ----------
    path : `str`
        path to cosmic data

    met : `float`
        metallicity of cosmic data file

    sample_to_match : `DataFrame`
        A dataframe containing a population of stars with
        metallicities, ages, and positions

    pop_var : `int or str`
        Can be supplied for populations where sys_type is the same but the
        population is varied in some way, like if qmin is different. If no
        variants, pop_var = 0

    Returns
    -------
    initC_dat_sample : `DataFrame`
        cosmic initial conditions with assigned ages, positions, and metallicities
    """
    # read in the simulated binary data that has metallicities which 
    # are matched to sub_sample_sys_met
    sim_dat, initC_dat = sim_data_read(path=path, metallicity=met, qmin=pop_var)
    
    initC_dat['acc_lim'] = -1
    initC_dat['don_lim'] = -1
    
    # sample the same number of systems from sim_dat as sub_sample_sys_met
    initC_dat_sample = initC_dat.sample(len(sample_to_match), replace=True)
    initC_dat_sample = pd.concat([initC_dat_sample.reset_index(drop=True), sample_to_match.reset_index(drop=True)], axis=1)
    initC_dat_sample['assigned_age'] = np.array(sample_to_match['AGE'].values) * 1000
    
    return initC_dat_sample


def filter_sim_set(sim_set, lifetime_interp):
    """Filter out systems based on star types and coarse single star lifetime
    assumptions to reduce the datasize of stellar systems to evolve to the present

    Parameters
    ----------
    sim_set : `DataFrame`
        Dataframe containing initial conditions of cosmic data with ages

    lifetime_interp : `scipy.interpolate.interp1d`
        interpolation for single star lifetime as a function of mass
        for the population metallicity

    Returns
    -------
    sim_set_filter : `DataFrame`
       filtered DataFrame with systems that have lifetimes within reason
       based on their assigned ages
    """

    if sim_set.sys_type.all() == 0:
        sim_set = sim_set.loc[sim_set.assigned_age - sim_set.tphys < 100 * lifetime_interp(sim_set.mass_1)]
    

    elif sim_set.sys_type.all() == 1:
        sim_set = sim_set.loc[sim_set.assigned_age - sim_set.tphys < 100 * lifetime_interp(sim_set.mass_2)]
    
    elif sim_set.sys_type.all() == 2:
        sim_set = sim_set.loc[sim_set.assigned_age - sim_set.tphys < 1000 * lifetime_interp(sim_set.mass_2)]
    
    return sim_set

def get_evolved_systems(initC, sys_type, n_proc):
    """Evolves set of matched cosmic and sfh data up to present day population

    Parameters
    ----------
    initC : `DataFrame`
        Dataframe containing initial conditions of cosmic data with ages, positions, and metallicities

    sys_type : `int`
        singles = 0; binaries = 1; bh binaries = 2;

    n_proc : `int`
        number of processors to use when evolving cosmic binaries

    Returns
    -------
    dat_today : `DataFrame`
        present day population of stellar systems from cosmic data with ages, positions, and,
        metallicities provided by initC data
    """
    bpp_columns = ['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'sep', 'porb', 'ecc',
                   'RRLO_1', 'RRLO_2', 'evol_type', 'aj_1', 'aj_2', 'tms_1', 'tms_2',
                   'massc_1', 'massc_2', 'rad_1', 'rad_2', 'mass0_1', 'mass0_2', 'lum_1',
                   'lum_2', 'teff_1', 'teff_2', 'radc_1', 'radc_2', 'menv_1', 'menv_2',
                   'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1', 'B_2',
                   'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'epoch_1', 'epoch_2',
                   'bhspin_1', 'bhspin_2', 'bin_num']

    
    initC['tphysf'] = initC.assigned_age.values
    initC['dtp'] = initC.assigned_age.values
    bpp, bcm, initCond, kick_info = Evolve.evolve(initialbinarytable=initC, BSEDict={}, nproc=n_proc)
    bpp_nan_bin_num = bpp.loc[bpp.mass_1.isnull()].bin_num.unique()
    
    bcm_nan = bcm.loc[bcm.bin_num.isin(bpp_nan_bin_num)]
    initCond_nan = initCond.loc[initCond.bin_num.isin(bpp_nan_bin_num)]
    bcm_nan.to_hdf('nan_dat.h5', key='bcm')
    initCond_nan.to_hdf('nan_dat.h5', key='initC')

    bcm = bcm.loc[~bcm.bin_num.isin(bpp_nan_bin_num)]
    initC = initC.loc[~initC.bin_num.isin(bpp_nan_bin_num)]

    bcm = bcm.loc[bcm.tphys > 0].copy()
    bcm = bcm.reset_index(drop = True)
    if len(bcm) != len(initC):
        initC_weird = initC.loc[~initC.bin_num.isin(bcm.bin_num)]
        print('warning: there are {} systems that did not match in the evolution'.format(len(initC_weird)))
        print(bpp.loc[bpp.bin_num.isin(initC_weird.bin_num)][['tphys', 'mass_1', 'mass_2', 'sep', 'evol_type']])
        initC_weird.to_hdf('initC_weird_{}.h5'.format(len(initC_weird)), key='initC')
        initC = initC.loc[initC.bin_num.isin(bcm.bin_num)]
    initC['tphys'] = bcm.tphys.values
    initC['kstar_1'] = bcm.kstar_1.values
    initC['kstar_2'] = bcm.kstar_2.values
    initC['mass_1'] = bcm.mass_1.values
    initC['mass_2'] = bcm.mass_2.values
    initC['porb'] = bcm.porb.values
    initC['sep'] = bcm.sep.values
    initC['ecc'] = bcm.ecc.values
    initC['lum_1'] = bcm.lum_1.values
    initC['lum_2'] = bcm.lum_2.values
    initC['rad_1'] = bcm.rad_1.values
    initC['rad_2'] = bcm.rad_2.values
    initC['teff_1'] = bcm.teff_1.values
    initC['teff_2'] = bcm.teff_2.values
    columns_keep = ['tphys', 'kstar_1', 'kstar_2', 'mass_1', 'mass_2',
                    'porb', 'ecc', 'sep', 'rad_1', 'rad_2', 'teff_1', 'teff_2', 
                    'lum_1', 'lum_2', 'bin_num', 'assigned_age', 
                    'R', 'R0', 'AGE', 'X', 'Y', 'Z', 'dist', 'FeH', 'met_stars', 
                    'sys_type', 'met_sim']
    
    if sys_type == 0:
        # APOGEE defo won't see any white dwarfs because they are too hot
        initC = initC.loc[(initC.kstar_1 < 10) & (initC.mass_1 > 0.1)] 

    elif sys_type == 1:
        initC = initC.loc[(initC.porb > 0)]
        initC = initC.loc[((initC.kstar_2 < 10) & (initC.kstar_1 < 10) & (initC.mass_2 > 0.1)) |
                              ((initC.kstar_1 < 10) & (initC.mass_2 > 0.1)) | 
                              ((initC.kstar_2 < 10) & (initC.mass_2 > 0.1))]
            
    elif sys_type == 2:
        initC = initC.loc[initC.porb > 0]
        initC = initC.loc[((initC.kstar_1 == 14) & (initC.kstar_2 < 10) & (initC.mass_2 > 0.1))]


    dat_today = initC[columns_keep]
    return dat_today


def sample_stars(stars, mets, n_samp):
    """Generates a sample of size n_samp from the sfh data set
    and matches the sampled metallicities to the cosmic data
    metallicity grid (mets)

    Parameters
    ----------
    stars : `DataFrame`
        A dataframe containing a population of stars with
        metallicities, ages, and positions

    mets : `list`
        A list of metallicities in the cosmic data

    n_samp : `int`
        Number of stars to sample form the stars data set

    Returns
    -------
    samp : `DataFrame`
        A sample from the stars data frame with metallicities
        matched to the cosmic data metallicities
    """

    # Sample a population from the stars DataFrame
    samp = stars.sample(n_samp, replace = True

    # find cosmic metallicities which closest match to sfh metallicities
    ind_match = match_metallicities(met_list=mets,
                                    met_stars=np.array(sub_sample['met_stars']))

    samp['met_cosmic'] = np.array(mets)[ind_matches]

    return samp


def connect_simulations_to_stars(sample, sys_type, path, mets, lifetime_interp, n_proc, pop_var):
    """Connects sample of stars with positions, ages, and metallicities to cosmic populations
    with the specified sys_type

    Parameters
    ----------
    sample : `DataFrame`
       A sample from the stars data frame with metallicities
       matched to the cosmic data metallicities

    sys_type : `int`
        current sys_types include: 
        0 = general single stars
        1 = general binary stars
        2 = binaries containing black holes

    path : `str`
        path to the cosmic data

    mets : `list`
        list of metallicity bins in the simulated cosmic data

    lifetime_interp : `scipy.interpolate.interp1d`
        interpolation for single star lifetime as a function of mass
        for the population metallicity

    n_proc : `int`
        number of processors to evolve cosmic binaries

    pop_var : `int or string`
        Can be supplied for populations where sys_type is the same but the
        population is varied in some way, like if qmin is different. If no
        variants, pop_var = 0

    Returns
    -------
    sim_set : `DataFrame`
        Dataset of cosmic binaries with positions, ages, and metallicities
        according to the sfh of the samp data
    """ 

    sim_set = []
    initC_set = []
    # for each metallicity, grab all the systems with metallicity=met
    # then assign binaries in that metallicity bin to those systems
    for met in mets:
        sample_sys_met = sample.loc[sample.met_sim == met]
        if len(sample_sys_met) >= 1:
            initC_matches = get_simulated_matches(path=path, met=met, sample_to_match=sample_sys_met, pop_var=pop_var)
            initC_matches['sys_type'] = np.ones_like(initC_matches.mass_1) * sys_type
            # filter the stars/binary systems to get rid of those which have already evolved to 
            # become (double) compact objects, (double) white dwarfs, or have merged/disrupted
            # already base on the supplied birth time from sample_sys_met and ZAMS lifetimes 
            # for single stars. 
            # (NOTE: this is not very strict, the lifetime can be 1000x's the age and still not get cut)
            initC_matches = filter_sim_set(initC_matches, lifetime_interp)
             
            if len(initC_matches) > 0:
                if len(initC_set) == 0:
                    initC_set = initC_matches
                else:
                    initC_set = initC_set.append(initC_matches)
            # clear out the match_data
            initC_matches = []

    if len(initC_set) > 0:
        # reindex initC_set so that we can easily link 
        # the grids to the Galactic population
        initC_set = initC_set.reset_index(drop=True)
        initC_set = initC_set.reset_index(drop=False)
        initC_set = initC_set.rename(columns={"bin_num": "bin_num_grid", "index": "bin_num"}) 

        # evolve the selected binaries to the precise ages 
        # supplied by the star sample
        initC_set = initC_set.reset_index(drop=True)
        sim_set = get_evolved_systems(initC_set, sys_type, lifetime_interp, n_proc)
        
    return sim_set
