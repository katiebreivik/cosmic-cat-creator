"""A collection of utility functions for building catalogs"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve


__all__ = ['get_Z_from_FeH', 'get_FeH_from_Z', 'sim_data_read',
           'get_lifetime_interp'] 

def get_Z_from_FeH(FeH, Z_sun=0.014):
    """Converts from FeH to Z under the assumption that
    all stars have the same abundance as the sun
    
    Parameters
    ----------
    FeH : `array`
        array of Fe/H values to convert
        
    Z_sun : `float`
        solar metallicity
        
    Returns
    -------
    Z : `array`
        array of metallicities
    """
    Z = 10**(FeH + np.log10(Z_sun))
    
    return Z

def get_FeH_from_Z(Z, Z_sun=0.014):
    """Converts from FeH to Z under the assumption that
    all stars have the same abundance as the sun
    
    Parameters
    ----------
    Z : `array`
        array of metallicities
        
    Z_sun : `float`
        solar metallicity


    Returns
    -------
    FeH : `array`
        array of Fe/H values to convert
    """
    FeH = np.log10(Z) - np.log10(Z_sun)

    return FeH

def sim_data_read(path, metallicity, qmin):
    """Read in the data from the simulated binary data set
    and get it in a format that is easy to use for this project

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
    dat : DataFrame
        Contains all the data we need to map the simulated binaries
        onto the sampled stars from the Frankel disk model
    """

    if 'single' in path:
        filename = 'singles.h5'
        dat = pd.read_hdf(path+str(metallicity)+'/'+filename, key='bpp')
        initC = pd.read_hdf(path+str(metallicity)+'/'+filename, key='initC')
    
    elif 'binaries' in path:
        filename = 'binaries_qmin_{}.h5'.format(qmin)
        dat = pd.read_hdf(path+str(metallicity)+'/'+filename, key='bpp')
        initC = pd.read_hdf(path+str(metallicity)+'/'+filename, key='initC')
    
    elif 'BH' in path:
        filename = 'dat_kstar1_14_kstar2_0_9_SFstart_13700.0_SFduration_0.0_metallicity_'+str(metallicity)+'.h5'
        dat = pd.read_hdf(path+'/'+filename, key='bpp')
        initC = pd.read_hdf(path+'/'+filename, key='initCond')

    return dat, initC

def get_lifetime_interp(metallicity):
    """Gets the lifetime of single stars for the cosmic range of
    valid stellar masses (0.1-150 Msun) for coarse population filtering

    Parameters
    ----------
    metallicity : `float`
        metallicity of the population to get lifetime interpolation for

    Returns
    -------
    lifetime_interp : `scipy.interpolate.interp1d`
        interpolator for single star lifetime as a function of mass
    """
  
    mass = np.linspace(0.1, 150.0, 100)
    init_bin_hi_met = InitialBinaryTable.InitialBinaries(m1=mass, 
                                                         m2=np.zeros_like(mass), 
                                                         porb=np.zeros_like(mass), 
                                                         ecc=np.zeros_like(mass), 
                                                         tphysf=np.ones_like(mass) * 10000000.0, 
                                                         kstar1=np.ones_like(mass), 
                                                         kstar2=np.zeros_like(mass), 
                                                         metallicity=np.ones_like(mass) * metallicity)
    # Now we evolve the binaries up to their *actual* age as assigned by Neiege's stars.
    BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.05, 'pts3': 0.02, 'pts2': 0.01, 
               'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1, 'tflag': 1, 'acc2': 1.5, 
               'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0, 
               'natal_kick_array' : [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]], 'bhsigmafrac' : 1.0, 
               'polar_kick_angle' : 90, 'qcrit_array' : [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], 'cekickflag' : 2, 
               'cehestarflag' : 0, 'cemergeflag' : 0, 'ecsn' : 2.5, 'ecsn_mlow' : 1.4, 'aic' : 1, 'ussn' : 0, 'sigmadiv' :-20.0, 'qcflag' : 2, 
               'eddlimflag' : 0, 'fprimc_array' : [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,
                                                   2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0], 
               'bhspinflag' : 0, 'bhspinmag' : 0.0, 'rejuv_fac' : 1.0, 'rejuvflag' : 0, 'htpmb' : 1, 'ST_cr' : 1, 'ST_tide' : 1, 'bdecayfac' : 1, 
               'rembar_massloss' : 0.5, 'kickflag' : 0, 'zsun' : 0.014, 'grflag' : 1, 'bhms_coll_flag':0, 'acc_lim' : -1, 'don_lim' : -1}
    bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=init_bin_hi_met, BSEDict = BSEDict)
    bpp = bpp.loc[bpp.kstar_1.isin([10,11,12,13,14,15])]
    t_lifetime = bpp.groupby('bin_num').first().tphys
    lifetime_interp = interp1d(mass, t_lifetime)

    return lifetime_interp
