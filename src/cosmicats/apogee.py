"""A collection of methods to create synthetic APOGEE catalogs"""

from cosmicats import photobs as obs

__all__ = ['get_APOGEE_phot', 'binary_select', 'phot_select']

def get_APOGEE_phot(sim_set, sys_type, bc_grid):
    """Computes J,H,K photometry subject to dust extinction
    using the MIST boloemtric correction grid which contains
    filters J,H,K

    Parameters
    ----------
    sim_set : `DataFrame`
        dataset of cosmic binaries at present dat

    sys_type : `int`
        system type; choose from:
        singles = 0; binaries = 1; bh binaries = 2; 

    bc_grid : `MISTBolometricCorrectionGrid`
        bolometric correction grid which works with isochrones
        to compute BCs

    Returns
    -------
    sim_set : `DataFrame`
        dataset of cosmic binaries at present dat with added 
        photometry and extinction information
    """
    sim_set['Av'] = obs.get_extinction(sim_set)
    sim_set = sim_set.loc[sim_set.Av < 6].copy()
    
    if sys_type == 0:
        phot_1 = obs.get_photometry_1(sim_set, bc_grid)
        m_app_1, J_app_1, H_app_1, K_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1 = phot_1          
        
        sim_set['mbol_app'] = m_app_1
        sim_set['J_app'] = J_app_1
        sim_set['H_app'] = H_app_1
        sim_set['K_app'] = K_app_1
        
        sim_set['mbol_abs'] = m_abs_1
        sim_set['J_abs'] = J_abs_1
        sim_set['H_abs'] = H_abs_1
        sim_set['K_abs'] = K_abs_1
        
        # if single: the bright system is just the star
        sim_set['sys_bright'] = np.ones(len(sim_set))
        sim_set['logg_obs'] = sim_set.logg_1.values
        sim_set['teff_obs'] = sim_set.teff_1.values

    elif sys_type == 1:
        phot_1 = obs.get_photometry_1(sim_set, bc_grid)
        m_app_1, J_app_1, H_app_1, K_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1 = phot_1
        
        phot_2 = obs.get_photometry_2(sim_set, bc_grid)
        m_app_2, J_app_2, H_app_2, K_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2 = phot_2
        
        # check if the primary or secondary is brighter
        sys_bright = np.ones(len(sim_set))
        ind_2_bright, = np.where(m_app_2 > m_app_1)
        ind_1_bright, = np.where(m_app_2 <= m_app_1)
        sys_bright[ind_2_bright] = 2.0
        sim_set['sys_bright'] = sys_bright
        
        logg_obs = np.zeros(len(sim_set))
        logg_obs[ind_1_bright] = sim_set.loc[sim_set.sys_bright == 1].logg_1
        logg_obs[ind_2_bright] = sim_set.loc[sim_set.sys_bright == 2].logg_2
        sim_set['logg_obs'] = logg_obs
        
        teff_obs = np.zeros(len(sim_set))
        teff_obs[ind_1_bright] = sim_set.loc[sim_set.sys_bright == 1].teff_1
        teff_obs[ind_2_bright] = sim_set.loc[sim_set.sys_bright == 2].teff_2
        sim_set['teff_obs'] = teff_obs
        
        
        sim_set['J_app'] = obs.addMags(J_app_1, J_app_2)
        sim_set['H_app'] = obs.addMags(H_app_1, H_app_2)
        sim_set['K_app'] = obs.addMags(K_app_1, K_app_2)
        sim_set['mbol_app'] = obs.addMags(m_app_1, m_app_2)
    
        sim_set['J_abs'] = obs.addMags(J_abs_1, J_abs_2)
        sim_set['H_abs'] = obs.addMags(H_abs_1, H_abs_2)
        sim_set['K_abs'] = obs.addMags(K_abs_1, K_abs_2)
        sim_set['mbol_abs'] = obs.addMags(m_abs_1, m_abs_2)
    
    elif sys_type == 2:
        phot_2 = obs.get_photometry_2(sim_set, bc_grid)
        m_app_2, J_app_2, H_app_2, K_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2 = phot_2
        
        sim_set['mbol_app'] = m_app_2
        sim_set['J_app'] = J_app_2
        sim_set['H_app'] = H_app_2
        sim_set['K_app'] = K_app_2
        
        sim_set['mbol_abs'] = m_abs_2
        sim_set['J_abs'] = J_abs_2
        sim_set['H_abs'] = H_abs_2
        sim_set['K_abs'] = K_abs_2
        
        # if single: the bright system is just the star
        sim_set['sys_bright'] = 2*np.ones(len(sim_set))
        sim_set['logg_obs'] = sim_set.logg_2.values
        sim_set['teff_obs'] = sim_set.teff_2.values
        
    return sim_set

def binary_select(dat):
    # assign everything to be observed as singles
    dat['obs_type'] = np.zeros(len(dat))
    
    #re-assign the systems that are likely to be picked up as a binary
    #given their orbital periods
    dat.loc[(dat.sys_type > 0) & (dat.porb > 2) & (dat.porb < 365.25), 'obs_type'] = 1.0
    return dat

def phot_select(dat, logg_lo = -0.5, logg_hi = 5.5, teff_lo = 3500, teff_hi = 10000):
    
    #dat = dat.loc[dat.H_app < 15].copy()
    #ignoring [M/H], s_MAP for now
    dat['apogee_select'] = np.zeros(len(dat))
    dat.loc[(dat.logg_obs < logg_hi) & (dat.logg_obs > logg_lo) &
            (dat.teff_obs < teff_hi) & (dat.teff_obs > teff_lo) &
            (dat.H_app < 15), 'apogee_select'] = 1.0
    
    return dat