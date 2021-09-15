"""A collection of methods to create synthetic APOGEE catalogs"""

import numpy as np
from cosmicats import photobs as obs
from scipy.stats import gaussian_kde

__all__ = ['get_2MASS_phot', 'binary_select', 'phot_select']

def get_2MASS_phot(sim_set, sys_type, bc_grid):
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
    print('pop size before extinction cut: {}'.format(len(sim_set)))
    sim_set.loc[sim_set.Av < 6, ['Av']] = 6
    print('pop size after extinction cut: {}'.format(len(sim_set)))
    
    if sys_type == 0:
        phot_1 = obs.get_photometry_1(sim_set, bc_grid)
        m_app_1, J_app_1, H_app_1, K_app_1, G_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1, G_abs_1 = phot_1          
        
        sim_set['mbol_app'] = m_app_1
        sim_set['J_app'] = J_app_1
        sim_set['H_app'] = H_app_1
        sim_set['K_app'] = K_app_1
        sim_set['G_app'] = G_app_1
        
        sim_set['mbol_abs'] = m_abs_1
        sim_set['J_abs'] = J_abs_1
        sim_set['H_abs'] = H_abs_1
        sim_set['K_abs'] = K_abs_1
        sim_set['G_abs'] = G_abs_1
        
        # if single: the bright system is just the star
        sim_set['sys_bright'] = np.ones(len(sim_set))
        sim_set['logg_obs'] = sim_set.logg_1.values
        sim_set['teff_obs'] = sim_set.teff_1.values

    elif sys_type == 1:
        phot_1 = obs.get_photometry_1(sim_set, bc_grid)
        m_app_1, J_app_1, H_app_1, K_app_1, G_app_1, m_abs_1, J_abs_1, H_abs_1, K_abs_1, G_abs_1 = phot_1  
        
        phot_2 = obs.get_photometry_2(sim_set, bc_grid)
        m_app_2, J_app_2, H_app_2, K_app_2, G_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2, G_abs_2 = phot_2
                
        # check if the primary or secondary is brighter in 2MASS K
        sys_bright = np.ones(len(sim_set))
        
        # next handle the systems where there was merger and the leftover star
        # is left in kstar_2 instead of kstar_1
        kstar_1 = sim_set.kstar_1.values
        ind_single_1 = np.where(kstar_1 == 15)[0]
        sys_bright[ind_single_1] = 2.0
        
        # next; in some instances, there are systems which are too dim to register
        # in the isochrones/MIST grids
        ind_dim_1 = np.where(np.isnan(J_app_1))[0]
        sys_bright[ind_dim_1] = 2.0
        ind_dim_2 = np.where(np.isnan(J_app_2))[0]
        #ind_dim_2 already covered above
        
        ind_2_bright = np.where(K_app_2 < K_app_1)[0]
        ind_1_bright = np.where(K_app_2 >= K_app_1)[0]
        sys_bright[ind_2_bright] = 2.0
        #ind_1_bright already covered above
        
        sim_set['sys_bright'] = sys_bright
        
        logg_obs = np.zeros(len(sim_set))
        logg_obs[sys_bright == 1.0] = sim_set.loc[sim_set.sys_bright == 1].logg_1
        logg_obs[sys_bright == 2.0] = sim_set.loc[sim_set.sys_bright == 2].logg_2
        sim_set['logg_obs'] = logg_obs
        
        teff_obs = np.zeros(len(sim_set))
        teff_obs[sys_bright == 1.0] = sim_set.loc[sim_set.sys_bright == 1].teff_1
        teff_obs[sys_bright == 2.0] = sim_set.loc[sim_set.sys_bright == 2].teff_2
        sim_set['teff_obs'] = teff_obs
        
        
        sim_set['J_app'] = obs.addMags(J_app_1, J_app_2)
        sim_set['H_app'] = obs.addMags(H_app_1, H_app_2)
        sim_set['K_app'] = obs.addMags(K_app_1, K_app_2)
        sim_set['G_app'] = obs.addMags(G_app_1, G_app_2)
        sim_set['mbol_app'] = obs.addMags(m_app_1, m_app_2)
    
        sim_set['J_abs'] = obs.addMags(J_abs_1, J_abs_2)
        sim_set['H_abs'] = obs.addMags(H_abs_1, H_abs_2)
        sim_set['K_abs'] = obs.addMags(K_abs_1, K_abs_2)
        sim_set['G_abs'] = obs.addMags(G_abs_1, G_abs_2)
        sim_set['mbol_abs'] = obs.addMags(m_abs_1, m_abs_2)
    
    elif sys_type == 2:
        phot_2 = obs.get_photometry_2(sim_set, bc_grid)
        m_app_2, J_app_2, H_app_2, K_app_2, G_app_2, m_abs_2, J_abs_2, H_abs_2, K_abs_2, G_abs_2 = phot_2
        
        sim_set['mbol_app'] = m_app_2
        sim_set['J_app'] = J_app_2
        sim_set['H_app'] = H_app_2
        sim_set['K_app'] = K_app_2
        sim_set['G_app'] = K_app_2
        
        sim_set['mbol_abs'] = m_abs_2
        sim_set['J_abs'] = J_abs_2
        sim_set['H_abs'] = H_abs_2
        sim_set['K_abs'] = K_abs_2
        sim_set['G_abs'] = K_abs_2
        
        # if single: the bright system is just the star
        sim_set['sys_bright'] = 2*np.ones(len(sim_set))
        sim_set['logg_obs'] = sim_set.logg_2.values
        sim_set['teff_obs'] = sim_set.teff_2.values
        
    return sim_set

def binary_select(dat, APOGEE_log_P):
    # assign everything to be observed as singles
    obs_type = np.zeros(len(dat))
    
    #re-assign the systems that are likely to be picked up as a binary
    #given their orbital periods
    p_prob = np.random.uniform(0, 1, len(dat))
    porb_kde = gaussian_kde(APOGEE_log_P)
    kde_eval = porb_kde.evaluate(np.log10(dat.porb.values))
    np.savetxt('p_prob.csv', p_prob, delimiter = ',')
    np.savetxt('kde_eval.csv', kde_eval, delimiter = ',')
    obs_type = np.where(p_prob < kde_eval, 1, 0)
    dat[['obs_type']] = obs_type
    return dat

#Add function that does color cut: J-K > 0.5 (main survey for APOGEE 1), APOGEE 2 made the cut bluer.
# so we probs want APOGEE 2

def phot_select(dat, JminusK_lo=0.5):
    dat['phot_select'] = np.zeros(len(dat))
    dat.loc[(dat.J_app - dat.K_app > JminusK_lo) & (dat.H_app < 15),
           'phot_select'] = 1.0

    return dat

def gold_select(dat, logg_lo = -0.5, logg_hi = 5.5, teff_lo = 3500, teff_hi = 10000):
    
    #dat = dat.loc[dat.H_app < 15].copy()
    #ignoring [M/H], s_MAP for now
    dat['gold_select'] = np.zeros(len(dat))
    dat.loc[(dat.logg_obs < logg_hi) & (dat.logg_obs > logg_lo) &
            (dat.teff_obs < teff_hi) & (dat.teff_obs > teff_lo) &
            (dat.H_app < 15), 'gold_select'] = 1.0
    
    return dat
