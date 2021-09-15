"""A collection of methods to generate synthetic photometry and other observables"""

import numpy as np
import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from dustmaps.bayestar import BayestarQuery

__all__ = ['log_g', 'M_absolute_bol', 'm_apparent', 'm_abs', 'get_mags',
           'addMags', 'get_extinction', 'get_photometry_1', 'get_photometry_2']


def log_g(mass, radius):
    """ Computes log g in cgs units

    Parameters
    ----------
    mass : `float/array`
        mass in solar masses

    radius : `float/array`
        radius in solar radii

    Returns
    -------
    log g : `float/array`
        log surface gravity in cgs
    """
    G_cgs = 6.67e-8
    Msun_cgs = 1.989e33
    Rsun_cgs = 6.9551e10

    g = G_cgs * mass * Msun_cgs / (radius * Rsun_cgs) ** 2

    return np.log10(g)


def M_absolute_bol(lum):
    """Computes the absolute bolometric luminosity

    Parameters
    ----------
    lum : `float/array`
        luminosity in solar luminosities
    
    Returns
    -------
    M_bol : `float/array`
        absolute bolometric magnitude
    """
    log_lum = np.log10(lum)
    M_bol = 4.75 - 2.7 * log_lum
    return M_bol


def m_apparent(M_abs, dist):
    # distance in parsecs
    m_app = M_abs + 5 * np.log10(dist / 10)
    return m_app


def m_abs(m_app, dist):
    # distance in parsecs
    M_abs = m_app - 5 * np.log10(dist / 10)
    return M_abs


def get_mags(lum, distance, teff, logg, Fe_h, Av, bc_grid, filters):
    """ Uses isochrones bolometric correction method to interpolate
    across the MIST bolometric correction grid
    
     Parameters
    ----------
    lum : `array`
        luminosity in Lsun
    
    distance : `array`
        distance in kpc
    
    teff : `array`
        effective temperature in K
    
    logg : `array`
        log g in cgs
    
    Fe_h : `array`
        metallicity
    
    Av : `array`
        extinction correction 
    
    bc_grid : `isochrones bolometric correction grid object`
        object which generates bolometric corrections!
        
    Returns
    -------
    mags : `list of arrays`
        list of apparent magnitude arrays that matches the filters provided 
        prepended with the bolometric apparent magnitude
    """

    M_abs = M_absolute_bol(lum=lum)
    m_app = m_apparent(M_abs=M_abs, dist=distance * 1000)
    BCs_app = bc_grid.interp([teff, logg, Fe_h, Av], filters)
    BCs_abs = bc_grid.interp([teff, logg, Fe_h, np.zeros_like(Av)], filters)
    mags_app = [m_app]
    for ii, filt in zip(range(len(filters)), filters):
        mags_app.append(m_app - BCs_app[:, ii])
    mags_abs = [m_app]
    for ii, filt in zip(range(len(filters)), filters):
        mags_abs.append(M_abs - BCs_abs[:, ii])

    return mags_app, mags_abs


def addMags(mag1, mag2):
    """ Adds two stellar magnitudes

    Parameters
    ----------
    mag1, mag2 : `float/array`
        two magnitudes from two stars

    Returns
    -------
    magsum : `float/array`
        returns the sum of mag1 and mag2
    """
    magsum = -2.5 * np.log10(10 ** (-mag1 * 0.4) + 10 ** (-mag2 * 0.4))
    return magsum


def get_extinction(dat):
    """Calculates the visual extinction values from the dat
    DataFrame using the dustmaps.bayestar query

    Parameters
    ----------
    dat : `pandas.DataFrame`
        contains Galactocentric cartesian coordinates
        with names [units]: X [kpc], Y [kpc], Z [kpc]

    Returns
    -------
    Av : `array`
        Visual extinction values for all points in dat

    """
    c = SkyCoord(x=np.array(dat.X) * u.kpc,
                 y=np.array(dat.Y) * u.kpc,
                 z=np.array(dat.Z) * u.kpc,
                 frame=coord.Galactocentric)
    bayestar = BayestarQuery(max_samples=2, version='bayestar2019')
    ebv = bayestar(c, mode='random_sample')
    Av = 3.2 * ebv

    return Av


def get_photometry(mass, rad, lum, dist, teff, FeH, Av, bc_grid):
    """Computes photometry for single star

    Parameters
    ----------
    mass : `numpy.array`
        mass in Msun

    rad : `numpy.array`
        radius in Rsun

    lum : `numpy.array`
        luminosity in Lsun

    dist : `numpy.array`
        distance in kpc

    teff : `numpy.array`
        effective temperature in K

    FeH : `numpy.array`
        iron abundance

    Av : `numpy.array`
        extinction value

    bc_grid : `isochrones.mist.bc`
        bolometric correction grid for photometry

    Returns
    -------
    m_app : `numpy.array`
        apparent bolometric magnitude

    J_app : `numpy.array`
        apparent 2MASS J magnitude

    H_app : `numpy.array`
        apparent 2MASS H magnitude

    K_app : `numpy.array`
        apparent 2MASS K magnitude

    G_app : `numpy.array`
        apparent Gaia G magnitude

    m_abs : `numpy.array`
        absolute bolometric magnitude

    J_abs : `numpy.array`
        absolute 2MASS J magnitude

    H_abs : `numpy.array`
        absolute 2MASS h magnitude

    K_abs : `numpy.array`
        absolute 2MASS K magnitude

    G_abs : `numpy.array`
        absolute Gaia G magnitude
    """

    # Now let's check out the brightness of the companions in 2MASS filters
    # for this we need to calculate log g of the companion
    logg = log_g(mass, rad)

    mags_app, mags_abs = get_mags(lum=lum,
                                  distance=dist,
                                  teff=teff,
                                  logg=logg,
                                  Fe_h=FeH,
                                  Av=Av,
                                  bc_grid=bc_grid,
                                  filters=['J', 'H', 'K', 'G'])

    [m_app, J_app, H_app, K_app, G_app] = mags_app
    [m_abs, J_abs, H_abs, K_abs, G_abs] = mags_abs

    return m_app, J_app, H_app, K_app, G_app, m_abs, J_abs, H_abs, K_abs, G_abs
