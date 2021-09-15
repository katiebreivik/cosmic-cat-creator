"""A class for generating stellar populations"""

from cosmicats import utils, popgen, apogee
from isochrones.mist.bc import MISTBolometricCorrectionGrid
import pandas as pd
import astropy.table as at
import numpy as np

class pop():
    """Class for generic populations

    Attributes
    ----------
    sys_type : `int`
        current sys_types include: 
        0 = general single stars
        1 = general binary stars
        2 = binaries containing black holes

    n_stop_APOGEE : `int`
        stopping condition which specifies the maximum size of the 
        APOGEE population

    n_stop_MW : `int`
        stopping condition which specifies the maximum size of the 
        MW population
        
    n_samp : `int`
        specifies the number of systems to sample from the cosmic and sfh data

    mets : `list`
        list of metallicity bins in the simulated cosmic data

    cosmic_path : `str`
        path to the cosmic data

    sfh_model : `str`
        model assumed for stellar ages and positions as a funciton of metallicity
        current models include:
        'Frankel19' : positions and ages from Frankel+2019

    seed : `int`
        random seed for reproduceability

    pop_var : `str`
        Can be supplied for populations where sys_type is the same but the
        population is varied in some way, like if qmin is different 

    lifetime_interp : `scipy.interpolate.interp1d`
        interpolation for single star lifetime as a function of mass
        for the population metallicity

    """
    def __init__(self, sys_type, n_stop_APOGEE, 
                 n_stop_MW, n_samp, mets, 
                 cosmic_path, lifetime_interp, 
                 sfh_model='Frankel19', seed=42,
                 pop_var=None, color_cut=0.3):

        self.sys_type = sys_type
        self.n_stop_APOGEE = n_stop_APOGEE
        self.n_stop_MW = n_stop_MW
        self.n_samp = n_samp
        self.mets = mets
        self.cosmic_path = cosmic_path
        self.sfh_model = sfh_model
        self.seed = seed

        # set up dat and log files
        if pop_var == None:
            self.pop_var = 0
        else:
            self.pop_var = pop_var

        self.color_cut = color_cut
            
        # set up the single star lifetime interpolator
        # note: we use the minimum metallicity which gives the maximum lifetime
        self.lifetime_interp = lifetime_interp


    def get_formation_efficiency(self, f_b=None):
        """Get the formation efficiency as a function of metallicity
        NOTE : for the moment, we put in f_b by hand using the methods
        in popgen, however in the future, it would be good to implement
        a general treatment!
    
        Parameters
        ----------
        f_b : `float/list of lists`
            binary fraction to weight single stars against binary stars
    
        Returns
        -------
        formation_efficiency:  `ndarray`
            relative formation number per total population size for each 
            metallicity in simulation grid for each system type
        """
        formation_efficiency = popgen.get_formation_efficiency(mets=self.mets,
                                                               path=self.cosmic_path,
                                                               var=self.pop_var,
                                                               sys_type=self.sys_type,
                                                               f_b=None)
    
        return formation_efficiency

    def build_pop(self, n_proc, run):
        """Generates an astrophysical population and associated APOGEE 
        catalog by matching a cosmic dataset to a star formation model 
        for a given population system type, and applying APOGEE-like
        selections on the data"""

        # load in the APOGEE binary data go get the orbital period data
        binaries = at.Table.read('lnK0.0_logL4.6_metadata.fits')
        all_star = at.Table.read('allStarLite-r12-l33.fits')
        
        binaries_join = at.unique(at.join(binaries, all_star, join_type='left', keys='APOGEE_ID'), 'APOGEE_ID')
        cols_drop = ['NINST', 'STABLERV_CHI2', 'STABLERV_RCHI2', 'CHI2_THRESHOLD', 'STABLERV_CHI2_PROB', 
                     'PARAM', 'FPARAM', 'PARAM_COV', 'FPARAM_COV', 'PARAMFLAG', 'FELEM', 'FELEM_ERR', 
                     'X_H', 'X_H_ERR', 'X_M', 'X_M_ERR', 'ELEM_CHI2', 'ELEMFLAG', 'ALL_VISIT_PK', 
                     'VISIT_PK', 'FPARAM_CLASS', 'CHI2_CLASS', 'binary_catalog']
        
        cols_keep = []
        for col in binaries_join.columns:
            if col not in cols_drop:
                cols_keep.append(col)
                
        binaries_join = binaries_join[cols_keep].to_pandas()
        APOGEE_log_P = np.log10(binaries_join.MAP_P.values)
        
        
        datfile_name = 'pop_{}_var_{}_run_{}.h5'.format(self.sys_type, self.pop_var, run)
        logfile_name = 'log_pop_{}_var_{}_run_{}.txt'.format(self.sys_type, self.pop_var, run)

        # open up a file to write data to in case the run stops
        # Open the hdf5 file to store the fixed population data
        try:
            dat_store = pd.HDFStore(datfile_name)
            n_samp_MW = pd.read_hdf(datfile_name, 'n_samp_MW').max()[0]
            n_samp_APOGEE = pd.read_hdf(datfile_name, 'n_samp_APOGEE').max()[0]

            MW_pop = pd.read_hdf(datfile_name, 'MW_pop')
            n_MW = len(MW_pop)
            APOGEE_pop = pd.read_hdf(datfile_name, 'APOGEE_pop')
            n_APOGEE = len(APOGEE_pop)

            if self.sys_type > 0:
                n_binary = len(APOGEE_pop.loc[APOGEE_pop.obs_type == 1])

            log_file = open(logfile_name, 'a')
            log_file.write('There are already: '+str(n_MW)+' MW systems in the population.\n')
            log_file.write('There are already: '+str(n_APOGEE)+' APOGEE systems in the population.\n')
            log_file.write('\n')
        except:
            dat_store = pd.HDFStore(datfile_name)
            n_MW = 0
            n_APOGEE = 0
            n_samp_MW = 0
            n_samp_APOGEE = 0
            n_samp_Gold = 0
            if self.sys_type > 0:
                n_binary = 0

            log_file = open(logfile_name, 'a')

        # set up Dataset of stars from sfh_model
        star_sample = popgen.get_sfh_stars(self.sfh_model)        

        # initialize bolometric correction grid with isochrones
        bc_grid = MISTBolometricCorrectionGrid(['J', 'H', 'K', 'G'])
        

        ## Get the formation efficiency as a function of metallicity
        ## NOTE : for the moment, we put in f_b by hand using the methods
        ## in popgen, however in the future, it would be good to implement
        ## a general treatment!
        #formation_efficiency = popgen.get_formation_efficiency(mets=self.mets,
        #                                                       path=self.cosmic_path,
        #                                                       var=self.pop_var,
        #                                                       sys_type=self.sys_type,
        #                                                       f_b=None)
        #dat_store.append('formation_efficiency', pd.DataFrame(formation_efficiency))

        # Sample with replacement from this dataset, and attach to simulated single stars
        # binary systems, and BH binaries, then calculate APOGEE photometry and filter
        # based on orbital period, brightness, temperature, and log g

        # Set condition for MW_write
        MW_sim_write = True

        # repeat the process until we have n_stop_APOGEE systems in the APOGEE population

        n_APOGEE = 0
        n_MW = 0
        n_IC = 0
        while (n_APOGEE < self.n_stop_APOGEE):
            # sample from SFH data set
            sample = popgen.sample_stars(stars=star_sample, 
                                         mets=self.mets,
                                         n_samp = self.n_samp)

            # connect the sampled ages, positions, and metallicities to the cosmic population
            pop_today = popgen.connect_simulations_to_stars(sample=sample, 
                                                            sys_type=self.sys_type, 
                                                            path=self.cosmic_path, 
                                                            mets=self.mets, 
                                                            lifetime_interp=self.lifetime_interp, 
                                                            n_proc=n_proc,
                                                            pop_var=self.pop_var)

            # compute the photometry of the population
            pop_today = apogee.get_2MASS_phot(sim_set=pop_today, 
                                              sys_type=self.sys_type,
                                              bc_grid=bc_grid)

            # perform photometry, binarity, and gold selections
            pop_today = apogee.binary_select(pop_today, APOGEE_log_P)
            pop_today = apogee.phot_select(pop_today, JminusK_lo=self.color_cut)
            pop_gold = apogee.gold_select(pop_today)
            
            # cut out the dimmest sources for data size
            pop_today = pop_today.loc[pop_today.G_app <= 21]
            n_MW += len(pop_today)

            # select the APOGEE population
            pop_APOGEE = pop_today.loc[pop_today.H_app <= 15]
            n_APOGEE += len(pop_APOGEE)

            # select the Gold population
            pop_gold = pop_today.loc[pop_today.gold_select == 1]
            
            # update the population size logs
            n_samp_MW += self.n_samp
            n_samp_APOGEE += self.n_samp
            n_samp_Gold += self.n_samp

            
            if MW_sim_write:
                dat_store.append('MW_pop', pop_today)
                dat_store.append('n_samp_MW', pd.DataFrame([n_samp_MW]))

            dat_store.append('APOGEE_pop', pop_APOGEE)
            dat_store.append('n_samp_APOGEE', pd.DataFrame([n_samp_APOGEE]))
            
            dat_store.append('Gold_pop', pop_gold)
            dat_store.append('n_samp_Gold', pd.DataFrame([n_samp_Gold]))
            
            log_file.write('size of sys_type= {}\n'.format(self.sys_type))
            log_file.write(str(n_MW)+'\n')
            log_file.write('\n')
            if self.sys_type > 0:
                n_binary += len(pop_today.loc[pop_today.obs_type == 1])
                log_file.write('size of observed binary population:\n')
                log_file.write(str(len(pop_today.loc[pop_today.obs_type == 1]))+'\n')
                log_file.write('\n')
            log_file.write('size of photometrically selected apogee population:\n')
            log_file.write(str(n_APOGEE)+'\n')
            log_file.write('\n')
            log_file.write('\n')
            log_file.flush()

            if n_MW > self.n_stop_MW:
                MW_sim_write = False
        dat_store.append('seed', pd.DataFrame([self.seed]))
        log_file.write('all done friend!')
        log_file.close()
        dat_store.close() 

