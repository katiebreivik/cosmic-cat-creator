"""A class for generating stellar populations"""

from cosmicats import utils, popgen, apogee
from isochrones.mist.bc import MISTBolometricCorrectionGrid
import pandas as pd

class pop():
    """Class for generic populations

    Attributes
    ----------
    sys_type : `int`
        current sys_types include: 
        0 = general single stars
        1 = general binary stars
        2 = binaries containing black holes

    n_stop : `int`
        stopping condition which specifies the maximum size of the population

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
    def __init__(self, sys_type, n_stop, n_samp, mets, 
                 cosmic_path, lifetime_interp, 
                 sfh_model='Frankel19', seed=42,
                 pop_var=None):

        self.sys_type = sys_type
        self.n_stop = n_stop
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
                                                               qmin=self.pop_var,
                                                               sys_type=self.sys_type,
                                                               f_b=None)
    
        return formation_efficiency

    def build_pop(self, n_proc):
        """Generates an astrophysical population and associated APOGEE 
        catalog by matching a cosmic dataset to a star formation model 
        for a given population system type, and applying APOGEE-like
        selections on the data"""

        datfile_name = 'pop_{}_var_{}.h5'.format(self.sys_type, self.pop_var)
        logfile_name = 'log_pop_{}_var_{}.txt'.format(self.sys_type, self.pop_var)

        # open up a file to write data to in case the run stops
        # Open the hdf5 file to store the fixed population data
        try:
            dat_store = pd.HDFStore(datfile_name)
            self.pop = pd.read_hdf(datfile_name, 'pop_sim')
            log_file = open(logfile_name, 'a')
            log_file.write('There are already: '+str(self.pop.shape[0])+' systems in the population.\n')
            log_file.write('\n')
        except:
            dat_store = pd.HDFStore(datfile_name)
            self.pop = pd.DataFrame()

            log_file = open(logfile_name, 'a')

        # set up Dataset of stars from sfh_model
        star_sample = popgen.get_sfh_stars(self.sfh_model)        

        # initialize bolometric correction grid with isochrones
        bc_grid = MISTBolometricCorrectionGrid(['J', 'H', 'K'])
        

        # Get the formation efficiency as a function of metallicity
        # NOTE : for the moment, we put in f_b by hand using the methods
        # in popgen, however in the future, it would be good to implement
        # a general treatment!
        formation_efficiency = popgen.get_formation_efficiency(mets=self.mets,
                                                               path=self.cosmic_path,
                                                               qmin=self.pop_var,
                                                               sys_type=self.sys_type,
                                                               f_b=None)


        # Sample with replacement from this dataset, and attach to simulated single stars
        # binary systems, and BH binaries, then calculate APOGEE photometry and filter
        # based on orbital period, brightness, temperature, and log g

        # repeat the process until we have n_stop systems
        n_sys = 0
        n_sample = 0
        n_pop = 0
        while (n_sys < self.n_stop):
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
            pop_today = apogee.get_APOGEE_phot(sim_set=pop_today, 
                                               sys_type=self.sys_type,
                                               bc_grid=bc_grid)

            # perform photometry and binarity selections
            pop_today = apogee.binary_select(pop_today)
            pop_today = apogee.phot_select(pop_today)

            n_sample += self.n_samp
            n_sys += len(pop_today.loc[pop_today.apogee_select == 1])
            n_pop += len(pop_today)

            dat_store.append('pop_sim', pop_today)
            log_file.write('size of sys_type= {}\n'.format(self.sys_type))
            log_file.write(str(n_pop)+'\n')
            log_file.write('\n')
            if self.sys_type > 0:
                log_file.write('size of observed single star and binary population:\n')
                log_file.write(str(len(pop_today.loc[pop_today.obs_type == 0]))+'\n')
                log_file.write(str(len(pop_today.loc[pop_today.obs_type == 1]))+'\n')
                log_file.write('\n')
            log_file.write('size of apogee selected log g and color population:\n')
            log_file.write(str(n_sys)+'\n')
            log_file.write('\n')
            log_file.write('\n')
            log_file.flush()

        dat_store.append('n_samp_tot', pd.DataFrame([n_sample]))
        dat_store.append('seed', pd.DataFrame([self.seed]))
        log_file.write('all done friend!')
        log_file.close()
        dat_store.close() 

