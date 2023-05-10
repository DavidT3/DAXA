# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by Jessica E Pilling (jp735@sussex.ac.uk) Wed May 10 2023, 11:22. Copyright (c) The Contributors
from typing import List

from astropy.io import fits

from daxa import NUM_CORES


def parse_erositacalpv_sum(raw_obs_path: str):
    """
    A function that takes a path to raw eROSITA Calibration and Performance validtion data
    that has been filtered for user's instrument choice. The header of the data will be read in 
    and parsed so that information relevant to DAXA processing valid scientific observations can
    be extracted. This includes information such aswhether the instrument was active, is the instrument 
    included in this observation, and whether the fitler wheel was closed or on the calibration source.

    :param str sum_path: The path to the raw data file whose header is to be parsed into a dictionary
        of relevant information.
    :return: Multi-level dictionary of information, with top-level keys being instrument names. Next 
        level contains information on whether the instrument was active, is it included in the instrument
        filtered raw data file for that observation, and what filter was used for this instrument (this 
        will be the same for all instruments in one observation, but the info is stored at this level to 
        adhere to the DAXA formatting.)
    :rtype: dict
    """
    # Defining the dictionary to be returned
    info_dict = {}

    # Reading in the data
    with fits.open(raw_obs_path) as fits_file:
        # Getting insts associated with obs
        data = fits_file[1].data
        t_col = data["TM_NR"]
        filter = fits_file[0].header['FILTER']
        # Getting a numpy array of the unique TM NRs, hence a list of the instruments
        insts = t_col.unique()
        for inst in insts:
            info_dict["TM" + str(inst)] = {}
            info_dict["TM" + str(inst)]['active'] = True
            info_dict["TM" + str(inst)]['included'] = True
            info_dict["TM" + str(inst)]['filter'] = filter
        
    return info_dict
        


            





    
         
        


    



    
    