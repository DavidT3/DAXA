#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 04/09/2024, 14:55. Copyright (c) The Contributors

import re
from typing import Union

import numpy as np
from astropy.io import fits

from daxa import BaseMission
from daxa.archive.base import Archive
from daxa.mission import eRASS1DE, eROSITACalPV


def _prepare_erosita_info(archive: Archive, mission: BaseMission):
    """
    A function to be used with in the esass_call wrapper. This is called only
    if no erosita processing has taken place yet. It populates two dictionaries
    with necessary information for esass tasks to be executed. The first is the
    _process_extra_info attribute for the given Archive, and it stores paths to raw data
    for each observation. The second dictionary is the observation summaries of the archive,
    which for erosita just parses the filter wheel status. 

    :param Archive archive: The Archive instance that has been parsed into the esass function 
        which esass_func wraps. 
    :param BaseMission mission: The eROSITACalPV mission for which this information must be prepared.
    """
    def get_obs_path(rel_miss: Union[eRASS1DE, eROSITACalPV], obs_id: str):
        """
        A function that returns the absolute raw data path for eROSITA Calibration 
        and Performance validation data, for a given mission and obs_id. Since the names of
        the data files change depending on the user's instrument choice for each mission, 
        this function is necessary for the esass functions to point to the correct raw data path.
        This method is used to populate an Archive._process_extra_info['erositacalpv']['obs']['path']
        attribute.

        :param BaseMission rel_miss: The eROSITACalPV mission containing observations with
            mission specific instrument filtering, that need paths pointing to.
        :param str obs_id: The obs_id for which the returned raw data path corresponds to.
        :return: The raw data path of the obs_id, with the appropriate instrument filtered suffix.
        :rtype: str
        """
        # In the case that no instrument filtering has taken place, the name of the raw fits file is unchanged
        #  from the output of the get_evlist_path_from_obs method
        obs_path = rel_miss.get_evt_list_path(obs_id)

        if len(rel_miss.chosen_instruments) != 7:
            # Otherwise, need to format the name of the fits file according to the instruments
            insts = rel_miss.chosen_instruments
            # Getting an ordered string of the telescope module numbers, which is how the fits
            #   file is named
            tm_nos = ''.join(sorted(re.findall(r'\d+', ''.join(insts))))
            # Reformatting the obs_path variable to include the instrument filtering suffix
            obs_path = obs_path[:-5] + '_if_{}.fits'.format(tm_nos)
        
        return obs_path
    
    def parse_erositacalpv_sum(raw_obs_path: str):
        """
        A function that takes a path to raw eROSITA Calibration and Performance validation data
        that has been filtered for user's instrument choice. The header of the data will be read in 
        and parsed so that information relevant to DAXA processing valid scientific observations can
        be extracted. This includes information such as to whether the instrument was active, is the instrument
        included in this observation, and whether the filter wheel was closed or on the calibration source.

        :param str raw_obs_path: The path to the raw data file whose header is to be parsed into a dictionary
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
            filt = fits_file[0].header['FILTER']
            # Getting a numpy array of the unique TM NRs, hence a list of the instruments
            insts = np.unique(t_col)
            for inst in insts:
                info_dict["TM" + str(inst)] = {}
                info_dict["TM" + str(inst)]['active'] = True
                info_dict["TM" + str(inst)]['filter'] = filt
            
        return info_dict
    
    # Firstly this function will populate the process_extra_info dictionary 
    #   with the necessary info needed for erositacalpv missions
    # Defining this extra_info dictionary
    extra_info = archive.process_extra_info[mission.name]

    # Next this function will fill in the observation_summaries for the archive
    # Defining the dictionary that will be assigned to the observation summary
    parsed_obs_info = {}
    parsed_obs_info[mission.name] = {}
    
    # Populating both dictionaries with the necessary info
    for obs in mission.filtered_obs_ids:
        if obs not in archive.observation_summaries[mission.name]:
            # getting the raw data path to each observation
            path_to_obs = get_obs_path(mission, obs)
            # Adding in the obs_id keys into the extra_info dictionary
            extra_info[obs] = {}
            # Adding in the path key and value into the obs layer of extra_info
            extra_info[obs]['path'] = path_to_obs
            # Then adding to the parsed_obs_info dictionary
            parsed_obs_info[mission.name][obs] = parse_erositacalpv_sum(path_to_obs)

    archive.observation_summaries = parsed_obs_info
