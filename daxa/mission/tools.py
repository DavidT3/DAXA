from typing import Union, List
import numpy as np
from warnings import warn

from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from .base import BaseMission
from . import MISS_INDEX
from ..exceptions import NoObsAfterFilterError

def multi_mission_filter_on_positions(positions: Union[list, np.ndarray, SkyCoord], 
                                      search_distance: Union[Quantity, float, int, list, 
                                      np.ndarray, dict] = None, missions: List[str] = None
                                      ) -> list[BaseMission]:
    """
    Convenience function to search around a position for observations across multiple missions. By
    default this function will search all available missions supported by DAXA. This will set up 
    Mission objects and filter their observations based on the input positions and search_distance
    argument. If a mission does not have any observations matched after the filtering, they will not
    be included in the list output.

    :param list/np.ndarray/SkyCoord positions: The positions for which you wish to search for 
        observations. They can be passed either as a list or nested list (i.e. [r, d] OR [[r1, d1],
        [r2, d2]]), a numpy array, or an already defined SkyCoord. If a list or array is passed then
        the coordinates are assumed to be in degrees, and the default mission frame will be used.
    :param Quantity/dict search_distance: The distance to search for observations within, the default 
        is None in which case standard search distances for different missions are used. The user 
        may pass a single Quantity to use for all missions or a dictionary with keys corresponding to 
        ALL or SOME of the missions specified by the 'mission' argument. In the case where only SOME
        of the missions are specified in a distance dictionary, the default DAXA values will be used
        for any that are missing. When specifying a search distance for a specific mission, this may 
        be either an Astropy Quantity that can be converted to degrees (a float/integer will be 
        assumed to be in units of degrees), as a dictionary of quantities/floats/ints where the keys
        are names of different instruments (possibly with different field of views), or as a 
        non-scalar Quantity, list, or numpy array with one entry per set of coordinates (for when 
        you wish to use different search distances for each object). The default is None, in which 
        case a value of 1.2 times the approximate field of view defined for each instrument will be 
        used; where different instruments have different FoVs, observation searches will be 
        undertaken on an instrument-by-instrument basis using the different field of views.
    :param list[str] missions: list of mission names that will have the filter performed on. If set 
        to None, this function will perform the search on all missions available within DAXA.
    :return: A list of missions that have observations associated with them. The list contains 
        Mission objects that have had the filtering applied and have found matching observations.
    :rtype: List[daxa.mission.BaseMission]
    """
    # Checking the user inputs to the search_distance argument, we only check the mission level
    # requirements to the input, for instrument specific checks, the indiviual filter_on_positions
    # methods for each mission check this
    if isinstance(search_distance, dict):
        # check that all the keys are valid missions
        if not all(miss in MISS_INDEX.keys() for miss in search_distance.keys()):
            raise ValueError("Keys of the search_distance input ductionary not recognised, "
                             f"acceptable input missions are as follows: {MISS_INDEX.keys()}")
        # In the case that some missions are specified but not all of them, we need to use the 
        # default search distances
        if len(search_distance) != len(MISS_INDEX):
            for miss in MISS_INDEX.keys():
                if miss not in search_distance.keys():
                    # By setting this to None, DAXA will use the default values
                    search_distance[miss] = None

    # For all input types into search_distance we translate into a dictionary, so that the argument
    # can be used generically later in the code
    elif isinstance(search_distance, Quantity):
        search_distance = {miss: search_distance for miss in MISS_INDEX.keys()}

    elif search_distance is None:
        search_distance = {miss: None for miss in MISS_INDEX.keys()}

    else:
        raise ValueError("The search distance argument must be input as either a Quantity, to be " 
                         "applied to all missions, a dictionary with keys of some or all missions, " 
                         "or left to None such that the DAXA default search distances are used.")

    # Checking inputs to missions argument
    if missions is not None:
        if not isinstance(missions, list):
            raise ValueError("The missions argument must be input as a list of strings.")

        if not all(isinstance(miss, str) for miss in missions):
            raise ValueError("The missions argument must be input as a list of strings.")
        
        if not all(miss in MISS_INDEX.keys() for miss in missions):
            raise ValueError("Input missions not recognised, acceptable input missions are as follows:"
                            f"{MISS_INDEX.keys()}")
        else:
            mission_keys = missions
            
    else:
        mission_keys = MISS_INDEX.keys()

    # This will be appended to if observations are found for a mission
    mission_list = []
    for mission_key in mission_keys:
        mission = MISS_INDEX[mission_key]()
        try:
            mission.filter_on_positions(positions, search_distance[mission_key])
            mission_list.append(mission)
        except NoObsAfterFilterError:
            warn(f'No observations found after the filter for {mission_key}, will not be included '
                 'in the output dictionary.', stacklevel=2)
            continue
        # All other errors are to do with the user input arguments to positions and 
        # search_distance so we still want to raise those
    
    return mission_list