#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 08/12/2022, 10:17. Copyright (c) The Contributors
from typing import Union, List
import numpy as np
from warnings import warn

from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from .mission.base import BaseMission
from .mission import MISS_INDEX
from .exceptions import NoObsAfterFilterError

def dict_search(key: str, var: dict) -> list:
    """
    This simple function was very lightly modified from a stackoverflow answer, and is an
    efficient method of searching through a nested dictionary structure for specfic keys
    (and yielding the values associated with them). In this case will extract all of a
    specific product type for a given source.

    :param key: The key in the dictionary to search for and extract values.
    :param var: The variable to search, likely to be either a dictionary or a string.
    :return list[list]: Returns information on keys and values
    """

    # Check that the input is actually a dictionary
    if hasattr(var, 'items'):
        for k, v in var.items():
            if k == key:
                yield v
            # Here is where we dive deeper, recursively searching lower dictionary levels.
            if isinstance(v, dict):
                for result in dict_search(key, v):
                    # We yield a string of the result and the key, as we'll need to return the
                    # ObsID and Instrument information from these product searches as well.
                    # This will mean the output is an unpleasantly nested list, but we can solve that.
                    yield [str(k), result]


def filter_on_positions(positions: Union[list, np.ndarray, SkyCoord], 
                       search_distance: Union[Quantity, float, int, list, np.ndarray, dict] = None,
                       missions: List[str] = None) -> dict[str, BaseMission]:
    """
    Convenience function to search around a position for observations across multiple missions. By
    default this function will search all available missions supported by DAXA. This will set up 
    Mission objects and filter their observations based on the input positions and search_distance
    argument. If a mission does not have any observations matched after the filtering, they will not
    be included in the dictionary output.

    :param list/np.ndarray/SkyCoord positions: The positions for which you wish to search for observations. They
        can be passed either as a list or nested list (i.e. [r, d] OR [[r1, d1], [r2, d2]]), a numpy array, or
        an already defined SkyCoord. If a list or array is passed then the coordinates are assumed to be in
        degrees, and the default mission frame will be used.
    :param Quantity/float/int/list/np.ndarray/dict search_distance: The distance within which to search for
        observations by this mission. Distance may be specified either as an Astropy Quantity that can be
        converted to degrees (a float/integer will be assumed to be in units of degrees), as a dictionary of
        quantities/floats/ints where the keys are names of different instruments (possibly with different field
        of views), or as a non-scalar Quantity, list, or numpy array with one entry per set of coordinates (for
        when you wish to use different search distances for each object). The default is None, in which case a
        value of 1.2 times the approximate field of view defined for each instrument will be used; where different
        instruments have different FoVs, observation searches will be undertaken on an instrument-by-instrument
        basis using the different field of views.
    :param list[str] missions: list of mission names that will have the filter performed on. If set 
        to None, this function will perform the search on all missions available within DAXA.
    :return: A dictionary of missions that have observations associated with them. The keys are 
        strings of the mission's names, and the values are the Mission objects that have had the
        filtering applied and have found matching observations.
    :rtype: dict[str, daxa.mission.BaseMission]
    """
    # TODO should we allow custom search distances for different telescopes here?

    # User inputs to the positions and search_distance argument are checked within the 
    # BaseMission.filter_on_positions method, so we dont check them here

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
    mission_dict = {}
    for mission_key in mission_keys:
        mission = MISS_INDEX[mission_key]()
        try:
            mission.filter_on_positions(positions, search_distance)
            mission_dict[mission_key] = mission
        except NoObsAfterFilterError:
            warn(f'No observations found after the filter for {mission_key}, will not be included '
                 'in the output dictionary.', stacklevel=2)
            continue
        # All other errors are to do with the user input arguments to positions and 
        # search_distance so we still want to raise those
    
    return mission_dict