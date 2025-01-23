#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 30/11/2022, 15:30. Copyright (c) The Contributors
from typing import Union, List
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.units import Quantity

from .base import Archive
from ..mission.tools import multi_mission_filter_on_positions

def assemble_archive_from_positions(archive_name: str, positions: Union[list, np.ndarray, SkyCoord], 
                                    search_distance: Union[Quantity, float, int, list, np.ndarray, 
                                    dict] = None, missions: List[str] = None, clobber: bool = False, 
                                    download_products: Union[bool, dict] = True, 
                                    use_preprocessed: Union[bool, dict] = False) -> Archive:
    """
    Assembles an archive from all observations that have been found by searching around a position.
    By default this function will search all available missions supported by DAXA. This will set up 
    Mission objects and filter their observations based on the input positions and search_distance
    argument. If a mission does not have any observations matched after the filtering, it will not
    be included in the final archive.

    :param str archive_name: Name of the archive to be assembled, it will be used for storage
            and identification. If an existing archive with this name exists it will be read in, 
            unless clobber=True.
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
    :param bool clobber: If an archive named 'archive_name' already exists, then setting clobber to True
        will cause it to be deleted and overwritten.
    :param bool/dict download_products: Controls whether pre-processed products should be downloaded for missions
        that offer it (assuming downloading was not triggered when the missions were declared). Default is
        True, but False may also be passed, as may a dictionary of DAXA mission names with True/False values.
    :param bool/dict use_preprocessed: Whether pre-processed data products should be used rather than re-processing
        locally with DAXA. If True then what pre-processed data products are available will be automatically
        re-organised into the DAXA processed data structure during the setup of this archive. If False (the default)
        then this will not automatically be applied. Just as with 'download_products', a dictionary may be passed for
        more nuanced control, with mission names as keys and True/False as values.
    :return: An Archive object that includes all missions where observations have been found after
        filtering by the given position.
    :rtype: daxa.archive.Archive
    """
    miss_list = multi_mission_filter_on_positions(positions, search_distance, missions)
    archive = Archive(archive_name, miss_list, clobber, download_products, use_preprocessed)

    return archive
    
