#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 09/10/2023, 17:27. Copyright (c) The Contributors

from typing import List, Union

import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.units import Quantity

from daxa.mission.base import BaseMission


# As
# REQUIRED_DIRS = {'all': ['auxil/', 'xis/'],
#                  'raw': {'xis': ['event_uf/', 'hk/', 'products/']},
#                  'processed': {'xis': ['event_uf/', 'event_cl/', 'hk/', 'products/']}}


class ASCA(BaseMission):
    """
    The mission class for ASCA observations, both from the GIS AND SIS instruments.
    The available observation information is fetched from the HEASArc ASCAMASTER table, and data are downloaded from
    the HEASArc https access to their FTP server.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include SIS0, SIS1, GIS2, and GIS3 (the
            default is all of them).
    """

    def __init__(self, insts: Union[List[str], str] = None):
        """
        The mission class for ASCA observations, both from the GIS AND SIS instruments.
        The available observation information is fetched from the HEASArc ASCAMASTER table, and data are downloaded from
        the HEASArc https access to their FTP server.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
                pass either a single string value or a list of strings. They may include SIS0, SIS1, GIS2, and GIS3 (the
                default is all of them).
        """
        super().__init__()

        # Sets the default instruments - all the instruments on ASCA
        if insts is None:
            insts = ['SIS0', 'SIS1', 'GIS2', 'GIS3']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - they all have their own telescopes
        self._miss_poss_insts = ['SIS0', 'SIS1', 'GIS2', 'GIS3']
        # The chosen_instruments property setter (see below) will use these to convert possible contractions
        #  to the names that the module expects. I'm not that familiar with ASCA currently, so
        #  I've just put in X0, X1, ... without any real expectation that anyone would use them.
        self._alt_miss_inst_names = {'S0': 'SIS0', 'S1': 'SIS1', 'G2': 'GIS2', 'G3': 'GIS3'}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available ASCA observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission

        :return: The mission name.
        :rtype: str
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass. Suggest keeping this in a format that would be good for a unix
        #  directory name (i.e. lowercase + underscores), because it will be used as a directory name
        self._miss_name = "asca"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "ASCA"
        return self._miss_name

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission.

        :return: The coordinate frame of the RA-Dec.
        :rtype: BaseRADecFrame
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_coord_frame = FK5
        return self._miss_coord_frame

    @property
    def id_regex(self) -> str:
        """
        Property getter for the regular expression (regex) pattern for observation IDs of this mission.

        :return: The regex pattern for observation IDs.
        :rtype: str
        """
        # The ObsID regular expression is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass - ASCA observations seem to have a unique 8-digit ObsID, though I can find
        #  no discussion of whether there is extra information in the ObsID (i.e. target type).
        self._id_format = '^[0-9]{8}$'
        return self._id_format

    @property
    def fov(self) -> Union[Quantity, dict]:
        """
        Property getter for the approximate field of view set for this mission. This is the radius/half-side-length of
        the field of view. In cases where the field of view is not square/circular, it is the half-side-length of
        the longest side.

        :return: The approximate field of view(s) for the mission's instrument(s). In cases with multiple instruments
            then this may be a dictionary, with keys being instrument names.
        :rtype: Union[Quantity, dict]
        """
        # The approximate field of view is defined here because I want to force implementation for each
        #  new mission class.
        # The instruments have different field of views but do observe simultaneously.
        # Used information from https://heasarc.gsfc.nasa.gov/docs/asca/asca.html
        # SIS instruments had a square 22x22', and GIS instruments had a 50' diameter circular FoV
        self._approx_fov = {'SIS0': Quantity(11, 'arcmin'), 'SIS1': Quantity(11, 'arcmin'),
                            'GIS2': Quantity(25, 'arcmin'), 'GIS3': Quantity(25, 'arcmin')}
        return self._approx_fov

    @property
    def all_obs_info(self) -> pd.DataFrame:
        """
        A property getter that returns the base dataframe containing information about all the observations available
        for an instance of a mission class.

        :return: A pandas dataframe with (at minimum) the following columns; 'ra', 'dec', 'ObsID', 'science_usable',
            'start', 'duration'
        :rtype: pd.DataFrame
        """
        return self._obs_info

    @all_obs_info.setter
    def all_obs_info(self, new_info: pd.DataFrame):
        """
        Property setter that allows the setting of a new all-observation-information dataframe. This is the dataframe
        that contains information on every possible observation for a mission.

        :param pd.DataFrame new_info: The new dataframe to update the all observation information.
        """
        # Frankly I'm not really sure why I made this an abstract method, but possibly because I thought some
        #  missions might need extra checks run on their observation information dataframes?
        # This _obs_info_checks method is defined in BaseMission, and uses the ObsID regex defined near the top of
        #  this class to ensure that the dataframe's ObsID column contains legal values.
        self._obs_info_checks(new_info)
        self._obs_info = new_info
        self.reset_filter()