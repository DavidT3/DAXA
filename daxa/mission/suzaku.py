#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 09/10/2023, 14:44. Copyright (c) The Contributors

import gzip
import io
import os
from datetime import datetime
from multiprocessing import Pool
from shutil import copyfileobj
from typing import List, Union, Any
from warnings import warn

import pandas as pd
import requests
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from bs4 import BeautifulSoup
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission

# https://heasarc.gsfc.nasa.gov/docs/suzaku/analysis/abc/node6.html#SECTION00610000000000000000
# REQUIRED_DIRS = {'raw': ['auxil/', 'event_uf/', 'hk/'],
#                  'processed': ['auxil/', 'event_uf/', 'event_cl/', 'hk/']}


class Suzaku(BaseMission):
    """
    The mission class for Suzaku observations, specifically those from the XIS instruments, as XRS' cooling system
    was damaged soon after launch, and HXD was not an imaging instrument.
    The available observation information is fetched from the HEASArc SUZAMASTER table, and data are downloaded from
    the HEASArc https access to their FTP server.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include XIS0, XIS1, XIS2, and XIS3 (the
            default is all of them).
    """

    def __init__(self, insts: Union[List[str], str] = None):
        """
        The mission class for Suzaku observations, specifically those from the XIS instruments, as XRS' cooling system
        was damaged soon after launch, and HXD was not an imaging instrument.
        The available observation information is fetched from the HEASArc SUZAMASTER table, and data are downloaded
        from the HEASArc https access to their FTP server.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include XIS0, XIS1, XIS2, and XIS3 (the
            default is all of them).
    """
        super().__init__()

        # Sets the default instruments - all the imaging spectrometers on Suzaku, and the only instruments supported
        #  by DAXA
        if insts is None:
            insts = ['XIS0', 'XIS1', 'XIS2', 'XIS3']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - the XIS instruments all had their own telescopes
        self._miss_poss_insts = ['XIS0', 'XIS1', 'XIS2', 'XIS3']
        # The chosen_instruments property setter (see below) will use these to convert possible contractions
        #  to the names that the module expects. I'm not that familiar with Suzaku currently, so
        #  I've just put in X0, X1, ... without any real expectation that anyone would use them.
        self._alt_miss_inst_names = {'X0': 'XIS0', 'X1': 'XIS1', 'X2': 'XIS2', 'X3': 'XIS3'}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available pointed Suzaku observations and stores that
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
        self._miss_name = "suzaku"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "Suzaku"
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
        #  the BaseMission superclass - Suzaku observations seem to have a unique 9-digit ObsID, though I can find
        #  no discussion of whether there is extra information in the ObsID (i.e. target type).
        self._id_format = '^[0-9]{9}$'
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
        #  new mission class - found slightly conflicting values for this, but went with the HEASArc info page's
        #  number of 19'x19' (https://heasarc.gsfc.nasa.gov/docs/suzaku/about/overview.html). Only need one value
        #  here because we're just supporting the XIS instruments for this mission
        self._approx_fov = Quantity(9.5, 'arcmin')
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

