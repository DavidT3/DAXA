#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 09/10/2023, 23:36. Copyright (c) The Contributors

from typing import List, Union

import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.units import Quantity

from daxa.mission.base import BaseMission

# The
REQUIRED_DIRS = {'all': ['auxil/'],
                 'raw': {'uvot': ['hk/', 'event/', 'image/'],
                         'xrt': ['event/', 'hk/', 'image/'],
                         'bat': ['event/', 'rate/', 'survey/']},
                 'processed': {'uvot': ['hk/', 'event/', 'image/', 'products/'],
                               'xrt': ['event/', 'hk/', 'image/', 'products/'],
                               'bat': ['event/', 'rate/', 'survey/']}}


class INTEGRALPointed(BaseMission):
    """
    The mission class for pointed observations by the INTErnational Gamma-Ray Astrophysics Laboratory (INTEGRAL); i.e.
    observations taken when the spacecraft isn't slewing, and is not in an engineering mode (and is public).
    The available observation information is fetched from the HEASArc INTSCWPUB table, and data are downloaded from
    the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

    NOTE - This class treats Science Window IDs as 'obs ids', though observation IDs are a separate concept in the
    HEASArc table at least. The way DAXA is set up however, science window IDs are a closer analogy to the ObsIDs
    used by the rest of the missions.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include JEMX1, JEMX2, ISGRI, PICsIT, and
        SPI (the default is JEMX1, JEMX2, and ISGRI). OMC is not supported by DAXA.
    """

    def __init__(self, insts: Union[List[str], str] = None):
        """
        The mission class for pointed observations by the INTErnational Gamma-Ray Astrophysics Laboratory
        (INTEGRAL); i.e. observations taken when the spacecraft isn't slewing, and is not in an engineering
        mode (and is public). The available observation information is fetched from the HEASArc INTSCWPUB table, and
        data are downloaded from the HEASArc https access to their FTP server. Proprietary data are not currently
        supported by this class.

        NOTE - This class treats Science Window IDs as 'obs ids', though observation IDs are a separate concept in the
        HEASArc table at least. The way DAXA is set up however, science window IDs are a closer analogy to the ObsIDs
        used by the rest of the missions.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include JEMX1, JEMX2, ISGRI, PICsIT, and
            SPI (the default is JEMX1, JEMX2, and ISGRI). OMC is not supported by DAXA.
        """
        super().__init__()

        # Sets the default instruments - the two soft X-ray (JEMX) instruments, and the ISGRI hard X-ray to low-energy
        #  gamma ray detector
        if insts is None:
            insts = ['JEMX1', 'JEMX2', 'ISGRI']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - INTEGRAL has two coded mask soft-band X-ray
        #  instruments (JEMX1&2), the two IBIS instruments (ISGRI and PICsIT), and SPI (8 keV - 8 MeV spectrometer)
        self._miss_poss_insts = ['JEMX1', 'JEMX2', 'ISGRI', 'PICsIT', 'SPI']
        # As far as I know there aren't any other common names for the instruments on INTEGRAL
        self._alt_miss_inst_names = {}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available INTEGRAL observations and stores that
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
        self._miss_name = "integral_pointed"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "INTEGRAL Pointed"
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
        #  the BaseMission superclass - INTEGRAL splits data into 'science windows', which each have a unique ID, and
        #  this is what we're using as an ObsIDs. They are 12 digits, and have information embedded in them:
        #  https://heasarc.gsfc.nasa.gov/W3Browse/integral/intscwpub.html#ScW_ID
        # ObsIDs are also a concept for INTEGRAL, but don't necessarily seem to be unique identifiers of data chunks
        #  like they are for the rest of the missions in DAXA - SCW IDs are closer to that behaviour
        self._id_format = '^[0-9]{12}$'
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
        # These definitions of field of view are taken from various sources, and don't necessarily represent the
        #  theoretical maximum field of view of the instruments.
        # JEMX info is taken from https://www.cosmos.esa.int/web/integral/instruments-jemx, and we choose to use
        #  the 'fully illuminated' value, where the FoV has a diameter of 4.8deg
        # IBIS info is taken from https://www.cosmos.esa.int/web/integral/instruments-ibis, and we choose to use the
        #  'fully coded' value, where the FoV is 8.3x8.0deg
        # SPI info is taken from https://www.cosmos.esa.int/web/integral/instruments-spi, and we choose to use the
        #  'fully coded flat to flat' value, where the FoV is 14deg in diameter
        self._approx_fov = {'JEMX1': Quantity(2.4, 'deg'), 'JEMX2': Quantity(2.4, 'deg'),
                            'ISGRI': Quantity(4.15, 'deg'), 'PICsIT': Quantity(4.15, 'deg'),
                            'SPI': Quantity(7, 'deg')}
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
