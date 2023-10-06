#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 06/10/2023, 16:04. Copyright (c) The Contributors

from typing import List, Union

from astropy.coordinates import BaseRADecFrame, FK5
from astropy.units import Quantity

from daxa.mission.base import BaseMission


# Don't require that the event_cl directory be present (cleaned events), as we download the level-1 data (event_uf)
#  and process it ourselves - THAT IS UNLESS the user wants to download the processed data
# REQUIRED_DIRS = {'raw': ['auxil/', 'event_uf/', 'hk/'],
#                  'processed': ['auxil/', 'event_uf/', 'event_cl/', 'hk/']}


class Swift(BaseMission):
    """
    The mission class for observations by the Neil Gehrels Swift Observatory observations.
    The available observation information is fetched from the HEASArc SWIFTMASTR table, and data are downloaded from
    the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include XRT, BAT, and UVOT (the default
        is both XRT and BAT).
    """

    def __init__(self, insts: Union[List[str], str] = None):
        """
        The mission class for observations by the Neil Gehrels Swift Observatory observations.
        The available observation information is fetched from the HEASArc SWIFTMASTR table, and data are downloaded
        from the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include XRT, BAT, and UVOT (the default
            is both XRT and BAT).
        """
        super().__init__()

        # Sets the default instruments - the two X-ray (though BAT sort of tends towards low energy gamma rays as
        #  well) instruments on Swift.
        # TODO decide whether UV data should be acquired as default considering this module focuses on X-rays
        if insts is None:
            insts = ['XRT', 'BAT']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - Swift has a focusing X-ray telescope (XRT), the burst
        #  alert telescope (BAT) which observes in the hard X-ray (15-150keV) and up to 500keV for non-imaging
        #  studies, and a UV telescope very similar to the optical monitor on XMM (but designed better).
        self._miss_poss_insts = ['XRT', 'BAT', 'UVOT']
        # As far as I know there aren't any other common names for the instruments on Swift
        self._alt_miss_inst_names = {}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = []

        # 'proprietary_end_date', 'exposure_a', 'exposure_b', 'ontime_a',
        # 'ontime_b', 'nupsdout', 'issue_flag', 'target_category',
        # 'proprietary_usable'

        # Runs the method which fetches information on all available Swift observations and stores that
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
        self._miss_name = "swift"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "Swift"
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
        #  the BaseMission superclass - Swift observations have a unique 11-digit ObsID, the construction of
        #  which is discussed here (https://heasarc.gsfc.nasa.gov/w3browse/swift/swiftmastr.html#ObsID)
        self._id_format = '^[0-9]{11}$'
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
        #  new mission class. XRT is described here (https://swift.gsfc.nasa.gov/about_swift/xrt_desc.html),
        #  UVOT is described here (https://swift.gsfc.nasa.gov/about_swift/uvot_desc.html), and BAT is described
        #  here (https://swift.gsfc.nasa.gov/about_swift/bat_desc.html).
        # BAT is somewhat complicated, because the half-coded region (which can do imaging) has a 100x60deg FoV, so I
        #  have gone with half the long side
        self._approx_fov = {'XRT': Quantity(11.8, 'arcmin'), 'BAT': Quantity(50, 'arcmin'),
                            'UVOT': Quantity(8.5, 'arcmin')}
        return self._approx_fov

