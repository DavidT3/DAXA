#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 25/07/2023, 06:34. Copyright (c) The Contributors

from astropy.coordinates import BaseRADecFrame, FK5

from daxa.mission.base import BaseMission


class ROSATAllSky(BaseMission):
    """

    No instrument choice is offered for this mission class because all RASS observations were taken with PSPC-C.
    """

    def __init__(self):
        """

        """
        super().__init__()

        # Sets the default instrument - I have this in the same format (i.e. a list) as every other mission class, but
        #  given that the RASS data were all taken with PSPC I don't give the user a choice of instruments.
        insts = ['PSPC']

        # These are the allowed instruments for this mission - again it is just PSPC, but the mission class expects
        #  this attribute to be set
        self._miss_poss_insts = ['PSPC']
        # There are no alternative instrument names, especially because the user can't set the instruments.
        self._alt_miss_inst_names = {}

        # Setting the chosen instruments property, still using the BaseMission infrastructure even though we know
        #  there will only ever be the PSPC instrument for this mission
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # TODO Revisit this when I've explored what is actually in the table for RASS
        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['proprietary_end_date', 'target_category', 'detector', 'grating',
                                                'data_mode', 'proprietary_usable']

        # Runs the method which fetches information on all available RASS observations and stores that
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
        self._miss_name = "rosat_all_sky"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "RASS"
        return self._miss_name

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission. Not completely
        certain that FK5 is the correct frame for RASS, but a processed image downloaded from HEASArc used FK5 as
        the reference frame for its WCS.

        :return: The coordinate frame of the RA-Dec.
        :rtype: BaseRADecFrame
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_coord_frame = FK5
        return self._miss_coord_frame

