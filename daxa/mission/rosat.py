#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 25/07/2023, 06:12. Copyright (c) The Contributors

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

        # TODO Assess whether I actually need this here
        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        # This instrument stuff is down here because for Chandra I want it to happen AFTER the Observation info
        #  table has been fetched. As Chandra uses one instrument per observation, this will effectively be another
        #  filtering operation rather than the download-time operation is has been for NuSTAR for instance
        self.chosen_instruments = insts
