#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/11/2022, 15:58. Copyright (c) The Contributors
from datetime import datetime
from warnings import warn

import numpy as np
import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astroquery import log
from astroquery.esa.xmm_newton import XMMNewton as AQXMMNewton

from .base import BaseMission

log.setLevel(0)


class XMMPointed(BaseMission):
    """

    """
    def __init__(self, output_archive_name: str, output_path: str = None):
        super().__init__(output_archive_name, '')

        self._required_mission_specific_cols = ['proprietary_end_date', 'usable_proprietary']
        # if output_path is None:
        # output_path =
        self.fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array of all True, rather than
        #  the initial None value
        self.reset_filter()

    # Defining properties first
    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission.

        :return: The mission name
        :rtype: str
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_name = "XMM Pointed"
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
        #  the BaseMission superclass
        self._id_format = '^[0-9]{10}$'
        return self._id_format

    @property
    def all_obs_info(self) -> pd.DataFrame:
        """
        A property getter that returns the base dataframe containing information about all the observations available
        for an instance of a mission class.

        :return: A pandas dataframe with (at minimum) the following columns; 'ra', 'dec', 'ObsID', 'usable_science',
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
        self._obs_info_checks(new_info)
        self._obs_info = new_info

    def fetch_obs_info(self):
        """

        """
        count_tab = AQXMMNewton.query_xsa_tap('select count(observation_id) from xsa.v_all_observations')
        num_obs = np.ceil(count_tab['count'].tolist()[0] / 1000).astype(int) * 1000
        # footprint_fov, stc_s
        obs_info = AQXMMNewton.query_xsa_tap("select TOP {} ra, dec, observation_id, start_utc, with_science, "
                                             "duration, proprietary_end_date from v_all_observations".format(num_obs))
        obs_info_pd: pd.DataFrame = obs_info.to_pandas()

        obs_info_pd['proprietary_end_date'] = pd.to_datetime(obs_info_pd['proprietary_end_date'], utc=False,
                                                             errors='coerce')
        obs_info_pd['start_utc'] = pd.to_datetime(obs_info_pd['start_utc'], utc=False, errors='coerce')
        today = datetime.today()

        obs_info_pd['usable_proprietary'] = obs_info_pd['proprietary_end_date'].apply(
            lambda x: ((x <= today) & (pd.notnull(x)))).astype(bool)

        obs_info_pd = obs_info_pd.rename(columns={'observation_id': 'ObsID', 'with_science': 'usable_science',
                                                  'start_utc': 'start'})

        # Converting the duration column to a timedelta object, which can then be directly added to the start column
        #  which should be a datetime object itself
        obs_info_pd['duration'] = pd.to_timedelta(obs_info_pd['duration'], 's')
        # Now creating an end column by adding duration to start
        obs_info_pd['end'] = obs_info_pd.apply(lambda x: x.start + x.duration, axis=1)

        obs_info_pd_cleaned = obs_info_pd[(~obs_info_pd['ra'].isna()) | (~obs_info_pd['dec'].isna())]

        if len(obs_info_pd_cleaned) != len(obs_info_pd):
            warn("{ta} of the {tot} observations located for this mission have been discarded due to NaN "
                 "RA or Dec values".format(ta=len(obs_info_pd)-len(obs_info_pd_cleaned), tot=len(obs_info_pd)),
                 stacklevel=2)

        self.all_obs_info = obs_info_pd_cleaned
