#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 04/11/2022, 16:44. Copyright (c) The Contributors
from datetime import datetime

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
        super().__init__(output_archive_name, '', '^[0-9]{10}$')

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
        Abstract property getter for the coordinate frame of the RA-Decs of the observations of this mission. Must
        be overwritten in any subclass. This is to ensure that any subclasses that people might add will definitely
        set a coordinate frame, which is not guaranteed by having it done in the init.

        :return: The coordinate frame of the RA-Dec
        :rtype: BaseRADecFrame
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_coord_frame = FK5
        return self._miss_coord_frame

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
        today = datetime.today()

        obs_info_pd['usable_proprietary'] = obs_info_pd['proprietary_end_date'].apply(
            lambda x: ((x <= today) & (pd.notnull(x)))).astype(bool)

        obs_info_pd = obs_info_pd.rename(columns={'observation_id': 'ObsID', 'with_science': 'usable_science',
                                                  'start_utc': 'start'})

        self.all_obs_info = obs_info_pd
