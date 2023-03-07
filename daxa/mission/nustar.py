#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/03/2023, 13:45. Copyright (c) The Contributors
import io
from typing import List
from urllib.request import urlopen

import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table

from daxa import BaseMission


class NuSTAR(BaseMission):
    """
    
    """
    
    def __init__(self):
        super().__init__()
        pass

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
        self._miss_name = "nustar"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "NuSTAR"
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
        #  the BaseMission superclass - NuSTAR observations have a unique 11-digit ObsID, the construction of
        #  which is discussed here (https://heasarc.gsfc.nasa.gov/W3Browse/nustar/numaster.html#obsid)
        self._id_format = '^[0-9]{11}$'
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
        # Frankly I'm not really sure why I made this an abstract method, but possibly because I thought some
        #  missions might need extra checks run on their observation information dataframes?
        # This _obs_info_checks method is defined in BaseMission, and uses the ObsID regex defined near the top of
        #  this class to ensure that the dataframe's ObsID column contains legal values.
        self._obs_info_checks(new_info)
        self._obs_info = new_info

    def fetch_obs_info(self):
        """
        This method adapts the 'browse_extract.pl' script (a copy of which can be found in daxa/files for the proper
        credit) to acquire the 'numaster' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available, proprietary, and scheduled NuSTAR observations, with important information such as pointing
        coordinates, ObsIDs, and exposure.
        """
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited, as we don't know how many NuSTAR observations there are, and the number will
        #  increase with time (so long as the telescope doesn't break...)
        result_max ="&ResultMax=0"
        action = "&Action=Query"
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20numaster"

        fetch_url = host_url + table_head + action + result_max + down_form

        # , stream=True
        # with requests.get(fetch_url) as streamo
        with urlopen(fetch_url) as urlo:
            with fits.open(io.BytesIO(urlo.read())) as full_fits:
                # print(full_fits[0].header)
                # print('\n\n\n')
                # print(full_fits[1].header)
                nustar_tab = Table(full_fits[1].data)

        return nustar_tab

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], level: str, filename: str):
        pass

    def download(self):
        pass

