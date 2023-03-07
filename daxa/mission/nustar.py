#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/03/2023, 09:51. Copyright (c) The Contributors
from typing import List

import pandas as pd
from astropy.coordinates import BaseRADecFrame

from daxa import BaseMission


class NuSTAR(BaseMission):
    """
    
    """
    
    def __init__(self):
        pass

    @property
    def name(self) -> str:
        pass

    @property
    def coord_frame(self) -> BaseRADecFrame:
        pass

    @property
    def id_regex(self) -> str:
        pass

    @property
    def all_obs_info(self) -> pd.DataFrame:
        pass

    @all_obs_info.setter
    def all_obs_info(self, new_info: pd.DataFrame):
        pass

    def fetch_obs_info(self):
        pass

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], level: str, filename: str):
        pass

    def download(self):
        pass

