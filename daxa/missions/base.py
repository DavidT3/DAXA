#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/11/2022, 16:04. Copyright (c) The Contributors
import os.path
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List
import numpy as np
import pandas as pd

REQUIRED_INFO = ['ra', 'dec', 'ObsID', 'usable_science', 'start', 'duration']


class BaseMission(metaclass=ABCMeta):
    """

    """
    def __init__(self, output_archive_name: str, id_format: str, connection_url: str = None):
        # TODO Perhaps remove the connection URL, we should probably try to go through astroquery as much
        #  as possible
        self._miss_name = None
        self._miss_poss_insts = []
        self._id_format = id_format
        self._obs_info = None

        self._access_url = connection_url

        self._archive_name = output_archive_name
        # self._archive_name_version =

        self._top_level_output_path = None

        self._filter_allowed = None

    # Defining properties first
    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission.

        :return: The mission name
        :rtype: str
        """
        return self._miss_name

    @property
    def mission_instruments(self) -> List[str]:
        """
        Property getter for the names of the instruments associated with this mission which will be
        processed into the current archive by DAXA functions.

        :return: A list of instrument names
        :rtype: List[str]
        """
        return self._miss_poss_insts

    @mission_instruments.setter
    def mission_instruments(self, new_insts: List[str]):
        """
        Property setter for the instruments associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value.

        :param List[str] new_insts: The new list of instruments associated with this mission which should
            be processed into the archive.
        """
        inst_test = [ni in self._miss_poss_insts for ni in new_insts]
        if all(inst_test):
            self._miss_poss_insts = new_insts
        else:
            bad_inst = np.array(self._miss_poss_insts[np.array(inst_test)])
            raise ValueError("The following new instruments were not already associated with this mission; "
                             "{bi}".format(bi=", ".join(bad_inst)))

    @property
    def top_level_path(self) -> str:
        """
        The property getter for the absolute path to the top-level directory where any archives generated
        from this object will be stored.

        :return: Absolute top-level storage path.
        :rtype: str
        """
        return self._top_level_output_path

    @top_level_path.setter
    def top_level_path(self, new_path: str):
        """
        The property setter for the path to the top-level directory where archives generated from this
        mission are stored. Path will be checked for validity (i.e. it must exist), and the converted to
        an absolute path if not already.

        :param str new_path: The new top-level storage path for archives.
        """
        if new_path is not None and not os.path.exists(new_path):
            raise FileNotFoundError("That top-level output_path ({op}) does not exist!".format(op=new_path))
        elif new_path is not None:
            self._top_level_output_path = os.path.abspath(new_path)
        else:
            pass

    @property
    @abstractmethod
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
    @abstractmethod
    def all_obs_info(self, new_info: pd.DataFrame):
        """
        Abstract property setter (will be overwritten in every subclass) that allows the setting of a new
        all-observation-information dataframe. This is the dataframe that contains information on every
        possible observation for a mission.

        :param pd.DataFrame new_info: The new dataframe to update the all observation information.
        """
        pass

    @property
    def filter_array(self) -> np.ndarray:
        """
        A property getter for the 'filter' array, which is set by the filtering methods built-in to this class
        (or can be set externally using the filter_array property setter) and controls which observations will
        be downloaded and processed.

        :return: An array of boolean values; True means that an observation is used, False means that it is not.
        :rtype: np.ndarray
        """
        return self._filter_allowed

    @filter_array.setter
    def filter_array(self, new_filter_array: np.ndarray):
        """
        A property setter for the 'filter' array which controls which observations will be downloaded and processed.
        The new passed filter array must be an array of boolean values, where True means an observation will be used
        and False means it will not; the array must be the same length as the all_obs_info dataframe.

        :param np.ndarray new_filter_array:
        """
        if new_filter_array.dtype != bool:
            raise TypeError("Please pass an array of boolean values for the filter array.")
        elif len(new_filter_array) != len(self._obs_info):
            raise ValueError("Length of the filter array ({lf}) does not match the length of the dataframe containing"
                             " all observation information for this mission ({la}).".format(lf=len(new_filter_array),
                                                                                            la=len(self._obs_info)))
        else:
            self._filter_allowed = new_filter_array

    # Then define internal methods
    @staticmethod
    def _obs_info_base_checks(new_info: pd.DataFrame):
        """
        Performs very simple checks on new inputs into the observation information dataframe, ensuring it at
        has the minimum required columns.

        :param pd.DataFrame new_info: The new dataframe of observation information that should be checked.
        """
        if not isinstance(new_info, pd.DataFrame) or not all([col in new_info.columns for col in REQUIRED_INFO]):
            raise ValueError("New all_obs_info values must be a Pandas dataframe with AT LEAST the following "
                             "columns; {}".format(', '.join(REQUIRED_INFO)))

    # Then define user-facing methods
    @abstractmethod
    def fetch_obs_info(self):
        """
        The abstract method (i.e. will be overridden in every sub-class of BaseMission) that pulls basic information
        on all observations for a given mission down from whatever server it lives on.
        """
        # self.all_obs_info = None
        pass

    def reset_filter(self):
        """
        Very simple method which simply resets the filter array, meaning that all observations will now be
        downloaded and processed, and any filters applied to the current mission have been undone.
        """
        self._filter_allowed = np.full(len(self._obs_info), True)








