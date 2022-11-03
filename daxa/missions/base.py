#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/11/2022, 16:04. Copyright (c) The Contributors
import os.path
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import List
import numpy as np


class BaseMission(metaclass=ABCMeta):
    """

    """
    def __init__(self, output_archive_name: str, connection_url: str, id_format: str):

        self._miss_name = None
        self._miss_poss_insts = []
        self._id_format = id_format
        self._obs_ids = []
        self._obs_cen_coords = None

        self._access_url = connection_url

        self._archive_name = output_archive_name
        # self._archive_name_version =

        self._top_level_output_path = None

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

    # Then define methods











