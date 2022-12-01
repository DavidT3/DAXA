#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 01/12/2022, 10:54. Copyright (c) The Contributors
import os
from typing import List, Union

import numpy as np

from daxa import BaseMission, OUTPUT
from daxa.exceptions import DuplicateMissionError, ArchiveExistsError


class Archive:
    """
    The Archive class, which is to be used to consolidate and provide some interface with a set
    of mission's data. Archives can be passed to processing and cleaning functions in DAXA, and also
    contain convenience functions for accessing summaries of the available data.

    :param List[BaseMission]/BaseMission missions: The mission, or missions, which are to be included
        in this archive - any setup processes (i.e. the filtering of data to be acquired) should be
        performed prior to creating an archive.
    :param str archive_name: The name to be given to this archive - it will be used for storage
        and identification.
    """
    def __init__(self, missions: Union[List[BaseMission], BaseMission], archive_name: str):
        """
        The init of the Archive class, which is to be used to consolidate and provide some interface with a set
        of mission's data. Archives can be passed to processing and cleaning functions in DAXA, and also
        contain convenience functions for accessing summaries of the available data.

        :param List[BaseMission]/BaseMission missions: The mission, or missions, which are to be included
            in this archive - any setup processes (i.e. the filtering of data to be acquired) should be
            performed prior to creating an archive.
        :param str archive_name: The name to be given to this archive - it will be used for storage
            and identification.
        """
        # First ensure that the missions variable is iterable even if there's only one missions that has
        #  been passed, makes it easier to generalise things.
        if isinstance(missions, BaseMission):
            missions = [missions]
        elif not isinstance(missions, list):
            raise TypeError("Please pass either a single missions class instance, or a list of missions class "
                            "instances to the 'missions' argument.")

        miss_names = [m.name for m in missions]
        if len(miss_names) != len(list(set(miss_names))):
            raise DuplicateMissionError("There are multiple instances of the same missions present in "
                                        "the 'missions' argument - only one instance of each is allowed for "
                                        "a particular archive.")

        self._archive_name = archive_name

        if not os.path.exists(OUTPUT + 'archives/' + archive_name):
            os.makedirs(OUTPUT + 'archives/' + archive_name)
        else:
            raise ArchiveExistsError("An archive named {an} already exists in the output directory "
                                     "({od}).".format(an=archive_name, od=OUTPUT + 'archives/'))

        # TODO maybe check for the existence of some late-stage product/file to see whether the archive
        #  has already been successfully generated
        # elif os.path.exists(OUTPUT + archive_name + '/')

        self._archive_path = OUTPUT + 'archives/' + archive_name + '/'

        self._missions = {m.name: m for m in missions}

        # This iterates through the missions that make up this archive, and ensures that they are 'locked'
        #  That means their observation content becomes immutable.
        for mission in self._missions.values():
            mission: BaseMission
            mission.locked = True

            # We also make sure that the data are downloaded
            if not mission.download_completed:
                mission.download()

    # Defining properties first
    @property
    def archive_name(self) -> str:
        """
        Property getter for the name assigned to this archive by the user.
        :return: The archive name.
        :rtype: str
        """
        return self._archive_name

    @property
    def top_level_path(self) -> str:
        """
        The property getter for the absolute path to the top-level DAXA storage directory.

        :return: Absolute top-level storage path.
        :rtype: str
        """
        return OUTPUT

    @property
    def archive_path(self) -> str:
        """
        The property getter for the absolute path to the output archive directory.

        :return: Absolute path to the archive.
        :rtype: str
        """
        return self._archive_path

    @property
    def mission_names(self) -> List[str]:
        """
        Property getter for the names of the missions associated with this Archive.

        :return:
        :rtype: List[str]
        """
        return [m.name for m in self._missions]

    @property
    def missions(self) -> Union[List[BaseMission], BaseMission]:
        """
        Property getter that returns either a list of missions associated with this Archive, or a single
        mission associated with this Archive (if only one mission was supplied).

        :return: Missions (or mission) associated with this archive.
        :rtype: Union[List[BaseMission], BaseMission]
        """
        if len(self._missions) == 1:
            return list(self._missions.values())[0]
        else:
            return list(self._missions.values())

    # Define the 'special' Python methods
    def __len__(self):
        """
        The result of using the Python len() command on this archive - the number of missions associated.

        :return: The number of missions in this archive.
        :rtype: int
        """
        return len(self._missions)

    def __iter__(self):
        """
        Called when initiating iterating through an archive. Resets the counter _n.
        """
        self._n = 0
        return self

    def __next__(self):
        """
        Iterates the counter _n uses it to find the name of the corresponding mission, then retrieves
        that source from the _missions dictionary. Missions are accessed using their name as a key.
        """
        if self._n < self.__len__():
            result = self.__getitem__(self.mission_names[self._n])
            self._n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, key: Union[int, str]) -> BaseMission:
        """
        This returns the relevant mission when an archive is addressed using the name of a mission as
        the key, or using an integer index.

        :param int/str key: The index or name of the mission to fetch.
        :return: The relevant mission object.
        :rtype: BaseMission
        """
        if isinstance(key, (int, np.integer)):
            mission = self._missions[self.mission_names[key]]
        elif isinstance(key, str):
            mission = self._missions[key]
        else:
            mission = None
            raise ValueError("Only a source name or integer index may be used to address an archive object")
        return mission

    def __delitem__(self, key: Union[int, str]):
        """
        This deletes a mission from the archive.

        :param int/str key: The index or name of the mission to delete.
        """
        if isinstance(key, (int, np.integer)):
            del self._missions[self.mission_names[key]]
        elif isinstance(key, str):
            del self._missions[key]
        else:
            raise ValueError("Only a mission name or integer index may be used to address an archive object")

