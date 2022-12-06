#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 06/12/2022, 15:14. Copyright (c) The Contributors
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
        # First ensure that the missions variable is iterable even if there's only one mission that has
        #  been passed, makes it easier to generalise things.
        if isinstance(missions, BaseMission):
            missions = [missions]
        elif not isinstance(missions, list):
            raise TypeError("Please pass either a single missions class instance, or a list of missions class "
                            "instances to the 'missions' argument.")

        # Here we ensure that there are no duplicate mission instances, each mission should be filtered in such
        #  a way that all observations for that mission are in one mission instance
        miss_names = [m.name for m in missions]
        if len(miss_names) != len(list(set(miss_names))):
            raise DuplicateMissionError("There are multiple instances of the same missions present in "
                                        "the 'missions' argument - only one instance of each is allowed for "
                                        "a particular archive.")

        # Store the archive name in an attribute
        self._archive_name = archive_name

        # Then make sure that the path to store the archive is created, and that it hasn't been created
        #  before, which would mean an existing archive with the same name
        if not os.path.exists(OUTPUT + 'archives/' + archive_name):
            os.makedirs(OUTPUT + 'archives/' + archive_name)
        else:
            raise ArchiveExistsError("An archive named {an} already exists in the output directory "
                                     "({od}).".format(an=archive_name, od=OUTPUT + 'archives/'))
        # TODO maybe check for the existence of some late-stage product/file to see whether the archive
        #  has already been successfully generated
        # elif os.path.exists(OUTPUT + archive_name + '/')

        # An attribute for the path to the particular archive directory is also setup, as it's a very useful
        #  piece of knowledge
        self._archive_path = OUTPUT + 'archives/' + archive_name + '/'

        # The mission instances (or single instance) used to create the archive are stored in a dictionary, with
        #  the key being the internal DAXA name for that mission
        self._missions = {m.name: m for m in missions}

        # An attribute to store a command queue for those missions which have a command line processing
        #  backend (like XMM for instance)

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
        return [m for m in self._missions]

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

    # Then define internal methods

    # Then define user-facing methods
    def get_processed_data_path(self, mission: [BaseMission, str] = None, obs_id: str = None):
        """
        This method is to construct paths to directories where processed data for a particular mission + observation
        ID combination will be stored. That functionality is added here so that any change to how those directories
        are named will take place in only one part of DAXA, and will propagate to other parts of the module. It is
        unlikely that a user will need to directly use this method.

        If no mission is passed, then no observation ID may be passed. In the case of 'mission' and 'obs_id' being
        None, the returned string will be constructed ready to format; {mn} should be replaced by the DAXA mission
        name, and {oi} by the relevant ObsID.

        :param BaseMission/str mission: The mission for which to retrieve the processed data path. Default is None
            in which case a path ready to be formatted with a mission name will be provided.
        :param str obs_id: The ObsID for which to retrieve the processed data path, cannot be set if 'mission' is
            set to None. Default is None, in which case a path ready to be formatted with an observation ID will
            be provided.
        :return: The requested path.
        :rtype: str
        """
        # Make sure that mission is not Null whilst obs_id has been set
        if mission is None and obs_id is not None:
            raise ValueError("The obs_id argument may only be set if the mission argument is set.")

        # Extract an internal DAXA name for the mission, regardless of whether a string name or an actual
        #  mission was passed to this method
        if isinstance(mission, BaseMission):
            m_name = mission.name
        else:
            m_name = mission

        # Need to check that the mission is actually associated with this archive.
        if m_name not in self:
            raise ValueError("The mission {m} is not a part of this Archive; the current missions are "
                             "{cm}".format(m=m_name, cm=', '.join(self.mission_names)))

        # Need to make sure that the passed ObsID (if one was passed) is in the correct format for the
        #  specified mission - can use this handy method for that.
        if obs_id is not None:
            self[m_name].check_obsid_pattern(obs_id)

        # Now we just run through the different possible combinations of circumstances.
        base_path = self.archive_path+'processed_data/{mn}/{oi}/'
        if mission is not None and obs_id is not None:
            ret_str = base_path.format(mn=m_name, oi=obs_id)
        elif mission is not None and obs_id is None:
            ret_str = base_path.format(mn=m_name, oi='{oi}')
        else:
            ret_str = base_path

        return ret_str

    def info(self):
        """
        A simple method to present summary information about this archive.
        """
        print("\n-----------------------------------------------------")
        print("Number of missions - {}".format(len(self)))
        print("Total number of observations - {}".format(sum([len(m) for m in self._missions.values()])))
        for m in self._missions.values():
            print('')
            print('-- ' + m.pretty_name + ' --')
            print('   Internal DAXA name - {}'.format(m.name))
            print('   Chosen instruments - {}'.format(', '.join(m.chosen_instruments)))
            print('   Number of observations - {}'.format(len(m)))
            print('   Fully Processed - {}'.format(m.processed))
        print("-----------------------------------------------------\n")

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

