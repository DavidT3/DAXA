#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 26/01/2024, 14:51. Copyright (c) The Contributors
import os
from shutil import rmtree
from typing import List, Union, Tuple
from warnings import warn

import numpy as np
from astropy import wcs
from astropy.units import Quantity
from regions import Region, read_ds9, PixelRegion, write_ds9

from daxa import BaseMission, OUTPUT
from daxa.exceptions import DuplicateMissionError, ArchiveExistsError, NoProcessingError, NoDependencyProcessError, \
    ObsNotAssociatedError
from daxa.misc import dict_search


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
    def __init__(self, missions: Union[List[BaseMission], BaseMission], archive_name: str, clobber: bool = False):
        """
        The init of the Archive class, which is to be used to consolidate and provide some interface with a set
        of mission's data. Archives can be passed to processing and cleaning functions in DAXA, and also
        contain convenience functions for accessing summaries of the available data.

        :param List[BaseMission]/BaseMission missions: The mission, or missions, which are to be included
            in this archive - any setup processes (i.e. the filtering of data to be acquired) should be
            performed prior to creating an archive.
        :param str archive_name: The name to be given to this archive - it will be used for storage
            and identification.
        :param bool clobber: If an archive with named 'archive_name' already exists, then setting clobber to True
            will cause it to be deleted and overwritten.
        """
        # First ensure that the missions variable is iterable even if there's only one mission that has
        #  been passed, makes it easier to generalise things.
        if isinstance(missions, BaseMission):
            missions = [missions]
        
        # Then checking that every element in the list is a BaseMission
        if not all(isinstance(mission, BaseMission) for mission in missions):
            raise TypeError("Please pass either a single mission class instance, or a list of missions class "
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
        elif os.path.exists(OUTPUT + 'archives/' + archive_name) and clobber:
            warn("An archive called {an} already existed, but as clobber=True it has been deleted and "
                 "overwritten.".format(an=archive_name), stacklevel=2)
            rmtree(OUTPUT + 'archives/' + archive_name)
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

        # These attributes are to store outputs from command-line based processes (such as the SAS processing
        #  tools for XMM missions). Top level keys are mission names, one level down from that uses process names
        #  as keys (the function name; e.g. cif_build), and one level down from that uses either an ObsID or ObsID
        #  + instrument combo as keys.
        # The _process_success_flags dictionary stores whether the process was successful, which means that the
        #  final output file exists, and that there were no errors from stderr
        self._process_success_flags = {mn: {} for mn in self.mission_names}
        # The _process_errors dictionary stores any error outputs that may have been generated, _process_warnings
        #  stores any warnings that (hopefully) aren't serious enough to rule that a process run was a complete
        #  failure, and _process_logs stores any relevant logs (mostly stdout for cmd line based tools) for
        #  each process
        self._process_errors = {mn: {} for mn in self.mission_names}
        self._process_warnings = {mn: {} for mn in self.mission_names}
        self._process_raw_errors = {mn: {} for mn in self.mission_names}  # Specifically for unparsed stderr
        self._process_logs = {mn: {} for mn in self.mission_names}

        # This attribute is used to store the 'extra info' that is sometimes passed out of processing functions (see
        # the DAXA cif_build, epchain, and emchain functions for examples).
        self._process_extra_info = {mn: {} for mn in self.mission_names}

        # This attribute will contain information on mission's observations. That could include whether a particular
        #  instrument was active for a particular observation, what sub-exposures there were (assuming there were
        #  any, XMM will often have some), what filter was applied, things like that.
        # I will attempt to normalise the information stored in here for each mission, as far as that is possible.
        self._miss_obs_summ_info = {mn: {} for mn in self.mission_names}
        # This dictionary will mimic the structure of the _miss_obs_summ_info dictionary, but will contain simple
        #  boolean information on whether the particular ObsID-instrument-sub exposure (or more likely
        #  ObsID-Instrument for most missions) should be reduced and processed for science
        self._use_this_obs = {mn: {} for mn in self.mission_names}

        # This stores the final judgement pronounced by the _final_process wrapper that should be used to decorate
        #  the last processing function for a particular mission. At the ObsID level it states whether there are
        #  any useful data (True) or whether no aspect of that observation reached the end of the final step
        #  successfully. The ObsIDs marked as False will be moved from the archive processed data directory to a
        #  separate failed data directory.
        self._final_obs_id_success = {mn: {} for mn in self.mission_names}

        # This attribute will store regions for the observations associated with different missions. By the time
        #  they are stored in this attribute they SHOULD be in RA-Dec, not in pixel coords or anything like that
        self._source_regions = {mn: {} for mn in self.mission_names}

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

    @property
    def process_success(self) -> dict:
        """
        Property getter for a nested dictionary containing boolean flags describing whether different processing steps
        applied to observations from various missions are considered to have completed successfully.

        :return: A nested dictionary where top level keys are mission names, next level keys are processing
            function names, and lowest level keys are either ObsID or ObsID+instrument names. The values
            attributed with the lowest level keys are boolean, with True indicating that the processing function
            was successful
        :rtype: dict
        """
        # Check to make sure that success information for at least one processing function on at least one mission
        #  has been added to this archive, otherwise an error is thrown.
        if sum([len(self._process_success_flags[mn]) for mn in self.mission_names]) == 0:
            raise NoProcessingError("No processing success information has been added to this archive, meaning "
                                    "that no data processing has been applied.")

        return self._process_success_flags

    @process_success.setter
    def process_success(self, process_name_success_dict: Tuple[str, dict]):
        """
        Property setter for a nested dictionary containing boolean flags describing whether different processing
        steps applied to observations from various missions are considered to have completed successfully. This
        shouldn't be used directly by a user, rather DAXA processing functions will use it themselves. This
        setter does not overwrite the existing dictionary, but rather adds extra information.

        :param Tuple[str, dict] process_name_success_dict: A tuple with the first element being the name of the
            process for which a success dictionary is being passed, and the second being the success dictionary
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        """
        # This applies checks to the input to this setter
        pr_name, success_flags = self._check_process_inputs(process_name_success_dict)

        # Iterate through the missions in the input dictionary
        for mn in success_flags:
            # If the particular process does not have an entry for the particular mission then we add it to the
            #  dictionary, but if it does then we warn the user and do nothing
            if pr_name in self._process_success_flags[mn]:
                warn("The process_success property already has an entry for {prn} under {mn}, no change will be "
                     "made.".format(prn=pr_name, mn=mn), stacklevel=2)
            else:
                self._process_success_flags[mn][pr_name] = success_flags[mn]

    @property
    def process_errors(self) -> dict:
        """
        Property getter for a nested dictionary containing error information from processing applied to mission data.

        :return: A nested dictionary where top level keys are mission names, next level keys are processing
            function names, and lowest level keys are either ObsID or ObsID+instrument names. The values
            attributed with the lowest level keys are error outputs (e.g. parsed from stderr from command line
            tools).
        :rtype: dict
        """
        # It is quite conceivable that no errors occur during processing, thus no check of any kind is applied
        #  to make sure that _process_errors actually has entries
        return self._process_errors

    @process_errors.setter
    def process_errors(self, process_name_error_dict: Tuple[str, dict]):
        """
        Property setter for a nested dictionary containing error information from processing applied to mission
        data. This shouldn't be used directly by a user, rather DAXA processing functions will use it themselves. This
        setter does not overwrite the existing dictionary, but rather adds extra information.

        :param Tuple[str, dict] process_name_error_dict: A tuple with the first element being the name of the
            process for which a error dictionary is being passed, and the second being the error dictionary
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        """
        # This applies checks to the input to this setter
        pr_name, error_info = self._check_process_inputs(process_name_error_dict)

        # Iterate through the missions in the input dictionary
        for mn in error_info:
            # If the particular process does not have an entry for the particular mission then we add it to the
            #  dictionary, but if it does then we warn the user and do nothing
            if pr_name in self._process_errors[mn]:
                warn("The process_errors property already has an entry for {prn} under {mn}, no change will be "
                     "made.".format(prn=pr_name, mn=mn), stacklevel=2)
            else:
                self._process_errors[mn][pr_name] = error_info[mn]

    @property
    def process_warnings(self) -> dict:
        """
        Property getter for a nested dictionary containing warning information from processing applied to mission data.

        :return: A nested dictionary where top level keys are mission names, next level keys are processing
            function names, and lowest level keys are either ObsID or ObsID+instrument names. The values
            attributed with the lowest level keys are warning outputs (e.g. parsed from stderr from command line
            tools).
        :rtype: dict
        """
        # It is quite conceivable that no warnings occur during processing, thus no check of any kind is applied
        #  to make sure that _process_warnings actually has entries
        return self._process_warnings

    @process_warnings.setter
    def process_warnings(self, process_name_warn_dict: Tuple[str, dict]):
        """
        Property setter for a nested dictionary containing warning information from processing applied to mission
        data. This shouldn't be used directly by a user, rather DAXA processing functions will use it themselves. This
        setter does not overwrite the existing dictionary, but rather adds extra information.

        :param Tuple[str, dict] process_name_warn_dict: A tuple with the first element being the name of the
            process for which a warning dictionary is being passed, and the second being the warning dictionary
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        """
        # This applies checks to the input to this setter
        pr_name, warn_info = self._check_process_inputs(process_name_warn_dict)

        # Iterate through the missions in the input dictionary
        for mn in warn_info:
            # If the particular process does not have an entry for the particular mission then we add it to the
            #  dictionary, but if it does then we warn the user and do nothing
            if pr_name in self._process_warnings[mn]:
                warn("The process_warnings property already has an entry for {prn} under {mn}, no change will be "
                     "made.".format(prn=pr_name, mn=mn), stacklevel=2)
            else:
                self._process_warnings[mn][pr_name] = warn_info[mn]

    @property
    def raw_process_errors(self) -> dict:
        """
        Property getter for a nested dictionary containing unparsed error information (e.g. the entire stderr
        output from an XMM SAS process) from processing applied to mission data.

        :return: A nested dictionary where top level keys are mission names, next level keys are processing
            function names, and lowest level keys are either ObsID or ObsID+instrument names. The values
            attributed with the lowest level keys are error outputs (e.g. stderr from command line
            tools).
        :rtype: dict
        """
        # It is quite conceivable that no errors occur during processing, thus no check of any kind is applied
        #  to make sure that _process_errors actually has entries
        return self._process_raw_errors

    @raw_process_errors.setter
    def raw_process_errors(self, process_name_error_dict: Tuple[str, dict]):
        """
        Property setter for a nested dictionary containing unparsed error information from processing applied to
        mission data. This shouldn't be used directly by a user, rather DAXA processing functions will use it
        themselves. This setter does not overwrite the existing dictionary, but rather adds extra information.

        :param Tuple[str, dict] process_name_error_dict: A tuple with the first element being the name of the
            process for which an error dictionary is being passed, and the second being the error dictionary
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        """
        # This applies checks to the input to this setter
        pr_name, error_info = self._check_process_inputs(process_name_error_dict)

        # Iterate through the missions in the input dictionary
        for mn in error_info:
            # If the particular process does not have an entry for the particular mission then we add it to the
            #  dictionary, but if it does then we warn the user and do nothing
            if pr_name in self._process_raw_errors[mn]:
                warn("The raw_process_errors property already has an entry for {prn} under {mn}, no change will be "
                     "made.".format(prn=pr_name, mn=mn), stacklevel=2)
            else:
                self._process_raw_errors[mn][pr_name] = error_info[mn]

    @property
    def process_logs(self) -> dict:
        """
        Property getter for a nested dictionary containing log information from processing applied to mission data.

        :return: A nested dictionary where top level keys are mission names, next level keys are processing
            function names, and lowest level keys are either ObsID or ObsID+instrument names. The values
            attributed with the lowest level keys are logs (e.g. stdout from command line tools).
        :rtype: dict
        """
        # Check to make sure that logging information for at least one processing function on at least one mission
        #  has been added to this archive, otherwise an error is thrown.
        if sum([len(self._process_logs[mn]) for mn in self.mission_names]) == 0:
            raise NoProcessingError("No processing log information has been added to this archive, meaning "
                                    "that no data processing has been applied.")

        return self._process_logs

    @process_logs.setter
    def process_logs(self, process_name_log_dict: Tuple[str, dict]):
        """
        Property setter for a nested dictionary containing log information from processing applied to mission
        data. This shouldn't be used directly by a user, rather DAXA processing functions will use it themselves. This
        setter does not overwrite the existing dictionary, but rather adds extra information.

        :param Tuple[str, dict] process_name_log_dict: A tuple with the first element being the name of the
            process for which a success dictionary is being passed, and the second being the log dictionary
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        """
        # This applies checks to the input to this setter
        pr_name, log_info = self._check_process_inputs(process_name_log_dict)

        # Iterate through the missions in the input dictionary
        for mn in log_info:
            # If the particular process does not have an entry for the particular mission then we add it to the
            #  dictionary, but if it does then we warn the user and do nothing
            if pr_name in self._process_logs[mn]:
                warn("The process_logs property already has an entry for {prn} under {mn}, no change will be "
                     "made.".format(prn=pr_name, mn=mn), stacklevel=2)
            else:
                self._process_logs[mn][pr_name] = log_info[mn]

    @property
    def process_extra_info(self) -> dict:
        """
        Property getter for a nested dictionary containing extra information from processing applied to mission data.
        This can be things like paths to event lists, or configuration information. It is unlikely to be necessary
        for users to directly access this property.

        :return: A nested dictionary where top level keys are mission names, next level keys are processing
            function names, and lowest level keys are either ObsID or ObsID+instrument names. The values
            attributed with the lowest level keys are dictionaries of extra information (e.g. config info).
        :rtype: dict
        """
        # It is quite conceivable that no extra information has been recorded from processing, thus no check of any
        #  kind is applied to make sure that _process_extra_info actually has entries
        return self._process_extra_info

    @process_extra_info.setter
    def process_extra_info(self, process_name_info_dict: Tuple[str, dict]):
        """
        Property setter for a nested dictionary containing extra information from processing applied to mission data.
         This shouldn't be used directly by a user, rather DAXA processing functions will use it themselves. This
        setter does not overwrite the existing dictionary, but rather adds extra information.

        :param Tuple[str, dict] process_name_info_dict: A tuple with the first element being the name of the
            process for which a success dictionary is being passed, and the second being the log dictionary
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        """
        # This applies checks to the input to this setter
        pr_name, einfo_info = self._check_process_inputs(process_name_info_dict)

        # Iterate through the missions in the input dictionary
        for mn in einfo_info:
            # If the particular process does not have an entry for the particular mission then we add it to the
            #  dictionary, but if it does then we warn the user and do nothing
            if pr_name in self._process_extra_info[mn]:
                warn("The process_extra_info property already has an entry for {prn} under {mn}, no change will be "
                     "made.".format(prn=pr_name, mn=mn), stacklevel=2)
            else:
                self._process_extra_info[mn][pr_name] = einfo_info[mn]

    @property
    def observation_summaries(self) -> dict:
        """
        This property returns information on the different observations available to each mission. This information
        will vary from mission to mission, and is primarily intended for use by DAXA processing methods, but could
        include things such as whether an instrument was active for a particular observation, what sub-exposures
        there were (relevant for XMM for instance), what filter was active, etc.

        :return: A dictionary of information with missions as the top level keys, then ObsIDs, then instruments.
            Keys on levels below that will be determined by the information available for specific instruments of
            specific missions.
        :rtype: bool
        """
        return self._miss_obs_summ_info

    @observation_summaries.setter
    def observation_summaries(self, new_val: dict):
        """
        This is the property setter for information on observations available for each mission. This information
        will vary from mission to mission, and is primarily intended for use by DAXA processing methods, but could
        include things such as whether an instrument was active for a particular observation, what sub-exposures
        there were (relevant for XMM for instance), what filter was active, etc.

        The observation_summaries property shouldn't need to be set by the user, as ideally DAXA processes will
        acquire and provide that information.

        Using this setter will also trigger a process where each observation added will be assessed to determine
        whether it should be processed for scientific purposes. The factors that go into this decision will depend
        heavily on the telescope, but to give an example; an XMM observation that has been acquired would not
        be processed if the filter were found to be CalClosed - there are no data on astrophysical objects there.

        :param dict new_val: A dictionary of information, with mission as the top level key, next level down
            the ObsID, next level down the instruments. Beyond that the information can vary on a mission by
            mission basis.
        """
        for mn in new_val:
            for o_id in new_val[mn]:
                # If the particular observation does not have an entry for the particular mission then we add it to the
                #  dictionary, but if it does then we warn the user and do nothing
                if o_id in self._miss_obs_summ_info[mn]:
                    warn("The observation_summaries property already has an entry for {o_id} under {mn}, no change "
                         "will be made.".format(o_id=o_id, mn=mn), stacklevel=2)
                else:
                    # This just removes any entries that might exist for instruments that haven't been selected for
                    #  use by the mission class. This was driven by the fact that parsing XMM SAS summary files will
                    #  still include entries for instruments that don't have files (even if they're null).
                    rel_dat = {inst: info for inst, info in new_val[mn][o_id].items()
                               if inst in self._missions[mn].chosen_instruments}
                    # This part allows us to put a hard requirement on having certain entries at the instrument
                    #  level - initially I'm only including 'active' in that, but perhaps I shall add more
                    if not all(['active' in info for info in rel_dat.values()]):
                        raise KeyError("Observation information instrument level dictionaries must contain an "
                                       "'active' entry, a boolean value to determine whether they were turned on "
                                       "or not.")
                    # Make sure to actually store the observation info in our storage attribute
                    self._miss_obs_summ_info[mn][o_id] = rel_dat

                    # Now we can use the mission-specific observation-assessor to determine whether these particular
                    #  ObsID-Instrument(-subexposure) combinations should be processed for scientific use. The
                    #  assessor is mission specific because the decision criteria will vary
                    # That information is stored in another attribute, to be accessed by processing functions through
                    #  an archive property
                    self._use_this_obs[mn][o_id] = self._missions[mn].assess_process_obs(rel_dat)

    @property
    def process_observation(self) -> dict:
        """
        This property returns the dictionary of mission-ObsID-Instrument(-subexposure) boolean flags that indicate
        whether the data for that observation-instrument(-subexposure) should be processed for science. There is a
        companion get method that returns only the data identifiers that should be processed.

        :return: The dictionary containing information on whether particular data should be processed.
        :rtype: dict
        """
        return self._use_this_obs

    @property
    def final_process_success(self) -> dict:
        """
        This property returns the dictionary which stores the final judgement (at the ObsID level) of whether there
        are any useful data (True) or whether no aspect of that observation reached the end of the final processing
        step successfully. The ObsIDs marked as False will be moved from the archive processed data directory to a
        separate failed data directory.

        The flags are only added once the final processing step for a particular mission has been run.

        :return: The dictionary of final processing success flags.
        :rtype: dict
        """
        return self._final_obs_id_success

    @final_process_success.setter
    def final_process_success(self, new_val: dict):
        """
        Setter for the final_process_success property, where a final judgement on if an ObsID has any useful
        data in it once processed can be passed. This setter should not be called manually by the user.

        :param dict new_val: A dictionary of information, with missions as the top level keys, and ObsIDs as the
            bottom level keys. The values attribute to ObsID keys should be True or False.
        """
        for mn in new_val:
            if not all([isinstance(new_val[mn][o_id], bool) for o_id in new_val[mn]]):
                # Just check that all the value entries are boolean.
                raise TypeError("All values associated with ObsID keys passed to final_process_success "
                                "must be boolean.")
            # Now iterating through the ObsIDs to add the new information into the storage attribute
            for o_id in new_val[mn]:
                # If the particular observation does not have an entry for the particular mission then we add it to the
                #  dictionary, but if it does then we warn the user and do nothing
                if o_id in self._final_obs_id_success[mn]:
                    warn("The final_process_success property already has an entry for {o_id} under {mn}, no change "
                         "will be made.".format(o_id=o_id, mn=mn), stacklevel=2)
                else:
                    # Adding in the success flags
                    self._final_obs_id_success[mn][o_id] = new_val[mn][o_id]

    @property
    def source_regions(self) -> dict:
        """
        This property returns all source regions which have been associated with missions in this archive. The top
        level keys of the dictionary are mission names, the bottom level keys are observation identifiers, and the
        values are lists of region objects.

        If an observation in this archive has had regions added for it, then those regions will also have been
        written to permanent storage in the archive directory structure. The path can be identified using the
        get_region_file_path method of this archive.

        :return: Dictionary containing regions on a mission-observation basis.
        :rtype: dict
        """
        return self._source_regions

    @source_regions.setter
    def source_regions(self, new_val: dict):
        """
        The setter method for the source regions property. This takes source region information for observations of
        missions associated with this archive, as well as (if necessary) the WCS information required to convert
        the regions from pixel coordinates to RA-Dec coordinates.

        The input dictionary should be formatted in one of the following ways:

        {'mission_name': {'ObsID': 'path to regions'}}

        OR

        {'mission_name': {'ObsID': [list of region objects]}}

        OR

        {'mission_name': {'ObsID': {'region': 'path to regions'}}}

        OR

        {'mission_name': {'ObsID': {'region': [list of region objects]}}}

        OR

        {'mission_name': {'ObsID': {'region': ..., 'wcs_src': 'path to image'}}}

        OR

        {'mission_name': {'ObsID': {'region': ..., 'wcs_src': XGA Image}}}

        OR

        {'mission_name': {'ObsID': {'region': ..., 'wcs_src': Astropy WCS object}}}

        :param dict new_val: A dictionary containing new region information, with top level keys being mission names,
            the next level down's keys being ObsIDs, and the values being either a list of regions, a string path
            to a region file, or a dictionary containing (at least) a 'region' key whose value is either a path to
            a region file or a list of regions. If a dictionary is the value you may also supply WCS information
            with the key 'wcs_src'. This information may either be an astropy WCS object, an XGA image, or a path
            to an image.
        """
        # Going to kick up a fuss if the top level keys aren't the format we expect; i.e. they are mission names
        #  actually associated with this archive
        if any([key not in self.mission_names for key in new_val]):
            raise KeyError("The top level keys of the dictionary passed to the source_regions property setter must "
                           "be mission names associated with this archive; i.e. "
                           "{}".format('. '.join(self.mission_names)))

        # Now we start iterating through the new values, starting at the top level with mission keys
        for mn in new_val:
            for o_id in new_val[mn]:
                cur_val = new_val[mn][o_id]
                # If the particular observation does not have an entry for the particular mission then we add it to the
                #  dictionary, but if it does then we warn the user
                if o_id in self._source_regions[mn]:
                    warn("The source_regions property already had an entry for {o_id} under {mn}, this has been "
                         "overwritten!".format(o_id=o_id, mn=mn), stacklevel=2)
                # The observation level keys should just be ObsIDs (without any sub-exposure or instrument
                #  identifiers), so I can check to see whether they are actually associated with the mission. The
                #  extra checks here are also a good idea because it is conceivable that the user will be setting
                #  this property themselves.
                elif o_id not in self._missions[mn].filtered_obs_ids:
                    raise ObsNotAssociatedError("The ObsID {oid} is not associated with the filtered dataset of "
                                                "{mn}.".format(mn=mn, oid=o_id))

                # If a dictionary is the value then regions must be one entry, which should either be a path to a
                #  regions file or a list of regions
                if isinstance(cur_val, dict) and 'regions' not in cur_val:
                    raise KeyError("The new regions entry for {mn}-{o} is a dictionary but does not contain a "
                                   "'regions' key - this is mandatory.".format(mn=mn, o=o_id))
                # Really if the value is a dictionary it should be because a source of WCS information required to
                #  convert the regions to RA-Dec has also been provided, but I won't make that mandatory.
                elif isinstance(cur_val, dict) and 'wcs_src' not in cur_val:
                    cur_reg = cur_val['regions']
                    cur_wcs_src = None
                # In case the regions AND a source of WCS information has been passed - I will let that WCS source be
                #  a couple of things, either a path to an image, a WCS object, or an XGA image that I can grab the
                #  WCS from.
                elif isinstance(cur_val, dict) and 'wcs_src' in cur_val:
                    cur_reg = cur_val['regions']
                    cur_wcs_src = cur_val['wcs_src']
                # If the value isn't a dictionary then it's just the regions, either in path or list form
                else:
                    cur_reg = cur_val
                    cur_wcs_src = None

                # Now we've got set values for the current regions and the available WCS information, we need to check
                #  to see what form they're in - damn me for making this so general. The goal for this is for cur_reg
                #  to be (or become) a list of region objects.
                if isinstance(cur_reg, list) and any(not isinstance(cr, Region) for cr in cur_reg):
                    raise TypeError("If a list of regions is passed, all elements must be a region instance.")
                # If the regions were passed as a string, we use that as a file path, but obviously have to check
                #  that it exists first
                elif isinstance(cur_reg, str) and not os.path.exists(cur_reg):
                    raise FileNotFoundError("The region file ({cr}) for {mn}-{oi} does not "
                                            "exist.".format(mn=mn, oi=o_id, cr=cur_reg))
                # If the region is a string and we've got here, then that file must exist so we use the regions
                #  module to read it in (assuming it is in a DS9 format).
                elif isinstance(cur_reg, str):
                    cur_reg = read_ds9(cur_reg)
                # If none of the above were triggered then something weird has been passed and we throw a (hopefully)
                #  useful diagnostic error
                elif not isinstance(cur_reg, list):
                    raise TypeError("Illegal new regions value ({cr}) for {mn}-{oi}; it must either be a list of "
                                    "region objects or a string path to a region file.".format(mn=mn, oi=o_id,
                                                                                               cr=cur_reg))

                # Now have to do the same sort of normalisation to whatever was passed for WCS info (if indeed
                #  anything was passed at all!) We'll check whether the passed regions actually need WCS information
                #  later (they only need them if they're in pixel coordinates, as we wish to convert them).
                if isinstance(cur_wcs_src, wcs.WCS):
                    cur_wcs = cur_wcs_src
                elif isinstance(cur_wcs_src, str) and not os.path.exists(cur_wcs_src):
                    raise FileNotFoundError("The image file path ({cw}) passed to provide WCS info for {mn}-{oi} does "
                                            "not exist!".format(cw=cur_wcs_src, mn=mn, oi=o_id))
                # In the case that the input is a path to an image, and that file exists, we read it in with XGA
                elif isinstance(cur_wcs_src, str):
                    # This local import makes me sad, but currently released XGA has an issue where you need to
                    #  fill out the configuration file even to use the products, and I want to avoid that error unless
                    #  its absolutely necessary - this won't be necessary once I update XGA to not require a config
                    #  file unless sources are being declared
                    from xga.products import Image
                    # Very cheesy image object declaration, only doing this to easily retrieve the WCS
                    #  information, energy inputs don't matter and are complete nonsense here
                    im = Image(cur_wcs_src, '', '', '', '', '', Quantity(0.01, 'keV'), Quantity(0.02, 'keV'))
                    cur_wcs = im.radec_wcs
                    del im
                # This isn't the right way to do this (should be using isinstance as above), but as I said I currently
                #  wish to avoid importing XGA unless completely necessary and to use isinstance here I'd have to
                #  import XGA at the top of this file, or in this property setter.
                elif str(type(cur_wcs_src)) == "<class 'xga.products.phot.Image'>":
                    cur_wcs = cur_wcs_src.radec_wcs
                else:
                    cur_wcs = None

                # At this point we've either raised an exception, or we have a list of region objects! We also have a
                #  WCS object, or a None value. Now we have to see whether they are pixel regions or sky regions (we
                #  want to end up with sky regions, as they are independent of a particular WCS)
                if any([isinstance(cr, PixelRegion) for cr in cur_reg]) and cur_wcs is None:
                    raise ValueError("{mn}-{oi} regions are in pixel coordinates and have no accompanying WCS "
                                     "information; the WCS can be passed as an astropy WCS object, and XGA image, or "
                                     "a path to a fits image.".format(mn=mn, oi=o_id))
                elif any([isinstance(cr, PixelRegion) for cr in cur_reg]):
                    # This accounts for the possibility that some psychopath has a list of regions that have
                    #  entries both in pixel and sky coordinates (yaaay go ternary operators).
                    fin_reg = [cr.to_sky(cur_wcs) if isinstance(cr, PixelRegion) else cr for cr in cur_reg]
                else:
                    fin_reg = cur_reg

                # And finally we store our final set of sky regions in the region attribute!
                self._source_regions[mn][o_id] = fin_reg

                # Finally finally, we write these regions to a directory for safe-keeping - firstly have to make sure
                #  that the directory we wish to store in has been created
                stor_dir = self.archive_path + 'regions/{mn}/{oi}/'.format(mn=mn, oi=o_id)
                if not os.path.exists(stor_dir):
                    os.makedirs(stor_dir)
                # This will overwrite an existing file so no need to delete one that might already be there if the
                #  ObsID has already had regions added to it
                write_ds9(fin_reg, stor_dir + 'source_regions_radec.reg')

    # Then define internal methods
    def _check_process_inputs(self, process_vals: Tuple[str, dict]) -> Tuple[str, dict]:
        """
        An internal function to check the format of inputs to setters of process information, such as process_success,
        process_errors, and process_logs.

        :param process_vals: A tuple with the first element being the name of the process for which an information
            dictionary is being passed, and the second being a dictionary of process related information
            with top level keys being mission names, and bottom level keys being ObsID or ObsID+instrument keys.
        :return: The name and information dictionary.
        :rtype: bool
        """
        if not isinstance(process_vals, tuple):
            raise TypeError("The value passed to this setter must be a tuple with two elements, the first being "
                            "the name of the process and the second the dictionary of process information.")
        # Ensure that the correct length of Tuple has been passed
        elif len(process_vals) != 2:
            raise ValueError("The value passed to this setter must be a tuple with two elements, the first being "
                             "the name of the process and the second the dictionary of process information.")
        else:
            # If it has we unpack the tuple into two variables for clarity
            pr_name, process_info = process_vals

        # Make sure that element one is the correct type
        if not isinstance(pr_name, str):
            raise TypeError("The first element of the value passed to the setter must be the "
                            "string name of a DAXA processing function.")
        # Do the same with element two
        elif not isinstance(process_info, dict):
            raise TypeError("The second element of the value passed to the setter must be a dictionary.")

        # Have to check that the top level keys are all mission names associated with this archive
        top_key_check = [top_key for top_key in process_info if top_key not in self.mission_names]

        # If not then we throw a hopefully quite informative error
        if len(top_key_check) != 0:
            raise KeyError("One or more top-level keys ({bk}) in the process information dictionary do not "
                           "correspond to missions associated with this archive; {mn} are "
                           "allowed.".format(bk=','.join(top_key_check), mn=','.join(self.mission_names)))

        return pr_name, process_info

    def _data_path_construct_checks(self, mission: Union[BaseMission, str] = None, obs_id: str = None) -> str:
        """
        These check inputs to the methods which will construct paths to the process/failed data processed
        from this archive.

        :param BaseMission/str mission: The mission for which to retrieve the processed data path. Default is None
            in which case a path ready to be formatted with a mission name will be provided.
        :param str obs_id: The ObsID for which to retrieve the processed data path, cannot be set if 'mission' is
            set to None. Default is None, in which case a path ready to be formatted with an observation ID will
            be provided.
        :return: The mission name.
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
        if m_name is not None and m_name not in self.mission_names:
            raise ValueError("The mission {m} is not a part of this Archive; the current missions are "
                             "{cm}".format(m=m_name, cm=', '.join(self.mission_names)))

        # Need to make sure that the passed ObsID (if one was passed) is in the correct format for the
        #  specified mission - can use this handy method for that.
        if obs_id is not None and not self[m_name].check_obsid_pattern(obs_id):
            raise ValueError("The passed ObsID ({oi}) does not match the pattern expected for {mn} "
                             "identifiers.".format(mn=m_name, oi=obs_id))

        return m_name

    # Then define user-facing methods
    def get_processed_data_path(self, mission: Union[BaseMission, str] = None, obs_id: str = None):
        """
        This method is to construct paths to directories where processed data for a particular mission + observation
        ID combination will be stored. That functionality is added here so that any change to how those directories
        are named will take place in only one part of DAXA, and will propagate to other parts of the module. It is
        unlikely that a user will need to directly use this method.

        If no mission is passed, then no observation ID may be passed. In the case of 'mission' and 'obs_id' being
        None, the returned string will be constructed ready to format; {mn} should be replaced by the DAXA mission
        name, and {oi} by the relevant ObsID.

        Retrieving a data path from this method DOES NOT guarantee that it has been created.

        :param BaseMission/str mission: The mission for which to retrieve the processed data path. Default is None
            in which case a path ready to be formatted with a mission name will be provided.
        :param str obs_id: The ObsID for which to retrieve the processed data path, cannot be set if 'mission' is
            set to None. Default is None, in which case a path ready to be formatted with an observation ID will
            be provided.
        :return: The requested path.
        :rtype: str
        """
        # This runs through a set of checks on the inputs to this method - those checks are in another method
        #  because they are also used by get_failed_data_path
        m_name = self._data_path_construct_checks(mission, obs_id)

        # Now we just run through the different possible combinations of circumstances.
        base_path = self.archive_path+'processed_data/{mn}/{oi}/'
        if m_name is not None and obs_id is not None:
            ret_str = base_path.format(mn=m_name, oi=obs_id)
        elif m_name is not None and obs_id is None:
            ret_str = base_path.format(mn=m_name, oi='{oi}')
        else:
            ret_str = base_path

        return ret_str

    def get_failed_data_path(self, mission: Union[BaseMission, str] = None, obs_id: str = None):
        """
        This method is to construct paths to directories where data for a particular mission + observation
        ID combination which failed to process will be stored. That functionality is added here so that any change
        to how those directories are named will take place in only one part of DAXA, and will propagate to other
        parts of the module. It is unlikely that a user will need to directly use this method.

        If no mission is passed, then no observation ID may be passed. In the case of 'mission' and 'obs_id' being
        None, the returned string will be constructed ready to format; {mn} should be replaced by the DAXA mission
        name, and {oi} by the relevant ObsID.

        Retrieving a data path from this method DOES NOT guarantee that it has been created.

        :param BaseMission/str mission: The mission for which to retrieve the failed data path. Default is None
            in which case a path ready to be formatted with a mission name will be provided.
        :param str obs_id: The ObsID for which to retrieve the failed data path, cannot be set if 'mission' is
            set to None. Default is None, in which case a path ready to be formatted with an observation ID will
            be provided.
        :return: The requested path.
        :rtype: str
        """

        # This runs through a set of checks on the inputs to this method - those checks are in another method
        #  because they are also used by get_processed_data_path
        m_name = self._data_path_construct_checks(mission, obs_id)

        # The mission name might be None here, in which case using m_name as a key would break things!
        if m_name is not None:
            # Want to know which ObsIDs have been marked as overall failures
            failed_obsids = [obs_id for obs_id, success in self.final_process_success[m_name].items() if not success]
        else:
            failed_obsids = None

        # Now we just run through the different possible combinations of circumstances.
        base_path = self.archive_path+'failed_data/{mn}/{oi}/'
        if m_name is not None and obs_id is not None and obs_id in failed_obsids:
            ret_str = base_path.format(mn=m_name, oi=obs_id)
        elif m_name is not None and obs_id is not None and obs_id not in failed_obsids:
            raise ValueError("The observation ({oid}) has not been marked as an overall failure, no "
                             "path can be retrieved".format(oid=obs_id))
        elif m_name is not None and obs_id is None:
            ret_str = base_path.format(mn=m_name, oi='{oi}')
        else:
            ret_str = base_path

        return ret_str

    def get_region_file_path(self, mission: Union[BaseMission, str] = None, obs_id: str = None) -> str:
        """
        This method is to construct paths to files where the regions associated with a particular observation of a
        particular mission are stored after being added to the archive. If a mission and ObsID are specified then
        this method will check whether region information for that particular ObsID of that particular mission
        exists in this archive, and raise an error if it does not.

        If no mission is passed, then no observation ID may be passed. In the case of 'mission' and 'obs_id' being
        None, the returned string will be constructed ready to format; {mn} should be replaced by the DAXA mission
        name, and {oi} by the relevant ObsID.

        Retrieving a region file path from this method without passing mission and ObsID DOES NOT guarantee that one
        has been created for whatever mission and ObsID are added to the string later.

        :param BaseMission/str mission: The mission for which to retrieve the region file path. Default is None
            in which case a path ready to be formatted with a mission name will be provided.
        :param str obs_id: The ObsID for which to retrieve the region file path, cannot be set if 'mission' is
            set to None. Default is None, in which case a path ready to be formatted with an observation ID will
            be provided.
        :return: The requested path.
        :rtype: str
        """
        # This runs through a set of checks on the inputs to this method - those checks are in another method
        #  because they are also used by get_failed_data_path and get_processed_data_path
        m_name = self._data_path_construct_checks(mission, obs_id)

        # This is the file path of where region files are stored when information is added to a DAXA archive, but
        #  it hasn't been filled in with mission and ObsID yet.
        base_path = self.archive_path + 'regions/{mn}/{oi}/source_regions_radec.reg'
        # Now we just run through the different possible combinations of circumstances.
        if m_name is not None and obs_id is not None and obs_id not in self.source_regions[m_name]:
            raise ValueError("The observation ({oid}) has no region information in this "
                             "archive.".format(oid=obs_id))
        elif m_name is not None and obs_id is not None:
            ret_str = base_path.format(mn=m_name, oi=obs_id)
        elif m_name is not None and obs_id is None:
            ret_str = base_path.format(mn=m_name, oi='{oi}')
        else:
            ret_str = base_path

        return ret_str

    def get_obs_to_process(self, mission_name: str, search_ident: str = None) -> List[List[str]]:
        """
        This method will provide a list of lists of [ObsID, Instrument, SubExposure (depending on mission)] that
        should be processed for scientific use for a specific mission. The idea is that this method can be
        called, and just by iterating through the result you will get the identifiers of all valid data that
        match your input.

        It shouldn't really need to be used directly by users, but instead will be very useful for the processing
        functions - it will tell them which data need to be processed.

        :param str mission_name: The internal DAXA name of the mission to retrieve information for.
        :param str search_ident: Either an ObsID or an instrument name to retrieve matching information for. An ObsID
            will search through all the instruments/subexposures, an instrument will search all ObsIDs and
            sub-exposures. The default is None, in which case all ObsIDs, instruments, and sub-exposures will
            be searched.
        :return: List of lists of [ObsID, Instrument, SubExposure (depending on mission)].
        :rtype: List[List]
        """
        # Check to make sure that the mission name is valid for this archive
        if mission_name not in self.mission_names:
            raise ValueError("The mission {mn} is not associated with this archive. Available missions are "
                             "{am}".format(mn=mission_name, am=', '.join(self.mission_names)))

        # This is the final output, and will be a list of lists of [ObsID, Instrument, SUBEXP DEPENDING ON MISSION]
        all_res = []

        # If the search identifier is none it behaves slightly differently
        if search_ident is None:
            # No specific search ident has been passed (i.e. an instrument or an ObsID), but I still want the
            #  dictionary output to be in the same format as if there were, for re-formatting later. As such I
            #  just search for our mission name.
            search_ident = mission_name
            rel_use_obs = self._use_this_obs
        else:
            # In the case where a search ident has been provided, I only search the information for my
            # specific mission
            rel_use_obs = self._use_this_obs[mission_name]

            # I try to check that the search_ident value is legal - either an instrument for the selected mission, or
            #  an ObsID
            if search_ident not in self._missions[mission_name].filtered_obs_ids \
                    and search_ident not in self._missions[mission_name].chosen_instruments:
                raise ValueError("The passed search ident ({si}) should be either an ObsID associated with the "
                                 "mission, or a mission instrument enabled for this archive; otherwise it must be "
                                 "set to None.".format(si=search_ident))

        # Using dict_search and iterating (as it as a generator) - I know this method isn't super well explained, but
        #  in theory it should work for _use_this_obs which are upto four levels deep (mission-obsid-inst-subexp), and
        #  support the other possible depth where there aren't subexposures. Its disgusting and janky but I can't be
        #  bothered to improve it
        # The goal of these different processing
        #  steps is to construct a list of lists of [ObsID, Instrument, SubExp] that CAN be used, and the return
        #  from this function can be iterated through.
        for res in dict_search(search_ident, rel_use_obs):
            # This will catch when the dictionary is three deep (ObsID-Instrument-SubExp) and the search_ident was
            #  on the second level
            if isinstance(res, list) and isinstance(res[1], dict):
                proc_res = [[res[0], search_ident, sp_id] for ll in res[1:] for sp_id, to_use in ll.items()
                            if to_use]

            # This will catch when the dictionary is two deep (ObsID-Instrument) and the search_ident was on the
            #  second level
            elif isinstance(res, list):
                proc_res = [[res[0], search_ident] for to_use in res[1:] if to_use]

            # This case is generally when search_ident is None and the storage structure is mission-ObsID-Inst-SubExp
            elif isinstance(res, dict) and isinstance(list(res.values())[0], dict)\
                    and isinstance(list(list(res.values())[0].values())[0], dict):
                proc_res = [[tl_key, ll_key, sp_id] for tl_key, tl_val in res.items() for ll_key, ll_val in
                            tl_val.items()
                            for sp_id, to_use in ll_val.items() if to_use]

            #  This is when the dictionary is two deep and the search_ident is the top level
            elif isinstance(res, dict) and isinstance(list(res.values())[0], dict):
                proc_res = [[tl_key, ll_key] for tl_key, tl_val in res.items() for ll_key, to_use in
                            tl_val.items() if to_use]

            # This will catch when the dictionary is three deep (ObsID-Instrument-SubExp) and the search_ident is
            #  on the top level
            elif isinstance(res, dict):
                proc_res = [[search_ident, ll_key, sp_id] for ll_key, ll_val in res.items()
                            for sp_id, to_use in ll_val.items() if to_use]

            # Add the current processed result into the overall results
            all_res += proc_res

        return all_res

    def check_dependence_success(self, mission_name: str, obs_ident: Union[str, List[str], List[List[str]]],
                                 dep_proc: Union[str, List[str]], no_success_error: bool = True) -> np.ndarray:
        """
        This method should be used by processing functions, rather than the user, to determine whether previous
        processing steps (specified in the input to this function) ran successfully for the specified data.

        Each processing function should be setup to call this method with appropriate previous steps and
        identifiers, and will know from its boolean array return which data can be processed safely. If no data
        has successfully run through a previous step, or no attempt to run a previous step occurred, then an
        error will be thrown.

        :param str mission_name: The name of the mission for which we wish to check the success of
            previous processing steps.
        :param str/List[str], List[List[str]] obs_ident: A set (or individual) set of observation identifiers. This
            should be in the style output by get_obs_to_process (i.e. [ObsID, Inst, SubExp (depending on mission)],
            though does also support just an ObsID.
        :param str/List[str] dep_proc: The name(s) of the process(es) that have to have been run for further
            processing steps to be successful.
        :param bool no_success_error: If none of the specified previous processing steps have been run
            successfully, should a NoDependencyProcessError be raised. Default is True, but if set to False the
            error will not be raised and the return will be an all-False array. This will NOT override the
            error raised if a previous process hasn't been run at all.
        :return: A boolean array that defines whether the process(es) specified in the input were successful. Each
            set of identifying information provided in obs_ident has a corresponding entry in the return.
        :rtype: np.ndarray
        """
        # Check to make sure that the mission name is valid for this archive
        if mission_name not in self.mission_names:
            raise ValueError("The mission {mn} is not associated with this archive. Available missions are "
                             "{am}".format(mn=mission_name, am=', '.join(self.mission_names)))

        # This doesn't often happen when dealing with many observations assigned to a mission, but I did notice it
        #  happen - this should never be triggered by DAXA functions as I've put checks to ensure that zero length
        #  lists are never passed
        if isinstance(obs_ident, list) and len(obs_ident) == 0:
            raise ValueError("If 'obs_ident' is a list, it cannot be zero-length.")

        # Normalising the input so that dep_proc can always be iterated through. I imagine most of the time this
        #  method is used it will be for individual process checking, but you never know.
        if isinstance(dep_proc, str):
            dep_proc = [dep_proc]

        # I also want to normalise the obs_ident input as either a single set of identifying information, or
        #  multiple sets, can be passed to this method. Thus everything must become a list of lists
        if isinstance(obs_ident, list) and len(obs_ident) != 0 and isinstance(obs_ident[0], str):
            obs_ident = [obs_ident]
        # If just a string is passed I will assume it is the overall ObsID and double wrap it in a list, one because
        #  identifiers are expected to be in lists of [ObsID, Inst, SubExp (depending on mission)], and the second
        #  to make it overall a list of lists
        elif isinstance(obs_ident, str):
            obs_ident = [[obs_ident]]
        # Setting up an empty array of Trues, one for each observation identifier. This way we can just multiply
        #  the boolean success value for each process for each obs by what's in the array - this works for
        #  multiple processes
        run_success = np.full(len(obs_ident), True)

        # Cycling through the processes we have to check ran and succeeded
        for dp in dep_proc:
            # If there has been no processing done, then when we try to access process_logs it will throw an error -
            #  otherwise if a process has been run for a particular observation then that will be in the process logs.
            try:
                # If there is no entry for the process we're checking, then we know it hasn't run and we throw an
                #  appropriately useful error
                if dp not in self.process_logs[mission_name]:
                    raise NoDependencyProcessError("The required process {p} has not been run for "
                                                   "{mn}.".format(p=dp, mn=mission_name))

                # If we got this far then the required task has been run, so now we see whether it succeeded
                for ident_ind, ident in enumerate(obs_ident):
                    # As process functions will call this method rather than the user, I'm assuming that the
                    #  list of identifying information is everything required to specify entry in the 'success' logs
                    #  I'm looking for. As such they just get combined
                    comb_id = ''.join(ident)
                    # Then I look in the process success logs, and multiply that bool by the (originally True) entry
                    #  in run_success. If my retrieved success flag is True nothing will happen, if its False then
                    #  the run_success entry will be set to False
                    if comb_id in self.process_success[mission_name][dp]:
                        run_success[ident_ind] *= self.process_success[mission_name][dp][comb_id]
                    # If comb_id isn't in that dictionary then the process wasn't even run for that comb_id, likely
                    #  meaning a previous step failed, so we just set to False
                    else:
                        run_success[ident_ind] = False

            # If we get the error thrown by process logs because nothing at all has been processed, we catch that
            #  and turn it into an error more useful for this specific method
            except NoProcessingError:
                raise NoDependencyProcessError("The required process {p} has not been run for "
                                               "{mn}.".format(p=dp, mn=mission_name))

        # If we sum the run success array and its 0, then that means nothing ran successfully so we actually throw
        #  an exception - though only if we haven't been told not too.
        if run_success.sum() == 0 and no_success_error:
            process_plural = '(es)' if len(dep_proc) > 1 else ''
            raise NoDependencyProcessError("The required process{pp} {p} was run for {mn}, but was not "
                                           "successful for any data.".format(p=', '.join(dep_proc),
                                                                             mn=mission_name,
                                                                             pp=process_plural))

        return run_success

    def info(self):
        """
        A simple method to present summary information about this archive.
        """
        print("\n-----------------------------------------------------")
        print("Number of missions - {}".format(len(self)))
        print("Total number of observations - {}".format(sum([len(m) for m in self._missions.values()])))
        print("Beginning of earliest observation - {}".format(min([m.filtered_obs_info['start'].min()
                                                                   for m in self._missions.values()])))
        print("End of latest observation - {}".format(max([m.filtered_obs_info['end'].max()
                                                           for m in self._missions.values()])))
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
            raise ValueError("Only a mission name or integer index may be used to address an archive object")
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

