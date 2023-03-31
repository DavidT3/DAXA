#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 31/03/2023, 15:04. Copyright (c) The Contributors
import os
from shutil import rmtree
from typing import List, Union, Tuple
from warnings import warn

import numpy as np

from daxa import BaseMission, OUTPUT
from daxa.exceptions import DuplicateMissionError, ArchiveExistsError, NoProcessingError, NoDependencyProcessError
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
                     "made.".format(prn=pr_name, mn=mn))
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
                     "made.".format(prn=pr_name, mn=mn))
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
                     "made.".format(prn=pr_name, mn=mn))
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
                     "made.".format(prn=pr_name, mn=mn))
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
                     "made.".format(prn=pr_name, mn=mn))
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
                     "made.".format(prn=pr_name, mn=mn))
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
                         "will be made.".format(o_id=o_id, mn=mn))
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

        Retrieving a data path from this method DOES NOT guarantee that it has been created.

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
        if m_name is not None and m_name not in self.mission_names:
            raise ValueError("The mission {m} is not a part of this Archive; the current missions are "
                             "{cm}".format(m=m_name, cm=', '.join(self.mission_names)))

        # Need to make sure that the passed ObsID (if one was passed) is in the correct format for the
        #  specified mission - can use this handy method for that.
        if obs_id is not None and not self[m_name].check_obsid_pattern(obs_id):
            raise ValueError("The passed ObsID ({oi}) does not match the pattern expected for {mn} "
                             "identifiers.".format(mn=m_name, oi=obs_id))

        # Now we just run through the different possible combinations of circumstances.
        base_path = self.archive_path+'processed_data/{mn}/{oi}/'
        if mission is not None and obs_id is not None:
            ret_str = base_path.format(mn=m_name, oi=obs_id)
        elif mission is not None and obs_id is None:
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
                                 "mission, or a mission instrument enabled for this archive; otherwise it must be"
                                 "set to None.".format(si=search_ident))

        # Using dict_search and iterating (as it as a generator) - I know this method isn't super well explained, but
        #  in theory it should work for _use_this_obs which are upto four levels deep (mission-obsid-inst-subexp), and
        #  support the other possible depth where there aren't subexposures. Its disgusting and janky but I can't be
        #  bothered to improve it
        # The goal of these different processing
        #  steps is to construct a list of lists of [ObsID, Instrument, SubExp] that CAN be used, and the return
        #  from this function can be iterated through.
        for res in dict_search(search_ident, rel_use_obs):
            print(res)
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
                                 dep_proc: Union[str, List[str]]) -> np.ndarray:
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
        :return: A boolean array that defines whether the process(es) specified in the input were successful. Each
            set of identifying information provided in obs_ident has a corresponding entry in the return.
        :rtype: np.ndarray
        """
        # Check to make sure that the mission name is valid for this archive
        if mission_name not in self.mission_names:
            raise ValueError("The mission {mn} is not associated with this archive. Available missions are "
                             "{am}".format(mn=mission_name, am=', '.join(self.mission_names)))

        # Normalising the input so that dep_proc can always be iterated through. I imagine most of the time this
        #  method is used it will be for individual process checking, but you never know.
        if isinstance(dep_proc, str):
            dep_proc = [dep_proc]

        # I also want to normalise the obs_ident input as either a single set of identifying information, or
        #  multiple sets, can be passed to this method. Thus everything must become a list of lists
        if isinstance(obs_ident, list) and isinstance(obs_ident[0], str):
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
                    run_success[ident_ind] *= self.process_success[mission_name][dp][comb_id]

            # If we get the error thrown by process logs because nothing at all has been processed, we catch that
            #  and turn it into an error more useful for this specific method
            except NoProcessingError:
                raise NoDependencyProcessError("The required process {p} has not been run for "
                                               "{mn}.".format(p=dp, mn=mission_name))

        # If we sum the run success array and its 0, then that means nothing ran successfully so we actually throw
        #  an exception
        if run_success.sum() == 0:
            raise NoDependencyProcessError("The required process(es) {p} was run for {mn}, but was not "
                                           "successful for any data.".format(p=', '.join(dep_proc), mn=mission_name))

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

