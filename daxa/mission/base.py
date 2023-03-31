#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 29/03/2023, 11:35. Copyright (c) The Contributors

import os.path
import re
from abc import ABCMeta, abstractmethod
from datetime import datetime
from functools import wraps
from typing import List, Union
from warnings import warn

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, BaseRADecFrame
from astropy.coordinates.name_resolve import NameResolveError
from astropy.units import Quantity
from tabulate import tabulate

from daxa import OUTPUT
from daxa.exceptions import MissionLockedError, NoObsAfterFilterError, IllegalSourceType, NoTargetSourceTypeInfo

# These are the columns which MUST be present in the all_obs_info dataframes of any sub-class of BaseMission. This
#  is mainly implemented to make sure developers who aren't me provide the right data formats
REQUIRED_COLS = ['ra', 'dec', 'ObsID', 'usable', 'start', 'duration', 'end']
# This defines the DAXA source category system, which can be employed by users to narrow down observations which
#  target specific types of source (if that data is available for a specific mission).
SRC_TYPE_TAXONOMY = {'AGN': 'Active Galaxies and Quasars', 'BLZ': 'Blazars',
                     'CAL': 'Calibration Observation (possibly of objects)', 'EGS': 'Extragalactic Surveys',
                     'GCL': 'Galaxy Clusters', 'GS': 'Galactic Survey',
                     'MAG': 'Magnetars and Rotation-Powered Pulsars', 'NGS': 'Normal and Starburst Galaxies',
                     'OAGN': 'Obscured Active Galaxies and Quasars', 'SNE': 'Non-ToO Supernovae',
                     'SNR': 'Supernova Remnants and Galactic diffuse', 'SOL': 'Solar System Observations',
                     'ULX': 'Ultra-luminous X-ray Sources', 'XRB': 'X-ray Binaries', 'TOO': 'Targets of Opportunity',
                     'EGE': 'Extended galactic or extragalactic', 'MISC': "Catch-all for other sources"}


def _lock_check(change_func):
    """
    An internal function designed to be used as a decorator for any methods of a mission class that can make
    changes to the selected observations - if the mission instance has been locked (i.e. the .locked property
    setter has been set to True) then this decorator will not allow the change.

    :param change_func: The method which is attempting to make changes to the selected observation data.
    """

    # The wraps decorator updates the wrapper function to look like wrapped function by copying attributes
    #  such as __name__, __doc__ (the docstring)
    @wraps(change_func)
    def wrapper(*args, **kwargs):
        # The first argument will be 'self' for any class method, so we check its 'locked' property
        if not args[0].locked:
            # If not locked then we can execute that method without any worries
            change_func(*args, **kwargs)
        else:
            # If the mission is locked then we have to throw an error
            raise MissionLockedError("This mission instance has been locked, and is now immutable.")

    return wrapper


class BaseMission(metaclass=ABCMeta):
    """
    The superclass for all missions defined in this module. Mission classes will be for storing and interacting
    with information about the available data for particular missions; including filtering the observations to be
    prepared and reduced in various ways. The mission classes will also be responsible for providing a consistent
    user experience of downloading data and generating processed archives.
    """

    def __init__(self):
        """
        The __init__ of the superclass for all missions defined in this module. Mission classes will be for storing
        and interacting with information about the available data for particular missions; including filtering
        the observations to be prepared and reduced in various ways. The mission classes will also be responsible
        for providing a consistent user experience of downloading data and generating processed archives.
        """
        # The string name of this mission, is overwritten in abstract properties required to be implemented
        #  by each subclass of BaseMission
        self._miss_name = None
        # Used for things like progress bar descriptions
        self._pretty_miss_name = None

        # The coordinate frame (e.g. FK5, ICRS) which the mission defines its coordinates in. Again to be
        #  overwritten in abstract properties in subclasses.
        self._miss_coord_frame = None
        # This will be overwritten in the init of subclasses if there are any required columns specific to that
        #  mission to be stored in the all observation information dataframe
        self._required_mission_specific_cols = []
        # All possible instruments are stored in this attribute in the init of a subclass
        self._miss_poss_insts = []
        # This attribute stores the instruments which have actually been chosen
        self._chos_insts = []
        # This is for missions that might have multiple common names for instruments, so they can be converted
        #  to the version expected by this module.
        self._alt_miss_inst_names = {}

        # This is again overwritten in abstract properties in subclasses, but this is the regular expression which
        #  observation identifiers for a particular mission must follow.
        self._id_format = None
        # This is what the overall observation information dataframe is stored in.
        self._obs_info = None

        # The output path is defined in the configuration file - considered allowing users to overwrite it
        #  when setting up missions but that then over-complicates the definition of archives (a user could
        #  conceivably set up different output directories for different missions).
        # We make sure that directory actually exists
        if not os.path.exists(OUTPUT):
            os.makedirs(OUTPUT)

        # This top level output path will have sub-directories in for the actual storing of raw files
        #  and processed archives
        self._top_level_output_path = OUTPUT

        # This sets up the filter array storage attribute.
        self._filter_allowed = None

        # This is set to True once the specified raw data for a mission have been downloaded
        self._download_done = False

        # If this is set to True then no further changes to the selection of observations in a mission
        #  will be allowed. This will be automatically applied when missions are added to an archive.
        self._locked = False

        # This attribute is for making sure the mission instance (and thus whatever archive it might be a
        #  part of) knows whether or not the raw data have been processed.
        self._processed = False

    # Defining properties first
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract property getter for the name of this mission. Must be overwritten in any subclass. This is to
        ensure that any subclasses that people might add will definitely set a proper name, which is not
        guaranteed by having it done in the init.

        :return: The mission name
        :rtype: str
        """
        # This is defined here (as well as in the init of BaseMission) because I want people to just copy this
        #  property if they're making a new subclass, then replace None with the name of the mission.
        self._miss_name = None
        # Used for things like progress bar descriptions
        self._pretty_miss_name = None
        return self._miss_name

    @property
    def pretty_name(self) -> str:
        """
        The property getter for the 'pretty name' of this mission. This version of the name will NOT be used
        to identify a mission internally in DAXA, or to name any directories, but will be used when the user
        sees a name (e.g. when a progress bar is running for a mission download).

        :return: The 'pretty' name.
        :rtype: str
        """
        if self._pretty_miss_name is None:
            raise ValueError("This mission class has not been fully setup (by the programmer), and the "
                             "_pretty_miss_name attribute is None - please set it in the name property of the "
                             "mission subclass.")
        else:
            return self._pretty_miss_name

    @property
    @abstractmethod
    def coord_frame(self) -> BaseRADecFrame:
        """
        Abstract property getter for the coordinate frame of the RA-Decs of the observations of this mission. Must
        be overwritten in any subclass. This is to ensure that any subclasses that people might add will definitely
        set a coordinate frame, which is not guaranteed by having it done in the init.

        :return: The coordinate frame of the RA-Dec
        :rtype: BaseRADecFrame
        """
        # This is defined here (as well as in the init of BaseMission) because I want people to just copy this
        #  property if they're making a new subclass, then replace None with the coordinate frame the mission uses.
        self._miss_coord_frame = None
        return self._miss_coord_frame

    @property
    @abstractmethod
    def id_regex(self) -> str:
        """
        Abstract property getter for the regular expression (regex) pattern for observation IDs of this mission. Must
        be overwritten in any subclass. This is to ensure that any subclasses that people might add will definitely
        set an ID pattern, which is not guaranteed by having it done in the init.

        :return: The regex pattern for observation IDs.
        :rtype: str
        """
        # This is defined here (as well as in the init of BaseMission) because I want people to just copy this
        #  property if they're making a new subclass, then replace None with the ID regular expression
        #  the mission uses.
        self._id_format = None
        return self._id_format

    @property
    def all_mission_instruments(self) -> List[str]:
        """
        Property getter for the names of all possible instruments associated with this mission.

        :return: A list of instrument names.
        :rtype: List[str]
        """
        return self._miss_poss_insts

    @property
    def chosen_instruments(self) -> List[str]:
        """
        Property getter for the names of the currently selected instruments associated with this mission which
        will be processed into an archive by DAXA functions.

        :return: A list of instrument names
        :rtype: List[str]
        """
        return self._chos_insts

    @chosen_instruments.setter
    @_lock_check
    def chosen_instruments(self, new_insts: List[str]):
        """
        Property setter for the instruments associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value.

        :param List[str] new_insts: The new list of instruments associated with this mission which should
            be processed into the archive.
        """
        self._chos_insts = self._check_chos_insts(new_insts)

    @property
    def top_level_path(self) -> str:
        """
        The property getter for the absolute path to the top-level directory where raw data storage directories
        are created.

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
    def raw_data_path(self) -> str:
        """
        Property getter for the directory in which raw data for the current mission is stored.

        :return: Storage path for raw data for this mission.
        :rtype: str
        """
        return self.top_level_path + self.name + '_raw/'

    @property
    def filter_array(self) -> np.ndarray:
        """
        A property getter for the 'filter' array, which is set by the filtering methods built-in to this class
        (or can be set externally using the filter_array property setter) and controls which observations will
        be downloaded and processed.

        :return: An array of boolean values; True means that an observation is used, False means that it is not.
        :rtype: np.ndarray
        """
        # Bit cheesy but if a subclass forgot to setup a proper filter array, then we can do it automatically
        if self._filter_allowed is None:
            self.reset_filter()

        return self._filter_allowed

    @filter_array.setter
    def filter_array(self, new_filter_array: np.ndarray):
        """
        A property setter for the 'filter' array which controls which observations will be downloaded and processed.
        The new passed filter array must be an array of boolean values, where True means an observation will be used
        and False means it will not; the array must be the same length as the all_obs_info dataframe.

        :param np.ndarray new_filter_array: The new filter array to be checked and stored. An array of boolean
            values; True means that an observation is used, False means that it is not.
        """
        if new_filter_array.dtype != bool:
            raise TypeError("Please pass an array of boolean values for the filter array.")
        elif len(new_filter_array) != len(self._obs_info):
            raise ValueError("Length of the filter array ({lf}) does not match the length of the dataframe containing"
                             " all observation information for this mission ({la}).".format(lf=len(new_filter_array),
                                                                                            la=len(self._obs_info)))
        elif new_filter_array.sum() == 0:
            raise NoObsAfterFilterError("Every value in the filter array is False, meaning that no observations "
                                        "remain. As such the new filter array has not been accepted")
        else:
            self._filter_allowed = new_filter_array
            # If the filter changes then we make sure download done is set to False so that any changes
            #  in observation selection are reflected in the download call
            self._download_done = False

    @property
    @abstractmethod
    def all_obs_info(self) -> pd.DataFrame:
        """
        A property getter that returns the base dataframe containing information about all the observations available
        for an instance of a mission class. This is an abstract method purely because its property setter is an
        abstract method, one cannot be without the other.

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
    def filtered_obs_info(self) -> pd.DataFrame:
        """
        A property getter that applies the current filter array to the dataframe of observation information, and
        returns filtered dataframe containing all columns available for this mission.

        :return: A filtered dataframe of observation information.
        :rtype: pd.DataFrame
        """
        return self._obs_info[self.filter_array]

    @property
    def usable(self) -> np.ndarray:
        """
        Property getter for the usable column of the all observation information dataframe. This usable column
        describes whether a particular observation is actually usable by this module; for instance that the data
        are suitable for scientific use (so far as can be identified by querying the storage service) and are not
        proprietary. This usable property is the basis for the filter array, resetting the filter array will return
        it to the values of this column.

        :return:
        :rtype: np.ndarray
        """
        return self.all_obs_info['usable'].values

    @property
    def ra_decs(self) -> SkyCoord:
        """
        Property getter for the RA-Dec coordinates of ALL the observations associated with this mission - for the
        coordinates of filtered observations (i.e. the observations that will actually be used for
        downloading/processing), see the filtered_ra_decs property.

        :return: The full set of RA-Dec coordinates of all observations associated with this mission.
        :rtype: SkyCoord
        """
        return SkyCoord(self._obs_info['ra'].values, self._obs_info['dec'].values, unit=u.deg, frame=self.coord_frame)

    @property
    def filtered_ra_decs(self) -> SkyCoord:
        """
        Property getter for the RA-Dec coordinates of the filtered set of observations associated with this
        mission - for coordinates of ALL observations see the ra_decs property.

        :return: The RA-Dec coordinates of filtered observations associated with this mission.
        :rtype: SkyCoord
        """
        return SkyCoord(self._obs_info['ra'].values[self.filter_array],
                        self._obs_info['dec'].values[self.filter_array], unit=u.deg, frame=self.coord_frame)

    @property
    def obs_ids(self) -> np.ndarray:
        """
        Property getter for the ObsIDs of ALL the observations associated with this mission - for the
        ObsIDs of filtered observations (i.e. the observations that will actually be used for
        downloading/processing), see the filtered_obs_ids property.

        :return: The full set of ObsIDs of all observations associated with this mission.
        :rtype: np.ndarray
        """
        return self._obs_info['ObsID'].values

    @property
    def filtered_obs_ids(self) -> np.ndarray:
        """
        Property getter for the ObsIDs of the filtered set of observations associated with this
        mission - for ObsIDs of ALL observations see the obs_ids property.

        :return: The ObsIDs of filtered observations associated with this mission.
        :rtype: np.ndarray
        """
        return self._obs_info['ObsID'].values[self.filter_array]

    @property
    def download_completed(self) -> bool:
        """
        Property getter that describes whether the specified raw data for this mission have been
        downloaded.

        :return: Boolean flag describing if data have been downloaded.
        :rtype: bool
        """
        return self._download_done

    @property
    def locked(self) -> bool:
        """
        Property getter for the locked attribute of this mission instance - if a mission is locked
        then no further changes can be made to the observations selected.

        :return: The locked boolean.
        :rtype: bool
        """
        return self._locked

    @locked.setter
    def locked(self, new_val: bool):
        """
        Property setter for the locked state of the mission instance. New values must be boolean, and if a
        mission has already been locked by setting locked = True, it cannot be unlocked again.

        :param bool new_val: The new locked value.
        """
        if not isinstance(new_val, bool):
            raise TypeError("The value of locked must be a boolean.")

        if self._locked:
            raise MissionLockedError("This mission has already been locked, you cannot unlock it.")
        else:
            self._locked = new_val

    @property
    def processed(self) -> bool:
        """
        A property getter that returns whether the observations associated with this mission have been
        fully processed or not.

        :return: The processed boolean flag.
        :rtype: bool
        """
        return self._processed

    @processed.setter
    def processed(self, new_val: bool):
        """
        A property setter for whether the observations associated with this mission have been fully
        processed or not. If processed has already been set to True, then it cannot be reset to False, and once
        processed has been set to True, the 'locked' property will also be set to True and the observation
        selection for this mission instance will become immutable.

        :param bool new_val: The new value for processed.
        """
        if not isinstance(new_val, bool):
            raise TypeError("New values for 'processed' must be boolean.")
        elif self._processed:
            raise ValueError("The processed property has already been set to True, and is now immutable.")
        elif new_val:
            self.locked = True
        self._processed = new_val

    # Then define internal methods
    def _obs_info_checks(self, new_info: pd.DataFrame):
        """
        Performs very simple checks on new inputs into the observation information dataframe, ensuring it at
        has the minimum required columns. This column check looks for both the columns defined in the REQUIRED_COLS
        constant, and the extra columns which can be required for individual missions defined in each mission
        subclass' __init__.

        :param pd.DataFrame new_info: The new dataframe of observation information that should be checked.
        """
        if not isinstance(new_info, pd.DataFrame) or not all([col in new_info.columns for col in
                                                              REQUIRED_COLS + self._required_mission_specific_cols]):
            raise ValueError("New all_obs_info values for this mission must be a Pandas dataframe with the following "
                             "columns; {}".format(', '.join(REQUIRED_COLS + self._required_mission_specific_cols)))

        if 'target_category' in new_info.columns:
            # Checking for target types in the obsinfo dataframe that are not in the DAXA taxonomy
            tt_check = [tt for tt in new_info['target_category'].value_counts().index.values
                        if tt not in SRC_TYPE_TAXONOMY]
            if len(tt_check) != 0:
                # Throw a hopefully useful error if the user has passed illegal values
                raise IllegalSourceType("Unsupported target type(s) ({it}) are present in the new observation info "
                                        "dataframe, use one of the following; "
                                        "{at}".format(it=', '.join(tt_check),
                                                      at=', '.join(list(SRC_TYPE_TAXONOMY.keys()))))

    def _check_chos_insts(self, insts: Union[List[str], str]):
        """
        An internal function to perform some checks on the validity of chosen instrument names for a given mission.

        :param List[str]/str insts:
        :return: The list of instruments (possibly altered to match formats expected by this module).
        :rtype: List
        """
        # Just makes sure we can iterate across instrument(s), regardless of how many there are
        if not isinstance(insts, list):
            insts = [insts]
        
        # Raising and error if the input is not Union[List[str], str]
        if not all(isinstance(inst, str) for inst in insts):
            raise TypeError("Instruments must be input as a string or a list of strings.")
        
        # Making sure the input is capitalised for compatibilty with the rest of the module
        insts = [i.upper() for i in insts]

        # I just check that there are actually entries in this list of instruments, because it would be silly if
        #  there weren't
        if len(insts) == 0:
            raise ValueError("No instruments have been selected, please pass at least one.")

        # I just check that there are actually entries in this list of instruments, because it would be silly if
        #  there weren't
        if len(insts) == 0:
            raise ValueError("No instruments have been selected, please pass at least one.")

        # This is clunky and inefficient but should be fine for these very limited purposes. It just checks whether
        #  this module has a preferred name for a particular instrument. We can also make sure that there are no
        #  duplicate instrument names here
        updated_insts = []
        altered = False
        for i in insts:
            if i in self._alt_miss_inst_names:
                altered = True
                inst_name = self._alt_miss_inst_names[i]
            else:
                inst_name = i

            # Checks for duplicate names as we go along
            if inst_name not in updated_insts:
                updated_insts.append(inst_name)

        # I warn the user if the name(s) of instruments have been altered.
        if altered:
            warn("Some instrument names were converted to alternative forms expected by this module, the instrument "
                 "names are now; {}".format(', '.join(updated_insts)))

        # This list comprehension checks that the input instrument names are in the allowed instruments for this
        #  particular mission
        inst_test = [i in self._miss_poss_insts for i in updated_insts]
        # If some aren't then we throw an error (hopefully quite an informative one).
        if not all(inst_test):
            bad_inst = np.array(updated_insts)[~np.array(inst_test)]
            print(bad_inst)
            raise ValueError("Some instruments ({bi}) are not associated with this mission, please choose from "
                             "the following; {ai}".format(bi=", ".join(bad_inst), ai=", ".join(self._miss_poss_insts)))

        # Return the possibly altered instruments
        return updated_insts

    # Then define user-facing methods
    @abstractmethod
    def _fetch_obs_info(self):
        """
        The abstract method (i.e. will be overridden in every subclass of BaseMission) that pulls basic information
        on all observations for a given mission down from whatever server it lives on.
        """
        # self.all_obs_info = None
        pass

    def reset_filter(self):
        """
        Very simple method which simply resets the filter array, meaning that all observations THAT HAVE BEEN
        MARKED AS USABLE will now be downloaded and processed, and any filters applied to the current mission
        have been undone.
        """
        self._filter_allowed = self.all_obs_info['usable'].values.copy()
        # If the filter changes then we make sure download done is set to False so that any changes
        #  in observation selection are reflected in the download call
        self._download_done = False

    def check_obsid_pattern(self, obs_id_to_check: str):
        """
        A simple method that will check an input ObsID against the ObsID regular expression pattern defined
        for the current mission class. If the input ObsID is compliant with the regular expression then
        True will be returned, if not then False will be returned.

        :param str obs_id_to_check: The ObsID that we wish to check against the ID pattern.
        :return: A boolean flag indicating whether the input ObsID is compliant with the ID regular expression.
            True means that it is, False means it is not.
        :rtype: bool
        """
        return bool(re.match(self.id_regex, obs_id_to_check))

    @_lock_check
    def filter_on_obs_ids(self, allowed_obs_ids: Union[str, List[str]]):
        """
        This filtering method will select only observations with IDs specified by the allowed_obs_ids argument.

        Please be aware that filtering methods are cumulative, so running another method will not remove the
        filtering that has already been applied, you can use the reset_filter method for that.

        :param str/List[str] allowed_obs_ids: The ObsID (or list of ObsIDs) that you wish to be let
            through the filter.
        """
        # Makes sure that the allowed_obs_ids variable is iterable over ObsIDs, even if just a single ObsID was passed
        if not isinstance(allowed_obs_ids, list):
            allowed_obs_ids = [allowed_obs_ids]

        # Runs the ObsID pattern checks for all the passed ObsIDs
        oid_check = [oid for oid in allowed_obs_ids if not self.check_obsid_pattern(oid)]
        if len(oid_check) != 0:
            # Raises an error if the ObsIDs don't all conform to the expected pattern defined for each mission.
            raise ValueError("One or more ObsID passed into this method does not match the expected pattern "
                             "for ObsIDs of this mission. The following are not compliant; "
                             "{}".format(', '.join(oid_check)))

        # Uses the Pandas isin functionality to find the rows of the overall observation table that match the input
        #  ObsIDs. This outputs a boolean array.
        sel_obs_mask = self._obs_info['ObsID'].isin(allowed_obs_ids)
        # Said boolean array can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array * sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter

    # TODO Figure out how to support survey-type missions (i.e. eROSITA) that release large sweeps of the sky
    #  when filtering based on position.
    @_lock_check
    def filter_on_rect_region(self, lower_left: Union[SkyCoord, np.ndarray, list],
                              upper_right: Union[SkyCoord, np.ndarray, list]):
        """
        A method that filters observations based on whether their CENTRAL COORDINATE falls within a rectangular
        region defined using coordinates of the bottom left and top right corners. Observations are kept if they
        fall within the region.

        Please be aware that filtering methods are cumulative, so running another method will not remove the
        filtering that has already been applied, you can use the reset_filter method for that.

        :param SkyCoord/np.ndarray/list lower_left: The RA-Dec coordinates of the lower left corner of the
            rectangular region. This can be passed as a SkyCoord, or a list/array with two entries - this
            will then be used to create a SkyCoord which assumes the default frame of the current mission and
            that the inputs are in degrees.
        :param SkyCoord/np.ndarray/list upper_right: The RA-Dec coordinates of the upper right corner of the
            rectangular region. This can be passed as a SkyCoord, or a list/array with two entries - this
            will then be used to create a SkyCoord which assumes the default frame of the current mission and
            that the inputs are in degrees.
        """
        # Checks to see if the user has passed the lower left coordinate as an array with an RA and Dec, rather
        #  than as an initialized SkyCoord. If so then we set up a SkyCoord assuming the default frame of this mission.
        if isinstance(lower_left, (list, np.ndarray)):
            lower_left = SkyCoord(*lower_left, unit=u.deg, frame=self.coord_frame)

        # Checks to see if the user has passed the upper right coordinate as an array with an RA and Dec, rather
        #  than as an initialized SkyCoord. If so then we set up a SkyCoord assuming the default frame of this mission.
        if isinstance(upper_right, (list, np.ndarray)):
            upper_right = SkyCoord(*upper_right, unit=u.deg, frame=self.coord_frame)

        # Creates a filter based on a rectangular region defined by the input coordinates
        box_filter = (self.ra_decs.ra >= lower_left.ra) & (self.ra_decs.ra <= upper_right.ra) & \
                     (self.ra_decs.dec >= lower_left.dec) & (self.ra_decs.dec <= upper_right.dec)

        # Have to check whether any observations have actually been found, if not then we throw an error
        if box_filter.sum() == 0:
            raise NoObsAfterFilterError("The box search has returned no {} observations.".format(self.pretty_name))

        # Updates the filter array
        new_filter = self.filter_array * box_filter
        self.filter_array = new_filter

    @_lock_check
    def filter_on_positions(self, positions: Union[list, np.ndarray, SkyCoord],
                            search_distance: Union[Quantity, float, int]):
        """
        This method allows you to filter the observations available for a mission based on a set of coordinates for
        which you wish to locate observations. The method searches for observations by the current mission that have
        central coordinates within the distance set by the search_distance argument.

        Please be aware that filtering methods are cumulative, so running another method will not remove the
        filtering that has already been applied, you can use the reset_filter method for that.

        :param list/np.ndarray/SkyCoord positions: The positions for which you wish to search for observations. They
            can be passed either as a list or nested list (i.e. [r, d] OR [[r1, d1], [r2, d2]]), a numpy array, or
            an already defined SkyCoord. If a list or array is passed then the coordinates are assumed to be in
            degrees, and the default mission frame will be used.
        :param Quantity/float/int search_distance: The distance within which to search for observations by this
            mission. Distance may be specified as an Astropy Quantity that can be converted to degrees, or as a
            float/integer that will be assumed to be in units of degrees.
        """

        # Checks to see if a list/array of coordinates has been passed, in which case we convert it to a
        #  SkyCoord (or a SkyCoord catalogue).
        if isinstance(positions, (list, np.ndarray)):
            positions = SkyCoord(positions, unit=u.deg, frame=self.coord_frame)

        # This is slightly cheesy, but the search_around_sky method will only work if there is a catalog
        #  of positions that is being searched around, rather than a single position. As such if a single
        #  coordinate is being searched around I just duplicate it to placate the method. This won't produce
        #  any ill effects because I just care about which observations are nearby, not which coordinates are
        #  specifically matched to which observation.
        if positions.isscalar:
            positions = SkyCoord([positions.ra, positions.ra], [positions.dec, positions.dec], unit=u.deg,
                                 frame=positions.frame)

        # Checks to see whether a quantity has been passed, if not then the input is converted to an Astropy
        #  quantity in units of degrees. If a Quantity that cannot be converted to degrees is passed then the
        #  else part of the statement will error.
        if not isinstance(search_distance, Quantity):
            search_distance = Quantity(search_distance, 'deg')
        else:
            search_distance = search_distance.to('deg')

        # Runs the 'catalogue matching' between all available observations and the input positions.
        which_pos, which_obs, d2d, d3d = self.ra_decs.search_around_sky(positions, search_distance)

        # Have to check whether any observations have actually been found, if not then we throw an error
        if len(which_obs) == 0:
            raise NoObsAfterFilterError("The positional search has returned no {} "
                                        "observations.".format(self.pretty_name))

        # Sets up a filter array that consists entirely of zeros initially (i.e. it would not let
        #  any observations through).
        pos_filter = np.zeros(self.filter_array.shape)
        # The which_obs array indicates which of the entries in the table of observation info for this mission are
        #  matching to one or more of the positions passed. The list(set()) setup is used to ensure that there are
        #  no duplicates. These entries in the pos_filter are set to one, which will allow those observations through
        pos_filter[np.array(list(set(which_obs)))] = 1
        # Convert the array of ones and zeros to boolean, which is what the filter_array property setter wants
        pos_filter = pos_filter.astype(bool)
        # Create the combination of the existing filter array and the new position filter
        new_filter = self.filter_array * pos_filter
        # And update the filter array
        self.filter_array = new_filter

    @_lock_check
    def filter_on_name(self, object_name: Union[str, List[str]], search_distance: Union[Quantity, float, int],
                       parse_name: bool = False):
        """
        This method wraps the 'filter_on_positions' method, and allows you to filter the mission's observations so
        that it contains data on a single (or a list of) specific objects. The names are passed by the user, and
        then parsed into coordinates using the Sesame resolver. Those coordinates and the search distance are
        then used to find observations that might be relevant.

        :param str/List[str] object_name: The name(s) of objects you would like to search for.
        :param Quantity/float/int search_distance: The distance within which to search for observations by this
            mission. Distance may be specified as an Astropy Quantity that can be converted to degrees, or as a
            float/integer that will be assumed to be in units of degrees.
        :param bool parse_name: Whether to attempt extracting the coordinates from the name by parsing with a regex.
            For objects catalog names that have J-coordinates embedded in their names, e.g.,
            'CRTS SSS100805 J194428-420209', this may be much faster than a Sesame query for the same object name.
        """
        # Turn a single name into a list with a single entry - normalises it for the rest of the method
        if isinstance(object_name, str):
            object_name = [object_name]

        # This is the list where coordinates will be stored
        coords = []
        # Any failed lookups will be stored in here, and the user will be warned that they couldn't be resolved.
        bad_names = []
        # Cycling through the names
        for n_ind, name in enumerate(object_name):
            # Try except is necessary to deal with the possibility of the name not being resolved
            try:
                # We read the coordinates out into the frame of mission, and let the user decide whether
                #  they want to use the parsing ability in from_name
                coords.append(SkyCoord.from_name(name, frame=self.coord_frame, parse=parse_name))
            except NameResolveError:
                # If we could not resolve the name, we save said name for the warning later
                bad_names.append(name)

        # Have to check whether there are any coordinates that have been resolved, if not we throw an error
        if len(coords) == 0:
            raise NameResolveError("The name(s) could be resolved into coordinates.")

        # Also, if this list has any entries, then some names failed to resolve (but if we're here then some of the
        #  names WERE resolved)
        if len(bad_names) != 0:
            # Warn the user what happened, with the names, so they can do some diagnosis
            warn('Some of the object names ({}) could not be resolved by Sesame'.format(', '.join(bad_names)),
                 stacklevel=2)

        # This combines the coordinate list into just one SkyCoord instance, with multiple coordinate entries. Now
        #  we can use this with the ObsID filtering method
        coords = SkyCoord(coords)

        # Now we just call the 'filter_on_positions' method
        self.filter_on_positions(coords, search_distance)

    @_lock_check
    def filter_on_time(self, start_datetime: datetime, end_datetime: datetime, over_run: bool = False):
        """
        This method allows you to filter observations for this mission based on when they were taken. A start
        and end time are passed by the user, and observations that fall within that window are allowed through
        the filter. The exact behaviour of this filtering method is controlled by the over_run argument, if set
        to True then observations with a start or end within the search window will be selected, but if False
        then only observations with a start AND end within the window are selected.

        Please be aware that filtering methods are cumulative, so running another method will not remove the
        filtering that has already been applied, you can use the reset_filter method for that.

        :param datetime start_datetime: The beginning of the time window in which to search for observations.
        :param datetime end_datetime: The end of the time window in which to search for observations.
        :param bool over_run: This controls whether selected observations have to be entirely within the passed
            time window or whether either a start or end time can be within the search window. If set
            to True then observations with a start or end within the search window will be selected, but if False
            then only observations with a start AND end within the window are selected.
        """
        # This just selects the exact behaviour of whether an observation is allowed through the filter or not.
        if not over_run:
            time_filter = (self.all_obs_info['start'] >= start_datetime) & (self.all_obs_info['end'] <= end_datetime)
        else:
            time_filter = ((self.all_obs_info['start'] >= start_datetime) &
                           (self.all_obs_info['start'] <= end_datetime)) | \
                          ((self.all_obs_info['end'] >= start_datetime) & (self.all_obs_info['end'] <= end_datetime))

        # Have to check whether any observations have actually been found, if not then we throw an error
        if time_filter.sum() == 0:
            raise NoObsAfterFilterError("The temporal search has returned no {} "
                                        "observations.".format(self.pretty_name))

        # Combines the time filter with the existing filter and updates the property.
        new_filter = self.filter_array * time_filter
        self.filter_array = new_filter

    @_lock_check
    def filter_on_target_type(self, target_type: Union[str, List[str]]):
        """
        This method allows the filtering of observations based on what type of object their target source was. It
        is only supported for missions that have that data available, and will raise an exception for those
        missions that don't support this filtering.

        WARNING: You should not trust these target types without question, they are the result of crude mappings, and
        some may be incorrect. They also don't take into account sources that might serendipitously appear in
        a particular observation.

        :param str/List[str] target_type: The types of target source you would like to find observations of. For
            allowed types, please use the 'show_allowed_target_types' method. Can either be a single type, or
            a list of types.
        """
        # If only one target type is passed, we still make sure it's a list - normalises it for the rest
        #  of the method
        if isinstance(target_type, str):
            target_type = [target_type]
        # Also make sure whatever the user has passed is set to all uppercase
        target_type = [tt.upper() for tt in target_type]

        # Look for passed target types that AREN'T in the DAXA taxonomy
        tt_check = [tt for tt in target_type if tt not in SRC_TYPE_TAXONOMY]
        if len(tt_check) != 0:
            # Throw a hopefully useful error if the user has passed illegal values
            raise IllegalSourceType("Unsupported target type(s) ({it}) have been passed to this method, use one of the "
                                    "following; {at}".format(it=', '.join(tt_check),
                                                             at=', '.join(list(SRC_TYPE_TAXONOMY.keys()))))

        # If there is no information on target source types in the observation info dataframe, then unfortunately
        #  this method can't be used.
        if 'target_category' not in self.all_obs_info.columns:
            raise NoTargetSourceTypeInfo("No target source type information is available "
                                         "for {}".format(self.pretty_name))

        # This creates a boolean array of dataframe entries that match the selected target type(s)
        sel_obs_mask = self._obs_info['target_category'].isin(target_type)
        # Check that we actually selected some observations
        if sel_obs_mask.sum() == 0:
            raise NoObsAfterFilterError("The target type search has returned no {} "
                                        "observations.".format(self.pretty_name))

        # The boolean array can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array * sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter

    def info(self):
        print("\n-----------------------------------------------------")
        print("Number of Observations - {}".format(len(self)))
        print("Number of Filtered Observations - {}".format(len(self.filtered_obs_info)))
        print("Total Duration - {}".format(self.all_obs_info['duration'].sum()))
        print("Total Filtered Duration - {}".format(self.filtered_obs_info['duration'].sum()))
        print("Earliest Observation Date - {}".format(self.all_obs_info['start'].min()))
        print("Latest Observation Date - {}".format(self.all_obs_info['end'].max()))
        print("Earliest Filtered Observation Date - {}".format(self.filtered_obs_info['start'].min()))
        print("Latest Filtered Observation Date - {}".format(self.filtered_obs_info['end'].max()))
        print("-----------------------------------------------------\n")

    @abstractmethod
    def download(self):
        """
        An abstract method to actually acquire and download the mission data that have not been filtered out (if
        a filter has been applied, otherwise all data will be downloaded). This must be overwritten by every subclass
        as each mission might need a different method of downloading the data, the same reason fetch_obs_info
        must be overwritten in each subclass.
        """
        pass

    @abstractmethod
    def assess_process_obs(self, obs_info: dict):
        """
        A slightly unusual abstract method which will allow each mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument-subexposure (for missions like XMM) should be processed further for
        scientific use.

        Implemented as an abstract method because the information and decision-making process will likely be
        different for every mission.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        pass

    @staticmethod
    def show_allowed_target_types(table_format: str = 'fancy_grid'):
        """
        This simple method just displays the DAXA source type taxonomy (the target source types you can filter by)
        in a nice table, with descriptions of what each source type means. Filtering on target source type is not
        guaranteed to work with every mission, as target type information is not necessarily available, but this
        filtering is used through the filter_on_target_type method.

        :param str table_format: The style format for the table to be displayed (should be one of the 'tabulate'
            module formats). The default is 'fancy_grid'.
        """
        # Reads out the keys (i.e. what the user can filter with), and their descriptions
        data = [[k, v] for k, v in SRC_TYPE_TAXONOMY.items()]
        # Create the two column titles
        cols = ['Target Type', 'Description']
        # Now simply print them in a nice table
        print(tabulate(data, cols, tablefmt=table_format))

    def __len__(self):
        """
        The method triggered by the len() operator, returns the number of observations in the filtered,
        info dataframe for this mission.

        :return: The number of observations for this mission that made it through the filter.
        :rtype: int
        """
        return len(self.filtered_obs_info)
