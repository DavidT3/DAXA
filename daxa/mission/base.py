#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 09/04/2024, 13:44. Copyright (c) The Contributors
import inspect
import json
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
from daxa.exceptions import MissionLockedError, NoObsAfterFilterError, IllegalSourceType, NoTargetSourceTypeInfo, \
    DAXANotDownloadedError, IncompatibleSaveError

# This global helps to ensure that filtering functions that call another filtering function don't end up storing
#  every single filter in the filtering operations history - we only want the outer call (see _capture_filter for use)
_no_filtering_op_store = False

# These are the columns which MUST be present in the all_obs_info dataframes of any sub-class of BaseMission. This
#  is mainly implemented to make sure developers who aren't me provide the right data formats
REQUIRED_COLS = ['ra', 'dec', 'ObsID', 'science_usable', 'start', 'duration', 'end']
# This defines the DAXA source category system, which can be employed by users to narrow down observations which
#  target specific types of source (if that data is available for a specific mission).
SRC_TYPE_TAXONOMY = {'AGN': 'Active Galaxies and Quasars', 'BLZ': 'Blazars', 'CV': 'Cataclysmic Variables',
                     'CAL': 'Calibration Observation (possibly of objects)', 'EGS': 'Extragalactic Surveys',
                     'GCL': 'Galaxy Clusters', 'GS': 'Galactic Survey', 'ASK': 'All Sky Survey',
                     'MAG': 'Magnetars and Rotation-Powered Pulsars', 'NGS': 'Normal and Starburst Galaxies',
                     'NS': 'Neutron stars and Black Holes', 'STR': 'Non-degenerate and White Dwarf Stars',
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
            any_ret = change_func(*args, **kwargs)
        else:
            # If the mission is locked then we have to throw an error
            raise MissionLockedError("This mission instance has been locked, and is now immutable.")

        return any_ret

    return wrapper


def _capture_filter(change_func):
    """
    An internal function designed to be used as a decorator for any methods of a mission class that perform filtering
    operations on the available observations, to capture (and record) what filtering was performed, with what
    arguments, and in what order. That information can then be saved in the mission state, and any future reloading
    of the mission will be able to update itself by running the same filtering options on the more up to date list
    of observations.

    :param change_func: The method which is filtering the available mission data.
    """

    # The wraps decorator updates the wrapper function to look like wrapped function by copying attributes
    #  such as __name__, __doc__ (the docstring)
    @wraps(change_func)
    def wrapper(*args, **kwargs):
        # This global helps us to keep track of whether we should be recording the filtering operation information, as
        #  we DON'T want that to happen when one filtering function calls another (e.g. filter_on_name calls
        #  filter_on_positions)
        global _no_filtering_op_store

        # In this case, _no_filtering_op_store is True, which is NOT the default value, and we know then that this
        #  decorator has been triggered by a filtering operation called within another filtering operation, and we
        #  just want to run the filter without saving the information
        if _no_filtering_op_store:
            # First off we run the filtering method, so we don't save a filtering method that failed
            any_ret = change_func(*args, **kwargs)

        # However in this case, we know that this is the outermost filtering operation, so we're going to do more than
        #  just run the filtering method
        else:
            # First of all, we set the global flag to True, so if the filtering method we're about to call has calls
            #  to other filtering methods (and thus this decorator is triggered again), then the filter operation is
            #  not saved
            _no_filtering_op_store = True
            # Then we run the filtering method, so we don't save a filtering method that failed
            any_ret = change_func(*args, **kwargs)
            # And now we reset the global flag and continue on with saving the information we need to save
            _no_filtering_op_store = False

            # The first argument will be 'self' for any class method, which we need so we can add to the filtering
            #  operations history
            rel_miss = args[0]

            # If there are no positional arguments, then all will be well, and we just use the keyword arguments
            #  dictionary as the entry for the filtering operation history - otherwise we're going to need to add
            #  some information
            final_args = kwargs

            # In this case there are positional arguments other than 'self' - we care about these and need to add them
            #  to the arguments dictionary
            if len(args) != 1:
                # We extract the signature (i.e. the argument and type hints) part of the function
                meth_sig = inspect.signature(change_func)
                # Then we specifically extract an ordered dictionary of parameters
                meth_pars = meth_sig.parameters

                # This will store any positional arguments that have to be added to the final arguments dictionary
                pos_arg_vals = {}
                # Iterating through all the parameters
                for par_ind, par_name in enumerate(meth_pars):
                    # Read out the parameter object
                    cur_par = meth_pars[par_name]

                    # We don't care about self, so we skip it
                    if par_name == 'self':
                        continue
                    # As extracting the parameters from the function will also extract keyword arguments, we only
                    #  do things with the ones that DON'T already appear in the keyword arguments dictionary
                    elif par_name not in kwargs:
                        # In that case we can extract the value from the args tuple using the current positional index,
                        #  which I THINK should always correspond to the right value because meth_pars is an ordered
                        #  dictionary - this only works for non-keyword arguments though
                        if cur_par.default is cur_par.empty:
                            pos_arg_vals[par_name] = args[par_ind]
                        # If a keyword argument has a default value, it won't appear in kwargs, and the above case
                        #  is for positional arguments, so now we extract the default value from the signature
                        else:
                            pos_arg_vals[par_name] = cur_par.default
                # We add in the newly extracted positional arguments to the final argument dictionary
                final_args.update(pos_arg_vals)

            # Finally, we add the name of the filtering method to the filtering operations dictionary
            filtering_op_entry = {'name': change_func.__name__, 'arguments': final_args}
            # And we add it to the mission's filtering operations property, which will check it and store it
            rel_miss.filtering_operations = filtering_op_entry

        return any_ret
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

        # This attribute will be for the storage of an approximate field of view, ostensibly used to define a
        #  default value the search radius of filtering methods. In cases where there are multiple instruments,
        #  perhaps with different field of views, this attribute will be a dictionary.
        # Will take the same approach as the name property, where it is defined as an abstract method so it must
        #  be implemented for a new mission class
        self._approx_fov = None

        # This is a very rarely used attribute (I think only eROSITACalPV at the time of writing) that stores which
        #  particular named fields were chosen
        self._chos_fields = None

        # This attribute stores which type of data were downloaded, and are thus associated with this mission - there
        #  are three possible values; 'raw', 'preprocessed', or 'raw+preprocessed' (or four if you count the initial
        #  None value which is present until a download is actually done).
        # TODO need to actually have this set in the download methods of the various mission classes
        self._download_type = None

        # This attribute stores the filtering operations that have been applied to the current mission, including the
        #  configurations that were used - they are stored in the order they were performed; i.e. element 0 is the
        #  first applied and element N is the last
        self._filtering_operations = []

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
    @abstractmethod
    def fov(self) -> Union[Quantity, dict]:
        """
        Abstract property getter for the approximate field-of-view of this mission's instrument(s). In cases where
        different instruments have different field-of-views this may be a dictionary (see ROSATPointed for an
        example). Must be overwritten in any subclass. This is to ensure that any subclasses that people might
        add will definitely set a FoV, which is not guaranteed by having it done in the init.

        The convention will be that the value supplied is the radius/half-side-length of the field of view. In cases
        where the field of view is not square/circular, it should be the half-side-length of the longest side.

        A dictionary should ONLY be defined if the instruments have different field of views, and have their own
        observations in the all_obs_info table (e.g. ROSAT's instruments are mutually exclusive and cannot have
        multiple per observation).

        :return: The approximate field of view(s) for the mission's instrument(s). In cases with multiple instruments
            then this may be a dictionary, with keys being instrument names.
        :rtype: Union[Quantity, dict]
        """
        # This is defined here (as well as in the init of BaseMission) because I want people to just copy this
        #  property if they're making a new subclass, then replace None with the FoV for the mission
        self._approx_fov = None
        return self._approx_fov

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
        self._chos_insts = self.check_inst_names(new_insts)

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
            warn("Every value in the filter array is False, meaning that no observations remain.", stacklevel=2)

        # Assign the filter array to the appropriate attribute
        self._filter_allowed = new_filter_array
        # If the filter changes then we make sure download done is set to False so that any changes
        #  in observation selection are reflected in the download call
        self._download_done = False

    @property
    def filtering_operations(self) -> List[dict]:
        """
        A property getter for the filtering operations that have been applied to this mission, in the order they
        were applied. This is mainly stored so that missions that have been reinstated from a save file can be updated
        by running the exact same filtering operations again.

        :return: A list of dictionaries which have two keys, 'name', and 'arguments'; the 'name' key corresponds to
            the name of the filtering method, and the 'arguments' key corresponds to a dictionary of arguments that
            were passed to the method. 0th element was applied first, Nth element was applied last.
        :rtype: List[dict]
        """

        return self._filtering_operations

    @filtering_operations.setter
    def filtering_operations(self, new_filter_operation: dict):
        """
        A property setter for the store of filtering operations that have been applied to this mission. This is
        slightly non-traditional in that it doesn't replace the entire filtering operations attribute, but just
        appends the new entry to what is already there.

        This shouldn't really be used directly, it is more for other DAXA methods than the user.

        :param np.ndarray new_filter_operation: The entry for the filtering operations history. A dictionary that has
            two keys, 'name', and 'arguments'; the 'name' key corresponds to the name of the filtering method, and
            the 'arguments' key corresponds to a dictionary of arguments that were passed to the method
        """
        # There are quite a few checks on what is being passed to this setter, as I really don't want anyone doing
        #  it who doesn't know what they are doing - really I don't want anything but the DAXA _capture_filter
        #  decorator doing this
        # First I check that the input is a dictionary, and that the keys I need to be there are present
        if not isinstance(new_filter_operation, dict) or ('name' not in new_filter_operation or
                                                          'arguments' not in new_filter_operation):
            raise TypeError("Only a dictionary containing entries for 'name' and 'arguments' may be passed to add a "
                            "new entry to the filtering operations history.")

        # Then we ensure that the data type for the name is correct
        if not isinstance(new_filter_operation['name'], str):
            raise TypeError("The filter operation method name must be a string, this entry ({}) is "
                            "not.".format(str(new_filter_operation['name'])))
        # And that it is a method of this mission class (this isn't perfect because you could pass the name of an
        #  attribute or property, or non-filtering method, and it would be an attribute, but honestly at that point
        #  you deserve to have things break)
        elif not hasattr(self, new_filter_operation['name']):
            raise ValueError("The filter operation method name ({}) is not a method of this mission "
                             "class.".format(str(new_filter_operation['name'])))

        # Check that the entry for arguments is a dictionary
        if not isinstance(new_filter_operation['arguments'], dict):
            raise TypeError("The filter operation arguments value must be a dictionary of passed values.")

        # Finally, if we've got to this point, it is safe to append the new entry to our existing filtering operations
        #  history list
        self._filtering_operations.append(new_filter_operation)

    @property
    @abstractmethod
    def all_obs_info(self) -> pd.DataFrame:
        """
        A property getter that returns the base dataframe containing information about all the observations available
        for an instance of a mission class. This is an abstract method purely because its property setter is an
        abstract method, one cannot be without the other.

        :return: A pandas dataframe with (at minimum) the following columns; 'ra', 'dec', 'ObsID', 'science_usable',
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
    def science_usable(self) -> np.ndarray:
        """
        Property getter for the 'science_usable' column of the all observation information dataframe. This
        'science_usable' column describes whether a particular observation is usable by this module; i.e. that
        the data are suitable for scientific use (so far as can be identified by querying the storage service).
        This science_usable property is the basis for the filter array, resetting the filter array will return
        it to the values of this column.

        Data that are marked as scientifically useful but are still in a proprietary period will return True here,
        as the user may have been the one to take those data. If suitable credentials cannot be produced at download
        time however, those proprietary data will be marked as unusable.

        :return: A boolean array detailing whether an observation is scientifically useful or not.
        :rtype: np.ndarray
        """
        return self.all_obs_info['science_usable'].values

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
        Property getter that describes whether the specified data for this mission have been
        downloaded.

        :return: Boolean flag describing if data have been downloaded.
        :rtype: bool
        """
        return self._download_done

    @property
    def downloaded_type(self) -> str:
        """
        Property getter that describes what type of data was downloaded for this mission (or raises an exception if
        no download has been performed yet). The value will be either 'raw', 'preprocessed', or 'raw+preprocessed'.

        :return: A string identifier for the type of data downloaded; the value will be either 'raw',
            'preprocessed', or 'raw+preprocessed'
        :rtype: str
        """
        if not self.download_completed:
            raise DAXANotDownloadedError("The 'download_type' cannot have a valid value until a download has "
                                         "been performed.")

        return self._download_type

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
        elif new_val and not self.locked:
            self.locked = True
        self._processed = new_val

    # Then define internal methods
    def _load_state(self, save_file_path: str):
        """
        This internal function can read in a saved mission state from a file, and replicate the mission as it was. This
        can be triggered by the user passing a save file to the init of a mission, but more importantly it can be
        used by archives to re-set-up a mission with the same information as when the archive was created.

        :param str save_file_path: The path to the saved mission state json (created by the BaseMission save() method).
        """
        if not os.path.exists(save_file_path):
            raise FileNotFoundError("The specified mission save file ({}) cannot be found.".format(save_file_path))

        with open(save_file_path, 'r') as stateo:
            # This json contains all the information we need to return the mission to its saved state
            save_dict = json.load(stateo)

            # First off, lets just sanity check that the file we've been pointed too belongs to this type of mission
            if save_dict['name'] != self.name:
                raise IncompatibleSaveError("A saved state for a '{smn}' mission is not compatible with this {mn} "
                                            "mission.".format(smn=save_dict['name'], mn=self.name))

            # Set the chosen instruments property from the save file - for all mission classes
            self.chosen_instruments = save_dict['chos_inst']
            # If the chosen field wasn't a null value, we'll do the same for that - this is used only rarely, for most
            #  classes of mission this will be None
            if save_dict['chos_field'] is not None:
                self.chosen_fields = save_dict['chos_field']

            # Reset the download_type attribute - lets the mission know what type of data were downloaded last time
            self._download_type = save_dict['downloaded_type']

            # Now we need to recreate the filter array from the stored information - not actually too difficult! The
            #  interesting bit is where we let the user re-run the exact same filtering steps, to update a previously
            #  created mission state/archive
            self.filter_array = self.filter_array*self.all_obs_info['ObsID'].isin(save_dict['selected_obs'])

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

    @abstractmethod
    def _fetch_obs_info(self):
        """
        The abstract method (i.e. will be overridden in every subclass of BaseMission) that pulls basic information
        on all observations for a given mission down from whatever server it lives on.

        NOTE - THE INDEX OF THE PANDAS DATAFRAME SHOULD BE RESET AT THE END OF EACH IMPLEMENTATION OF THIS
        METHOD - e.g. obs_info_pd = obs_info_pd.reset_index(drop=True)
        """
        # self.all_obs_info = None
        pass

    # Then define user-facing methods
    def reset_filter(self):
        """
        Very simple method which simply resets the filter array, meaning that all observations THAT HAVE BEEN
        MARKED AS USABLE will now be downloaded and processed, and any filters applied to the current mission
        have been undone.
        """
        self._filter_allowed = self.all_obs_info['science_usable'].values.copy()
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

    def check_inst_names(self, insts: Union[List[str], str], error_on_bad_inst: bool = True):
        """
        A method to perform some checks on the validity of chosen instrument names for a given mission.

        :param List[str]/str insts: Instrument names that are to be checked for the current mission, either a single
            name or a list of names.
        :param bool error_on_bad_inst: Controls whether an exception is raised if the instrument(s) aren't actually
            associated with this mission - intended for DAXA checking operations (see 'get_process_logs' of Archive
            for an example). Default is True.
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

        # This list comprehension checks that the input instrument names are in the allowed instruments for this
        #  particular mission
        inst_test = [i in self._miss_poss_insts for i in updated_insts]
        # If some aren't then we throw an error (hopefully quite an informative one).
        if not all(inst_test) and error_on_bad_inst:
            bad_inst = np.array(updated_insts)[~np.array(inst_test)]
            raise ValueError("Some instruments ({bi}) are not associated with this mission, please choose from "
                             "the following; {ai}".format(bi=", ".join(bad_inst),
                                                          ai=", ".join(self._miss_poss_insts)))
        elif not all(inst_test) and not error_on_bad_inst:
            updated_insts = [i for i in updated_insts if i in self._miss_poss_insts]

        # I warn the user if the name(s) of instruments have been altered.
        if altered:
            warn("Some instrument names were converted to alternative forms expected by this module, the instrument "
                 "names are now; {}".format(', '.join(updated_insts)), stacklevel=2)

        # Return the possibly altered instruments
        return updated_insts

    @_lock_check
    @_capture_filter
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

        # Just upper-cases everything, as that is what DAXA expects in cases where there are non-numerical characters
        #  in the ObsIDs
        allowed_obs_ids = [oid.upper() for oid in allowed_obs_ids]

        # Runs the ObsID pattern checks for all the passed ObsIDs
        oid_check = [oid for oid in allowed_obs_ids if not self.check_obsid_pattern(oid)]
        if len(oid_check) != 0:
            # Raises an error if the ObsIDs don't all conform to the expected pattern defined for each mission.
            raise ValueError("One or more ObsID passed into this method does not match the expected pattern "
                             "for ObsIDs of this mission. The following are not compliant; "
                             "{}".format(', '.join(oid_check)))
        
        # Uses the Pandas isin functionality to find the rows of the overall observation table that match the input
        #  ObsIDs. This outputs a boolean array.
        sel_obs_mask = self._obs_info['ObsID'].isin(allowed_obs_ids).values

        # A check to make sure that some ObsIDs made it past the filtering
        if (self.filter_array * sel_obs_mask).sum() == 0:
            self.filter_array = np.full(self.filter_array.shape, False)
            raise NoObsAfterFilterError("ObsID search has resulted in there being no observations associated "
                                        "with this mission.")
        
        # Said boolean array can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array * sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter

    # TODO Figure out how to support survey-type missions (i.e. eROSITA) that release large sweeps of the sky
    #  when filtering based on position.
    @_lock_check
    @_capture_filter
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
        box_filter = ((self.ra_decs.ra >= lower_left.ra) & (self.ra_decs.ra <= upper_right.ra) &
                      (self.ra_decs.dec >= lower_left.dec) & (self.ra_decs.dec <= upper_right.dec))

        # Have to check whether any observations have actually been found, if not then we throw an error
        if (self.filter_array*box_filter).sum() == 0:
            self.filter_array = np.full(self.filter_array.shape, False)
            raise NoObsAfterFilterError("The box search has returned no {} observations.".format(self.pretty_name))

        # Updates the filter array
        new_filter = self.filter_array * box_filter
        self.filter_array = new_filter

    @_lock_check
    @_capture_filter
    def filter_on_positions(self, positions: Union[list, np.ndarray, SkyCoord],
                            search_distance: Union[Quantity, float, int, list, np.ndarray, dict] = None,
                            return_pos_obs_info: bool = False) -> Union[None, pd.DataFrame]:
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
        :param Quantity/float/int/list/np.ndarray/dict search_distance: The distance within which to search for
            observations by this mission. Distance may be specified either as an Astropy Quantity that can be
            converted to degrees (a float/integer will be assumed to be in units of degrees), as a dictionary of
            quantities/floats/ints where the keys are names of different instruments (possibly with different field
            of views), or as a non-scalar Quantity, list, or numpy array with one entry per set of coordinates (for
            when you wish to use different search distances for each object). The default is None, in which case a
            value of 1.2 times the approximate field of view defined for each instrument will be used; where different
            instruments have different FoVs, observation searches will be undertaken on an instrument-by-instrument
            basis using the different field of views.
        :param bool return_pos_obs_info: Allows this method to return information (in the form of a Pandas dataframe)
            which identifies the positions which have been associated with observations, and the observations they have
            been associated with. Default is False.
        :return: If return_pos_obs_info is True, then a dataframe containing information on which ObsIDs are relevant
            to which positions will be returned. If return_pos_obs_info is False, then None will be returned.
        :rtype: Union[None,pd.DataFrame]
        """

        # Checks to see if a list/array of coordinates has been passed, in which case we convert it to a
        #  SkyCoord (or a SkyCoord catalogue).
        if isinstance(positions, (list, np.ndarray)):
            positions = SkyCoord(positions, unit=u.deg, frame=self.coord_frame)
        # If the input was already a SkyCoord, we should make sure that it is in the same frame as the current
        #  mission's observation position information (honestly probably doesn't make that much of a difference, but
        #  it is good to be thorough).
        elif isinstance(positions, SkyCoord):
            positions = positions.transform_to(self.coord_frame)

        # This is slightly cheesy, but the search_around_sky method will only work if there is a catalog
        #  of positions that is being searched around, rather than a single position. As such if a single
        #  coordinate is being searched around I just duplicate it to placate the method. This won't produce
        #  any ill effects because I just care about which observations are nearby, not which coordinates are
        #  specifically matched to which observation.
        # We do also create a boolean flag to tell later checks (if necessary) that there is actually only one
        #  position
        single_pos = False
        if positions.isscalar:
            positions = SkyCoord([positions.ra, positions.ra], [positions.dec, positions.dec], unit=u.deg,
                                 frame=positions.frame)
            # This flag tells later checks that there is actually only one unique position
            single_pos = True

        # The next lot of if statements really checks that the input search distances are in the correct format
        #  etc., but here we just check to see whether the input distance is non-scalar, which means that there
        #  should one entry per coordinate.
        if search_distance is not None and not isinstance(search_distance, dict) and \
                ((type(search_distance) == Quantity and not search_distance.isscalar) or
                 isinstance(search_distance, (list, tuple)) or type(search_distance) == np.ndarray):
            # That ugly if statement is essentially checking that the search distance is not None, is not a
            #  dictionary (which allows the user to pass one search radius per instrument of the mission), and isn't
            #  just a single value. Here we wish to examine search_distance only if it is non-scalar, as it should
            #  contain one entry per coordinate.
            if single_pos:
                raise ValueError("Only a single set of coordinates has been passed, but {} search distances have been"
                                 " passed.".format(len(search_distance)))
            elif len(search_distance) != len(positions):
                raise ValueError("If a set of search distances ({sdl}) are supplied, there must be the same number as "
                                 "there are search coordinates ({pl}).".format(sdl=len(search_distance),
                                                                               pl=len(positions)))

        # If the value is left as None, the default, then we use the defined FoV for this mission and multiply by 1.2
        if search_distance is None:
            # This is read out because it can trigger a warning and I only want it to happen once
            fov = self.fov
            if isinstance(fov, Quantity):
                search_distance = (fov * 1.2).to('deg')
            # Also possible for different instruments to have different FoVs, so we have to take that into
            #  account - maybe I should just have made .fov always return a dictionary but oh well
            # If there is an instrument column that will mean that one observation has one instrument, and we can
            #  safely search using different field of views
            elif 'instrument' in self.all_obs_info:
                search_distance = {i: (v * 1.2).to('deg') for i, v in fov.items()}
            else:
                # If there is no instrument columns it means that multiple simultaneous instruments with different
                #  field of views exist - as there isn't currently an elegant way of dealing with this, I will just
                #  choose the largest field of view that is relevant to the chosen instruments
                warn("There are multiple chosen instruments {ci} for {mn} with different FoVs, but they observe "
                     "simultaneously. As such the search distance has been set to the largest FoV of the chosen"
                     " instruments.".format(ci=", ".join(self.chosen_instruments), mn=self.name), stacklevel=2)
                search_distance = max(list({i: (v * 1.2).to('deg') for i, v in fov.items()
                                            if i in self.chosen_instruments}.values()))
        # Checks to see whether a quantity has been passed, if not then the input is converted to an Astropy
        #  quantity in units of degrees. If a Quantity that cannot be converted to degrees is passed then the
        #  else part of the statement will error.
        elif not isinstance(search_distance, dict):
            # This is read out because it can trigger a warning and I only want it to happen once
            fov = self.fov
            if isinstance(fov, dict):
                warn("The mission has FoVs defined for {}, but only one search_radius has been supplied. You may "
                     "wish to pass a dictionary of search radii.".format(", ".join(list(fov.keys()))),
                     stacklevel=2)
            # Make sure the values are as they should be
            if not isinstance(search_distance, Quantity):
                search_distance = Quantity(search_distance, 'deg')
            else:
                search_distance = search_distance.to('deg')
        # If the user passes a dictionary of search radii, and the mission doesn't have multiple FoV definitions, then
        #  something has probably gone awry, and we tell them so
        elif isinstance(search_distance, dict) and not isinstance(self.fov, dict):
            raise TypeError("The definition of {}'s field-of-view indicates that it does not have multiple "
                            "instruments with different field of views, so do not pass a dictionary "
                            "of search radii.".format(self.name))
        elif isinstance(search_distance, dict) and not all([i in search_distance for i in self.chosen_instruments]):
            missing = [i for i in self.chosen_instruments if i not in search_distance]
            raise KeyError("The search_distance dictionary is missing entries for the following "
                           "instruments; {}".format(", ".join(missing)))
        elif isinstance(search_distance, dict) and not all([isinstance(v, (Quantity, int, float))
                                                            for v in search_distance.values()]):
            raise TypeError("The values in the search_distance dictionary must be either Astropy quantities, "
                            "integers, or floats.")
        elif isinstance(search_distance, dict):
            search_distance = {i: d.to('deg') if isinstance(d, Quantity) else Quantity(d, 'deg')
                               for i, d in search_distance.items()}
        else:
            raise TypeError("Please pass a Quantity, float, integer, or dictionary for search_distance.")

        # At this point the search_distance should either be a dictionary of quantities (with instrument names as
        #  keys) or a single quantity. The quantities will be in degrees.
        # In the case where we have only a single search, it is relatively simple, and rather than trying to make
        #  this method more elegant by writing one generalised approach, we're just gonna use an if statement
        # This will store all those position indices that have been identified as being associated with an observation
        pos_with_data_ind = []
        if isinstance(search_distance, Quantity) and search_distance.isscalar:
            # Runs the 'catalogue matching' between all available observations and the input positions.
            which_pos, which_obs, d2d, d3d = self.ra_decs.search_around_sky(positions, search_distance)

            # Have to check whether any observations have actually been found, if not then we throw an error
            if len(which_obs) == 0:
                raise NoObsAfterFilterError("The positional search has returned no {} "
                                            "observations.".format(self.pretty_name))

            # Sets up a filter array that consists entirely of zeros initially (i.e. it would not let
            #  any observations through).
            pos_filter = np.zeros(self.filter_array.shape)
            # The which_obs array indicates which of the entries in the table of observation info for this
            #  mission are matching to one or more of the positions passed. The list(set()) setup is used to
            #  ensure that there are no duplicates. These entries in the pos_filter are set to one, which will
            #  allow those observations through
            pos_filter[np.array(list(set(which_obs)))] = 1

            # We only bother doing this if the user actually wants the information
            if return_pos_obs_info:
                # This is the simplest case - non-scalar positions and one search distance. In this case the positions
                #  associated with ObsIDs are just one of the returns from the search_around_sky method
                pos_with_data_ind = which_pos
                # This unfortunate one-liner connects position indices with specific ObsIDs that they were matched to,
                #  and will be processed into a dataframe at the end
                which_pos_which_obs = {pos_ind: [self.obs_ids[obs_ind] for
                                                 obs_ind in which_obs[np.where(which_pos == pos_ind)[0]]]
                                       for pos_ind in np.unique(which_pos)}

        elif isinstance(search_distance, Quantity) and not search_distance.isscalar:
            # Sets up a filter array that consists entirely of zeros initially (i.e. it would not let
            #  any observations through).
            pos_filter = np.zeros(self.filter_array.shape)

            # Used to store information on which position indices are connected with which ObsIDs, if the user
            #  has requested that that information be returned
            which_pos_which_obs = {}

            # This is the reason that we have to have a separate part of the if statement for cases where the search
            #  distance is non-scalar, because of the way search_around_sky is built it can't handle non-scalar
            #  search distance values. That means we search for each position separately, updating the pos_filter
            #  as we go.
            for sd_ind, sd in enumerate(search_distance):
                rel_pos = positions[sd_ind]
                # We have to use the same trick as earlier to make search_around_sky work with a single position
                rel_pos = SkyCoord([rel_pos.ra, rel_pos.ra], [rel_pos.dec, rel_pos.dec], unit=u.deg,
                                   frame=positions.frame)

                # Runs the 'catalogue matching' between all available observations and the current input position, with
                #  the current search distance for that position.
                which_pos, which_obs, d2d, d3d = self.ra_decs.search_around_sky(rel_pos, sd)

                if len(which_obs) != 0:
                    # This works essentially identically to the if statement above, in that the filter array is just
                    #  updated to reflect which observations make it through - just here it happens on an object by
                    #  object basis
                    pos_filter[np.array(list(set(which_obs)))] = 1

                    # We only bother doing this if the user actually wants the information
                    if return_pos_obs_info:
                        # Each position is dealt with separately here, so we just append the successful position indices
                        #  to our list that keeps track of the positions which are associated with observations
                        pos_with_data_ind.append(sd_ind)
                        # Store the ObsIDs relevant to this position
                        which_pos_which_obs[sd_ind] = list(self.obs_ids[which_obs])

        else:
            # Hopefully every mission class's all_obs_info table had its indices reset at the end of the method
            #  that grabs all the information, but just in case it didn't I'll do it here, because it would royally
            #  screw things up if it weren't reset
            self.all_obs_info = self.all_obs_info.reset_index(drop=True)

            # Used to store information on which position indices are connected with which ObsIDs, if the user
            #  has requested that that information be returned
            which_pos_which_obs = {}

            # Sets up a filter array that consists entirely of zeros initially (i.e. it would not let
            #  any observations through).
            pos_filter = np.zeros(self.filter_array.shape)
            for inst in search_distance:
                cur_search_distance = search_distance[inst]

                rel_rows = self.all_obs_info[self.all_obs_info['instrument'] == inst]
                # Extract the ObsIDs for later use in constructing a dataframe of the observations that are relevant
                #  to the positions passed in by the user (if the user wants that).
                rel_obs_ids = rel_rows['ObsID'].values
                # These will be used to determine which coordinates to grab, and which entries in the pos_filter
                #  must be updated
                rel_row_inds = rel_rows.index.values
                if len(rel_rows) == 0:
                    raise KeyError("Somehow an invalid instrument name has been included in the "
                                   "search_distance dictionary.")
                # Grabs only those observation RA-Dec coordinates that are for the current instrument. Of course those
                #  coordinates are in the table (all_obs_info), but the ra_decs property has them as an Astropy
                #  SkyCoord
                rel_radecs = self.ra_decs[rel_row_inds]
                # Runs the 'catalogue matching' between all available observations and the input positions.
                which_pos, which_obs, d2d, d3d = rel_radecs.search_around_sky(positions, cur_search_distance)

                if len(which_obs) != 0:
                    # This first converts the which_obs indices back to the indices relevant to the whole set
                    #  of observations, using rel_row_inds, and then uses those values to set the pos filter. Only
                    #  if there are any selected observations though!
                    pos_filter[np.array(list(set(rel_row_inds[which_obs])))] = 1
                    # In this case we are likely to be iterating through different search distances, so we'll append
                    #  each 'which_pos' to our list and sort it out at the end to find the unique indices that
                    #  describe which positions are associated with data.
                    pos_with_data_ind.append(which_pos)

                    # This deeply unfortunate one-liner connects position indices with specific ObsIDs that they
                    #  were matched to, and will be processed into a dataframe at the end - this has to account for
                    #  the possibility that there may already be a pos_ind entry in the dictionary whose information
                    #  we don't want to remove - definitely should have been a for loop for readability but oh well
                    to_add = {pos_ind: [rel_obs_ids[obs_ind] for obs_ind in
                                        which_obs[np.where(which_pos == pos_ind)[0]]]
                    if pos_ind not in which_pos_which_obs
                    else which_pos_which_obs[pos_ind] + [rel_obs_ids[obs_ind]
                                                         for obs_ind in which_obs[np.where(which_pos == pos_ind)[0]]]
                              for pos_ind in np.unique(which_pos)}
                    which_pos_which_obs.update(to_add)

            # Have to check whether any observations have actually been found, if not then we throw an error. Very
            #  similar to a check in the first part of the if statement, but here we only check at the end of the
            #  for loops, because it is fine if some of the instruments don't have any observations selected at the
            #  end, we only have to worry if NONE of them have observations selected
            if pos_filter.sum() == 0:
                raise NoObsAfterFilterError("The positional search has returned no {} "
                                            "observations.".format(self.pretty_name))

        # This makes sure that, particularly in the case where each instrument has a different field of view, we
        #  combine the pos_with_dat_ind list into a single, 1D, array
        if len(pos_with_data_ind) != 0:
            pos_with_data_ind = np.unique(np.hstack(pos_with_data_ind))
        else:
            pos_with_data_ind = np.array([])
        # If we were passed just one position, we did a little cheesy thing to make sure the searches always worked
        #  the same, so we have to account for the fact that the position is in the pos_with_data_ind array twice
        if single_pos and len(pos_with_data_ind) != 0:
            pos_with_data_ind = np.array([pos_with_data_ind[0]])

        # Convert the array of ones and zeros to boolean, which is what the filter_array property setter wants
        pos_filter = pos_filter.astype(bool)

        # Create the combination of the existing filter array and the new position filter
        new_filter = self.filter_array * pos_filter

        # And update the filter array
        self.filter_array = new_filter

        # And we only return the position indices with data if the user asked for it
        if return_pos_obs_info:
            pos_with_data = positions[pos_with_data_ind]
            rel_obs_ids = np.array([",".join(which_pos_which_obs[pos_ind]) for pos_ind in pos_with_data_ind])
            ret_df_cols = ['pos_ind', 'pos_ra', 'pos_dec', 'ObsIDs']
            ret_df_data = np.vstack([pos_with_data_ind, pos_with_data.ra.value, pos_with_data.dec.value,
                                     rel_obs_ids]).T

            return pd.DataFrame(ret_df_data, columns=ret_df_cols)

    @_lock_check
    @_capture_filter
    def filter_on_name(self, object_name: Union[str, List[str]],
                       search_distance: Union[Quantity, float, int, list, np.ndarray, dict] = None,
                       parse_name: bool = False):
        """
        This method wraps the 'filter_on_positions' method, and allows you to filter the mission's observations so
        that it contains data on a single (or a list of) specific objects. The names are passed by the user, and
        then parsed into coordinates using the Sesame resolver. Those coordinates and the search distance are
        then used to find observations that might be relevant.

        :param str/List[str] object_name: The name(s) of objects you would like to search for.
        :param Quantity/float/int/list/np.ndarray/dict search_distance: The distance within which to search for
            observations by this mission. Distance may be specified either as an Astropy Quantity that can be
            converted to degrees (a float/integer will be assumed to be in units of degrees), as a dictionary of
            quantities/floats/ints where the keys are names of different instruments (possibly with different field
            of views), or as a non-scalar Quantity, list, or numpy array with one entry per set of coordinates (for
            when you wish to use different search distances for each object). The default is None, in which case a
            value of 1.2 times the approximate field of view defined for each instrument will be used; where different
            instruments have different FoVs, observation searches will be undertaken on an instrument-by-instrument
            basis using the different field of views.
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
            raise NameResolveError("The name(s) could not be resolved into coordinates.")

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
    @_capture_filter
    def filter_on_time(self, start_datetime: datetime, end_datetime: datetime, over_run: bool = True):
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
            then only observations with a start AND end within the window are selected. Default is True.
        """
        # This just selects the exact behaviour of whether an observation is allowed through the filter or not.
        if not over_run:
            time_filter = ((self.all_obs_info['start'] >= start_datetime) &
                           (self.all_obs_info['end'] <= end_datetime)).values
        else:
            time_filter = (((self.all_obs_info['start'] >= start_datetime) &
                            (self.all_obs_info['start'] <= end_datetime)) |
                           ((self.all_obs_info['end'] >= start_datetime) &
                            (self.all_obs_info['end'] <= end_datetime)) |
                           ((self.all_obs_info['start'] <= start_datetime) &
                            (self.all_obs_info['end'] >= end_datetime))).values

        # Have to check whether any observations have actually been found, if not then we throw an error
        if (self.filter_array * time_filter).sum() == 0:
            self.filter_array = np.full(self.filter_array.shape, False)
            raise NoObsAfterFilterError("The temporal search has returned no {} "
                                        "observations.".format(self.pretty_name))

        # Combines the time filter with the existing filter and updates the property.
        new_filter = self.filter_array * time_filter
        self.filter_array = new_filter

    @_lock_check
    @_capture_filter
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

    @_lock_check
    @_capture_filter
    def filter_on_positions_at_time(self, positions: Union[list, np.ndarray, SkyCoord],
                                    start_datetimes: Union[np.ndarray, datetime],
                                    end_datetimes: Union[np.ndarray, datetime],
                                    search_distance: Union[Quantity, float, int, list, np.ndarray, dict] = None,
                                    return_obs_info: bool = False, over_run: bool = True):
        """

        This method allows you to filter the observations available for a mission based on a set of coordinates for
        which you wish to locate observations that were taken within a certain time frame. The method spatially
        searches for observations that have central coordinates within the distance set by the search_distance
        argument, and temporally by start and end times passed by the user; and observations that fall within that
        window are allowed through the filter.

        The exact behaviour of the temporal filtering method is controlled by the over_run argument, if set
        to True then observations with a start or end within the search window will be selected, but if False
        then only observations with a start AND end within the window are selected.

        Please be aware that filtering methods are cumulative, so running another method will not remove the
        filtering that has already been applied, you can use the reset_filter method for that.

        :param list/np.ndarray/SkyCoord positions: The positions for which you wish to search for observations. They
            can be passed either as a list or nested list (i.e. [r, d] OR [[r1, d1], [r2, d2]]), a numpy array, or
            an already defined SkyCoord. If a list or array is passed then the coordinates are assumed to be in
            degrees, and the default mission frame will be used.
        :param np.array(datetime)/datetime start_datetimes: The beginnings of time windows in which to search for
            observations. There should be one entry per position passed.
        :param np.array(datetime)/datetime end_datetimes: The endings of time windows in which to search for
            observations. There should be one entry per position passed.
        :param Quantity/float/int/list/np.ndarray/dict search_distance: The distance within which to search for
            observations by this mission. Distance may be specified either as an Astropy Quantity that can be
            converted to degrees (a float/integer will be assumed to be in units of degrees), as a dictionary of
            quantities/floats/ints where the keys are names of different instruments (possibly with different field
            of views), or as a non-scalar Quantity, list, or numpy array with one entry per set of coordinates (for
            when you wish to use different search distances for each object). The default is None, in which case a
            value of 1.2 times the approximate field of view defined for each instrument will be used; where different
            instruments have different FoVs, observation searches will be undertaken on an instrument-by-instrument
            basis using the different field of views.
        :param bool return_obs_info: Allows this method to return information (in the form of a Pandas dataframe)
            which identifies the positions which have been associated with observations, in the specified time
            frame, and the observations they have been associated with. Default is False.
        :param bool over_run: This controls whether selected observations have to be entirely within the passed
            time window or whether either a start or end time can be within the search window. If set
            to True then observations with a start or end within the search window will be selected, but if False
            then only observations with a start AND end within the window are selected. Default is True.
        """
        # Check that the start and end information is in the same style
        if isinstance(start_datetimes, datetime) != isinstance(end_datetimes, datetime):
            raise TypeError("The 'start_datetimes' and 'start_datetimes' must either both be individual datetimes, or "
                            "arrays of datetimes (for multiple positions).")
        # Need to make sure we make the datetimes iterable - even if there is only one position/time period being
        #  investigated
        elif isinstance(start_datetimes, datetime):
            start_datetimes = np.array([start_datetimes])
            end_datetimes = np.array([end_datetimes])

        # This should make sure that any lists of positions like [r, d] are turned into [[r, d]] - this should
        #  be more acceptable to downstream things
        if isinstance(positions, list) and not isinstance(positions[0], (list, SkyCoord)):
            positions = [positions]

        # We initially check that the arguments we will be basing the time filtering on are of the right length,
        #  i.e. every position must have corresponding start and end times
        if not positions.isscalar and (len(start_datetimes) != len(positions) or len(end_datetimes) != len(positions)):
            raise ValueError("The 'start_datetimes' (len={sd}) and 'end_datetimes' (len={ed}) arguments must have one "
                             "entry per position specified by the 'positions' (len={p}) "
                             "arguments.".format(sd=len(start_datetimes), ed=len(end_datetimes), p=len(positions)))
        elif positions.isscalar and (len(start_datetimes) != 1 or len(end_datetimes) != 1):
            raise ValueError("The 'start_datetimes' (len={sd}) and 'end_datetimes' (len={ed}) arguments must be "
                             "scalar if a single position is passed".format(sd=len(start_datetimes),
                                                                            ed=len(end_datetimes)))

        # Now we can use the filter on positions method to search for any observations that might be applicable to
        #  the search that the user wants to perform - we will also return the dataframe that
        rel_obs_info = self.filter_on_positions(positions, search_distance, True)
        # We save a copy of the filter as it was after the positional filtering - we'll need it later as we're going
        #  to be messing around with the filter array a bit
        after_pos_filt = self.filter_array.copy()

        # This array will build up into something that we will construct the final filter array from as we iterate
        #  through all the positions that have some data
        cumu_filt = np.zeros(len(self._obs_info))
        # This is a separate filtering array that will allow us to cut the 'rel_obs_info' dataframe down to only
        #  those entries that have relevant temporal and spatial data
        any_rel_data = np.full(len(rel_obs_info), False)
        # We essentially iterate through each of the user supplied positions which have some sort of observations
        #  that are SPATIALLY relevant - now we have to determine if any of those observations fit our temporal
        #  criteria
        for rel_df_ind, pos_ind in enumerate(rel_obs_info['pos_ind'].values):
            # Retrieve the relevant row in the dataframe we asked to be returned from the filter_on_positions method
            rel_row = rel_obs_info[rel_obs_info['pos_ind'] == pos_ind].iloc[0]
            # Turn the joined string of ObsIDs back into a list of ObsIDs
            rel_obs_ids = rel_row['ObsIDs'].split(',')

            # Just make sure that 'pos_ind' is an integer at this point, as we want to address some arrays with it
            pos_ind = int(pos_ind)
            # Get the start and end time that the user specified for the current position, we shall need them to
            #  do the time filtering
            start_time = start_datetimes[pos_ind]
            end_time = end_datetimes[pos_ind]

            # Set up a temporary filter that only includes those ObsIDs that are relevant to the current position
            #  that we are considering
            temp_filt = self._obs_info['ObsID'].isin(rel_obs_ids).values

            # It is possible that all the ObsIDs selected are not science usable, so we do just check the
            #  sum of the array we're going to be assigning to the 'filter_array' property
            if (after_pos_filt*temp_filt).sum() == 0:
                continue
            # Then make sure we assign that array to the actual current filter (this is why we made a copy of it
            #  earlier, so we can reset it after we modified it in each iteration).
            self.filter_array = after_pos_filt*temp_filt

            # Then we try the filter_on_time method, which will now only be searching the observations that are
            #  relevant to the current position - if something is found then no exception will be thrown
            try:
                self.filter_on_time(start_time, end_time, over_run)
                # If we get this far then there are matching data - so we add the current filter (which has been
                #  modified by the filter_on_time method) to the cumulative filter
                cumu_filt += self._filter_allowed
                rel_obs_info.loc[rel_df_ind, 'ObsIDs'] = ",".join(self.filtered_obs_info['ObsID'].values)
                any_rel_data[rel_df_ind] = True
            except NoObsAfterFilterError:
                pass

        # As we were adding the time filters (when they were successful) to what was originally a big array of zeros,
        #  this array is clearly not yet in the format we want for the filter array - hence we just check for anywhere
        #  the value is greater than zero - these will be set to True and False, which we want for the filter array
        cumu_filt = cumu_filt > 0

        # Have to check whether any observations have actually been found, if not then we throw an error
        if cumu_filt.sum() == 0:
            self.filter_array = cumu_filt
            raise NoObsAfterFilterError("The spatio-temporal search has returned no {} "
                                        "observations.".format(self.pretty_name))

        self.filter_array = after_pos_filt * cumu_filt

        # If the user wants a summary dataframe at the end, then we return one which is cut down to only those entries
        #  that represent positions with both temporal and spatial matches
        if return_obs_info:
            return rel_obs_info[any_rel_data]

    def save(self, save_root_path: str):
        """
        A method to save a file representation of the current state of a DAXA mission object. This may be used by
        the user, and can be safely sent to another user or system to recreate a mission. It is also used by the
        archive saving mechanic, so that mission objects can be re-set up - it is worth noting that the archive save
        files ARE NOT how to make a portable archive,

        :param str save_root_path: The DIRECTORY where you wish a save file to be stored, DO NOT pass a path
            with a filename at the end, as this method will create its own filename.
        """

        # We check to see whether the output root path exists, and if it doesn't then we shall create it
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)

        # We set up the actual name of the same file, then the full path to it
        file_name = self.name + '_state.json'
        miss_file_path = os.path.join(save_root_path, file_name)

        # This is where we set up the dictionary of information that will actually be saved - all the information
        #  common to all mission classes at least. Some will be None for most missions (like chosen field)
        mission_data = {'name': self.name, 'chos_inst': self.chosen_instruments, 'chos_field': self._chos_fields,
                        'downloaded_type': self._download_type, 'cur_date': str(datetime.today())}

        # The currently selected data need some more specialist treatment - we can't just save the filter
        #  array, because the available observations (and thus the information table that the filter gets applied
        #  too) are not necessarily static (for some they will be, because the missions are finished).
        # As such, we decided to just save the accepted ObsIDs, and any difference in data available can be inferred
        #  by re-running the stored filtering steps, rather than comparing a stored list of ObsIDs to a newly
        #  downloaded one
        sel_obs = self.filtered_obs_ids

        # Make sure to add the sel_obs dictionary into the overall one we're hoping to store
        mission_data['selected_obs'] = list(sel_obs)

        # TODO Need to store the applied filtering options, and the order

        # Now we write the required information to the state file path
        with open(miss_file_path, 'w') as stateo:
            json_str = json.dumps(mission_data, indent=4)
            stateo.write(json_str)

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

    @abstractmethod
    def ident_to_obsid(self, ident: dict):
        """
        A slightly unusual abstract method which will allow each mission convert a unique identifier being used
        in the processing steps to the ObsID (as these unique identifiers will contain the ObsID). This is necessary
        because XMM, for instance, has processing steps that act on whole ObsIDs (e.g. cifbuild), and processing steps
        that act on individual sub-exposures of instruments of ObsIDs, so the ID could be '0201903501M1S001'.

        Implemented as an abstract method because the unique identifier style may well be different for different
        missions - many will just always be the ObsID, but we want to be able to have low level control.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param str ident: The unique identifier used in a particular processing step.
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
