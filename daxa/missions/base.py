#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/11/2022, 14:36. Copyright (c) The Contributors
import os.path
import re
from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, BaseRADecFrame
from astropy.units import Quantity

REQUIRED_COLS = ['ra', 'dec', 'ObsID', 'usable_science', 'start', 'duration']


class BaseMission(metaclass=ABCMeta):
    """

    """
    def __init__(self, output_archive_name: str, connection_url: str = None):
        # TODO Perhaps remove the connection URL, we should probably try to go through astroquery as much
        #  as possible
        self._miss_name = None
        self._miss_coord_frame = None
        self._required_mission_specific_cols = []
        self._miss_poss_insts = []
        self._id_format = None
        self._obs_info = None

        self._access_url = connection_url

        self._archive_name = output_archive_name
        # self._archive_name_version =

        self._top_level_output_path = None

        self._filter_allowed = None

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
        return self._miss_name

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
            raise ValueError("Every value in the filter array is False, meaning that no observations remain. As "
                             "such the new filter array has not been accepted")
        else:
            self._filter_allowed = new_filter_array

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
    def filtered_obs_info(self) -> pd.DataFrame:
        """
        A property getter that applies the current filter array to the dataframe of observation information, and
        returns filtered dataframe containing all columns available for this mission.

        :return: A filtered dataframe of observation information.
        :rtype: pd.DataFrame
        """
        return self._obs_info[self.filter_array]

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
                             "columns; {}".format(', '.join(REQUIRED_COLS+self._required_mission_specific_cols)))

    # Then define user-facing methods
    @abstractmethod
    def fetch_obs_info(self):
        """
        The abstract method (i.e. will be overridden in every subclass of BaseMission) that pulls basic information
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
        new_filter = self.filter_array*sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter

    # TODO Figure out how to support survey-type missions (i.e. eROSITA) that release large sweeps of the sky
    #  when filtering based on position.
    def filter_on_rect_region(self, lower_left: Union[SkyCoord, np.ndarray, list],
                              upper_right: Union[SkyCoord, np.ndarray, list]):
        """
        A method that filters observations based on whether their CENTRAL COORDINATE falls within a rectangular
        region defined using coordinates of the bottom left and top right corners. Observations are kept if they
        fall within the region.

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
        # Updates the filter array
        new_filter = self.filter_array*box_filter
        self.filter_array = new_filter

    def filter_on_positions(self, positions: Union[list, np.ndarray, SkyCoord],
                            search_distance: Union[Quantity, float, int]):
        """
        This method allows you to filter the observations available for a mission based on a set of coordinates for
        which you wish to locate observations. The method searches for observations by the current mission that have
        central coordinates within the distance set by the search_distance argument.

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
        new_filter = self.filter_array*pos_filter
        # And update the filter array
        self.filter_array = new_filter

    def filter_on_time(self):
        pass








