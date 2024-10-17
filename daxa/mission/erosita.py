#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 04/09/2024, 12:38. Copyright (c) The Contributors

import gzip
import os
import re
import shutil
import tarfile
from copy import deepcopy
from multiprocessing import Pool
from shutil import copyfileobj
from typing import List, Union, Any
from warnings import warn

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.units import Quantity
from bs4 import BeautifulSoup
from tqdm import tqdm

from .base import BaseMission
from .base import _lock_check
from .. import NUM_CORES
from ..config import EROSITA_CALPV_INFO, ERASS_DE_DR1_INFO
from ..exceptions import DAXADownloadError

# So these are the style of directory names that we need - as I'm writing this the directory names are actually
#  EXP_010 and DET_010, but the numbers there refer to the pipeline that they were processed with, so I don't know
#  if that will stay forever - I am going to try and make it resilient to possible changes and let the user choose
#  the pipeline version when downloading
REQUIRED_DIRS = {'erosita_all_sky_de_dr1': {'all': ['EXP'],
                                            'products': ['EXP', 'DET']}}

# TODO Make sure the properties, internal methods, and user-facing methods are in the 'right' order for this project.

class eROSITACalPV(BaseMission):
    """
    The mission class for the eROSITA early data release observations made during the Calibration and Performance 
    Verification program. 

    :param List[str]/str fields: The eROSITA calibration field name(s) or type to download/process data from. 
    :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """
    def __init__(self, insts: Union[List[str], str] = None, fields: Union[List[str], str] = None,
                 save_file_path: str = None):
        """
        The mission class for the eROSITA early data release observations made during the Calibration and Performance 
        Verification program. 

        :param List[str]/str fields: The eROSITA calibration field name(s) or type to download/process data from.
        :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        # Call the init of parent class with the required information
        super().__init__() 

        # All the allowed names of fields 
        self._miss_poss_fields = EROSITA_CALPV_INFO["Field_Name"].tolist()
        # All the allowed types of field, i.e. survey, magellanic cloud, galactic field, extragalactic field
        self._miss_poss_field_types = EROSITA_CALPV_INFO["Field_Type"].unique().tolist()

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['Field_Name', 'Field_Type', 'active_insts']
        
        # Runs the method which fetches information on all available eROSITACalPV observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()

        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

        # Sets the default fields
        if fields is None:
            self.chosen_fields = self._miss_poss_fields
        else:
            # Using the property setter because it calls the internal _check_chos_fields function
            #  which deals with the fields being given as a name or field type
            self.chosen_fields = fields
            # Applying filters on obs_ids so that only obs_ids associated with the chosen fields are included
            self.filter_on_fields(fields)

        # Sets the default instruments
        if insts is None:
            insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        
        # Setting all the possible instruments that can be associated with eROSITA data
        self._miss_poss_insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        # Setting the user specified instruments
        self.chosen_instruments = insts

        # The event list name for filling in
        self._template_evt_name = "*m00_{oi}_020_EventList_c001.fits"

        # Call the name property to set up the name and pretty name attributes
        self.name


        # We now will read in the previous state, if there is one to be read in.
        if save_file_path is not None:
            self._load_state(save_file_path)

    # Defining properties first
    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission.

        :return: The mission name
        :rtype: str
        """
        # This is defined here (as well as in the init of BaseMission) because I want people to just copy this
        #  property if they're making a new subclass, then replace None with the name of the mission.
        self._miss_name = "erosita_calpv"
        # Used for things like progress bar descriptions
        self._pretty_miss_name = "eROSITACalPV"
        return self._miss_name
    
    @property
    def chosen_instruments(self) -> List[str]:
        """
        Property getter for the names of the currently selected instruments associated with this mission which
        will be processed into an archive by DAXA functions. Overwritten here because there
        are custom behaviours for eROSITA.

        :return: A list of instrument names
        :rtype: List[str]
        """
        return self._chos_insts

    @chosen_instruments.setter
    @_lock_check
    def chosen_instruments(self, new_insts: List[str]):
        """
        Property setter for the instruments associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value. Overwritten here because there
        are custom behaviours for eROSITA.

        :param List[str] new_insts: The new list of instruments associated with this mission which should
            be processed into the archive.
        """
        new_insts = self.check_inst_names(new_insts)

        if (len(self._chos_insts) == 0 or np.array(deepcopy(new_insts)).sort() !=
                np.array(deepcopy(self._chos_insts)).sort()):
            changed = True
            self._chos_insts = new_insts
        else:
            changed = False

        # Checking if the data has already been downloaded:
        if self._download_done and changed:
            # TODO I don't think this is a safe way of doing this - not sure we should be doing it at all
            # Only doing the instrument filtering if not all the instruments have been chosen
            if len(new_insts) != 7:
                # Getting all the path for each eventlist corresponding to an obs_id for the
                #  _inst_filtering function later
                fits_paths = [self.get_evt_list_path(o) for o in self.filtered_obs_ids]

                # Filtering out any events from the raw data that aren't from the selected instruments
                if NUM_CORES == 1:
                    with tqdm(total=len(self), desc="Selecting EventLists from "
                                                    "{}".format(new_insts)) as inst_filter_prog:
                        for path in fits_paths:
                            self._inst_filtering(insts=new_insts, evlist_path=path)
                            # Update the progress bar
                            inst_filter_prog.update(1)

                elif NUM_CORES > 1:
                    # List to store any errors raised during download tasks
                    raised_errors = []

                    # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(new_insts)) \
                            as inst_filter_prog, Pool(NUM_CORES) as pool:

                        # The callback function is what is called on the successful completion of a _download_call
                        def callback(download_conf: Any):
                            """
                            Callback function for the apply_async pool method, gets called when a download task
                            finishes without error.

                            :param Any download_conf: The Null value confirming the operation is over.
                            """
                            nonlocal inst_filter_prog  # The progress bar will need updating
                            inst_filter_prog.update(1)

                        # The error callback function is what happens when an exception is thrown
                        #  during a _download_call
                        def err_callback(err):
                            """
                            The callback function for errors that occur inside a download task running in the pool.

                            :param err: An error that occurred inside a task.
                            """
                            nonlocal raised_errors
                            nonlocal inst_filter_prog

                            if err is not None:
                                # Rather than throwing an error straight away I append them all to a list for later.
                                raised_errors.append(err)
                            inst_filter_prog.update(1)

                        # Again nested for loop through each Obs_ID
                        for path in fits_paths:
                            # Add each download task to the pool
                            pool.apply_async(self._inst_filtering, kwds={'insts': new_insts, 'evlist_path': path},
                                             error_callback=err_callback, callback=callback)
                        pool.close()  # No more tasks can be added to the pool
                        pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                    # Raise all the download errors at once, if there are any
                    if len(raised_errors) != 0:
                        raise DAXADownloadError(str(raised_errors))

                else:
                    raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission.

        :return: The coordinate frame of the RA-Dec.
        :rtype: BaseRADecFrame
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_coord_frame = FK5
        return self._miss_coord_frame
        
    @property
    def id_regex(self) -> str:
        """
        Property getter for the regular expression (regex) pattern for observation IDs of this mission.

        :return: The regex pattern for observation IDs.
        :rtype: str
        """
        # The ObsID regular expression is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._id_format = '^[0-9]{6}$'
        return self._id_format

    @property
    def fov(self) -> Union[Quantity, dict]:
        """
        Property getter for the approximate field of view set for this mission. This is the radius/half-side-length of
        the field of view. In cases where the field of view is not square/circular, it is the half-side-length of
        the longest side.

        NOTE - THIS FIELD OF VIEW IS SORT OF NONSENSE BECAUSE OF HOW SOME OF THE eROSITACalPV DATA WERE TAKEN
        IN POINTING AND SOME IN SLEWING MODE.

        :return: The approximate field of view(s) for the mission's instrument(s). In cases with multiple instruments
            then this may be a dictionary, with keys being instrument names.
        :rtype: Union[Quantity, dict]
        """
        # The approximate field of view is defined here because I want to force implementation for each
        #  new mission class.
        warn("A field-of-view cannot be easily defined for eROSITACalPV and this number is the approximate half-length "
             "of an eFEDS section, the worst case separation - this is unnecessarily large for pointed "
             "observations, and you should make your own judgement on a search distance.", stacklevel=2)
        self._approx_fov = Quantity(4.5, 'degree')
        return self._approx_fov

    @property
    def all_obs_info(self) -> pd.DataFrame:
        """
        A property getter that returns the base dataframe containing information about all the observations available
        for an instance of a mission class.

        :return: A pandas dataframe with (at minimum) the following columns; 'ra', 'dec', 'ObsID', 'science_usable',
            'start', 'duration'
        :rtype: pd.DataFrame
        """
        return self._obs_info

    @all_obs_info.setter
    def all_obs_info(self, new_info: pd.DataFrame):
        """
        Property setter that allows the setting of a new all-observation-information dataframe. This is the dataframe
        that contains information on every possible observation for a mission.

        :param pd.DataFrame new_info: The new dataframe to update the all observation information.
        """
        self._obs_info_checks(new_info)
        self._obs_info = new_info

    @property
    def all_mission_fields(self) -> List[str]:
        """
        Property getter for the names of all possible fields associated with this mission.

        :return: A list of field names.
        :rtype: List[str]
        """
        # _miss_poss_fields returns all the field_name column of EROSITA_CALPV_INFO
        # so set() is used to remove duplicate field names where obs_ids have the same field name
        return list(set(self._miss_poss_fields))
    
    @property
    def all_mission_field_types(self) -> List[str]:
        """
        Property getter for the names of all possible field types associated with this mission.

        :return: A list of field types.
        :rtype: List[str]
        """
        return self._miss_poss_field_types
    
    @property
    def chosen_fields(self) -> List[str]:
        """
        Property getter for the names of the currently selected fields associated with this mission which
        will be processed into an archive by DAXA functions.

        :return: A list of field names
        :rtype: List[str]
        """
        return self._chos_fields

    @chosen_fields.setter
    @_lock_check
    def chosen_fields(self, new_fields: List[str]):
        """
        Property setter for the fields associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value.

        :param List[str] new_fields: The new list of fields or field types associated with this mission which should
            be processed into the archive.
        """
        self._chos_fields = self._check_chos_fields(new_fields)

    # Then define user-facing methods
    def _fetch_obs_info(self):
        """
        This method uses the hard coded csv file to pull information on all eROSITACalPV observations.
        The data are processed into a Pandas dataframe and stored.
        """
        # Hard coded this information and saved it to the erosita_calpv_info.csv file in /files
        # Making a copy so that EROSITA_CALPV_INFO remains unchanged
        calpv_copy = EROSITA_CALPV_INFO.copy()

        # Need to split the times since they go to milisecond precision,
        #  which is a pain to translate to a datetime object, and is superfluous information anyway
        calpv_copy['start'] = [str(time).split('.', 1)[0] for time in calpv_copy['start']]
        calpv_copy['end'] = [str(time).split('.', 1)[0] for time in calpv_copy['end']]
        calpv_copy['start'] = pd.to_datetime(calpv_copy['start'], utc=False, format="%Y-%m-%dT%H:%M:%S",
                                             errors='coerce')
        calpv_copy['end'] = pd.to_datetime(calpv_copy['end'], utc=False, format="%Y-%m-%dT%H:%M:%S", errors='coerce')

        # Including the relevant information for the final all_obs_info DataFrame
        obs_info_pd = calpv_copy[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration', 'Field_Name',
                                  'Field_Type', 'active_insts']]

        self.all_obs_info = obs_info_pd

    def _check_chos_fields(self, fields: Union[List[str], str]):
        """
        An internal function to perform some checks on the validity of chosen field or field type for a given mission.
        If a field type is given, such as galactic fields, or magellanic clouds, this function will return a list of
        all the fields of this type.

        :param List[str]/str insts:
        :return: The list of fields.
        :rtype: List
        """
        # Just makes sure we can iterate across field(s), regardless of how many there are
        if not isinstance(fields, list):
            fields = [fields]

        # Making sure all the items in the list are strings
        if not all([isinstance(field, str) for field in fields]):
            raise ValueError("The fields input must be entered as a string, or a list of strings.")

        # Converting to upper case and replacing special characters and whitespaces
        #  with underscores to match the entries in EROSITA_CALPV_INFO
        fields = [re.sub("[-()+/. ]", "_", field.upper()) for field in fields]

        # In case people use roman numerals or don't include the brackets in their input
        # Lovely and hard coded but not sure if there is any better way to do this
        poss_alt_field_names = {"IGR_J13020_6359": "IGR_J13020_6359__2RXP_J130159_635806_",
                                "HR_3165": "HR_3165__ZET_PUP_", "CRAB_I": "CRAB_1", "CRAB_II": "CRAB_2",
                                "CRAB_III": "CRAB_3", "CRAB_IV": "CRAB_4",
                                "47_TUC": "47_TUC__NGC_04_", "TGUH2213P1": "TGUH2213P1__DARK_CLOUD_",
                                "A3391": "A3391_A3395", "A3395": "A3391_A3395"}

        # Finding if any of the fields entries are not valid CalPV field names or types
        bad_fields = [f for f in fields if f not in poss_alt_field_names and f not in self._miss_poss_fields
                      and f not in self._miss_poss_field_types and f != 'CRAB']
        if len(bad_fields) != 0:
            raise ValueError("Some field names or field types: {bf} are not associated with this mission, please "
                             "choose from the following fields; {gf} or field types; "
                             "{gft}".format(bf=",".join(bad_fields),
                                            gf=",".join(list(set(self._miss_poss_fields))),
                                            gft=",".join(self.all_mission_field_types)))
        

        # Extracting the alt_fields from fields
        alt_fields = [field for field in fields if field in poss_alt_field_names]
        # Making a list of the alt_fields DAXA compatible name
        alt_fields_proper_name = [poss_alt_field_names[field] for field in alt_fields]
        # Seeing if someone just input 'crab' into the fields argument
        if 'CRAB' in fields:
            crab = ['CRAB_1', 'CRAB_2', 'CRAB_3', 'CRAB_4']
        else:
            crab = []
        # Then the extracting the field_types
        field_types = [field for field in fields if field in self._miss_poss_field_types]
        # Turning the field_types into field_names
        field_types_proper_name = EROSITA_CALPV_INFO.loc[EROSITA_CALPV_INFO["Field_Type"].isin(field_types), "Field_Name"].tolist()
        # Then extracting the field names from fields
        field_names = [field for field in fields if field in self._miss_poss_fields]

        # Adding all these together to make the final fields list
        updated_fields = alt_fields_proper_name + crab + field_types_proper_name + field_names
        # Removing the duplicates from updated_fields
        updated_fields = list(set(updated_fields))

        # Return the chosen fields
        return updated_fields
    
    # Then define user-facing methods
    @staticmethod
    def _inst_filtering(insts: List[str], evlist_path: str):
        """
        Method to filter event lists for eROSITACalPV data based on instrument choice.

        :param List[str] insts: The self.chosen_instruments attribute.
        :param str evlist_path: This is the file path to the raw event list for a certain ObsID
            that has NOT been filtered for the users instrument choice yet.
        """

        # Getting a string of TM numbers to add to the end of the file name
        insts_str = ''.join(sorted(re.findall(r'\d+', ''.join(insts))))

        # Checking that this combination of instruments has not been filtered for before for this Obsid
        #  this is done by checking that there is no file with the _if_{}.fits ending where {} is the
        # number(s) of the TM(s) that the user has specified when declaring the mission.
        # Indexing the string at [:-5] removes the .fits part of the file path
        if os.path.exists(evlist_path[:-5] + '_if_{}.fits'.format(insts_str)):
            pass
        else:
            # Reading in the file
            with fits.open(evlist_path) as fits_file:
                # Selecting the telescope module number column
                data = fits_file[1].data
                t_col = data["TM_NR"]

                # Putting inst names into correct format to search in t_col for
                gd_insts = [int(re.sub('[^0-9]','', tscope)) for tscope in insts]

                # Getting the indexes of events with the chosen insts
                gd_insts_indx = np.where(np.isin(t_col, gd_insts))[0]

                # Filtering the data on those telescopes
                filtered_data = data[gd_insts_indx]

                # Replacing unfiltered event list in the fits file with the new ones
                fits_file[1].data = filtered_data

                # Writing this to a new file (the if is for instrument filtered)
                fits_file.writeto(evlist_path[:-5] + '_if_{}.fits'.format(insts_str))

    @staticmethod
    def _download_call(raw_dir: str, link: str):
        """
        This internal static method is purely to enable parallelised downloads of data, as defining
        an internal function within download causes issues with pickling for multiprocessing.

        :param str raw_dir: The raw data directory in which to create an ObsID directory and store
            the downloaded data.
        :param str link: The download_link of the particular field to be downloaded.
        """
        # Since you can't download a single observation for a field, you have to download them all in one tar file,
        #  I am making a temporary directories to download the tar file and unpack it in, then move the observations
        #  to their own directory afterwards in the _directory_formatting function

        # Getting the field name associated with the download link for directory naming purposes
        field_name = EROSITA_CALPV_INFO.loc[EROSITA_CALPV_INFO['download'].isin([link]), 'Field_Name'].tolist()[0]
        # The temporary
        temp_dir = os.path.join(raw_dir, "temp_download", field_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Download the requested data
        with requests.get(link, stream=True) as r:
            field_dir = os.path.join(temp_dir, field_name)
            os.makedirs(field_dir)
            with open(field_dir + "{f}.tar.gz".format(f=field_name), "wb") as writo:
                copyfileobj(r.raw, writo)

        # Unzipping the tar file
        tar_name = field_dir + "{f}.tar.gz".format(f=field_name)
        with tarfile.open(tar_name, "r:gz") as tar:
            tar.extractall(field_dir)
            os.remove(tar_name)

        return None

    def _directory_formatting(self):
        """
        Internal method to rearrange the downloaded files from field names into the Obs_ID top layer
        directory structure for consistency with other missions in DAXA. To be called after the initial
        download of the fields has been completed.
        """

        # Moving the eventlist for each obs_id from its downloaded path to the path DAXA expects
        for obs_id in self.filtered_obs_ids:
            # The field the obs_id was downloaded with
            field_name = EROSITA_CALPV_INFO["Field_Name"].loc[EROSITA_CALPV_INFO["ObsID"] == obs_id].values[0]
            # The path to where the obs_id was initially downloaded
            field_dir = os.path.join(self.raw_data_path, "temp_download", field_name)
            # Only executing the method if new data has been downloaded,
            #  can check if new data is there if there is a temp_download_{fieldname} directory
            if os.path.exists(field_dir):
                # The path to the obs_id directory (ie. the final DAXA constistent format)
                obs_dir = os.path.join(self.raw_data_path, obs_id)
                # Making the new ObsID directory
                if not os.path.exists(obs_dir):
                    os.makedirs(obs_dir)
                    # Not including hidden files in this list
                    all_files = [f for f in os.listdir(field_dir) if not f.startswith('.')]
                    # Some fields are in a folder, some are just the files not in a folder
                    # If they are in a folder, there will only be one file in all files
                    if len(all_files) == 1:
                        second_field_dir = all_files[0]
                        # redefining all_files so it lists the files in the folder
                        all_files = [f for f in os.listdir(os.path.join(field_dir, second_field_dir))
                                     if not f.startswith('.')]
                        # Redefining field_dir so in the later block, the source is correct
                        field_dir = os.path.join(field_dir, second_field_dir)

                        # Some of the fields are in another folder, so need to perform the same check again (pretty
                        #  sure this only applies to efeds and eta cha)
                        if len(all_files) == 1:
                            third_field_dir = all_files[0]
                            # Redefining all_files, so it lists the files in the folder
                            all_files = os.listdir(os.path.join(field_dir, third_field_dir))
                            # Redefining field_dir so in the later block, the source is correct
                            field_dir = os.path.join(field_dir, third_field_dir)

                    # Selecting the event list for the obs_id
                    obs_file_name = [obs_file for obs_file in all_files
                                     if obs_id in obs_file and "eRO" not in obs_file ][0]
                    source = os.path.join(field_dir, obs_file_name)
                    dest = os.path.join(obs_dir, obs_file_name)
                    shutil.move(source, dest)

            else:
                pass

        # Deleting temp_download directory containing the field_name directories that contained
        #  extra files that were not the obs_id event lists
        temp_dir = os.path.join(self.raw_data_path, "temp_download")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
    @_lock_check
    def filter_on_fields(self, fields: Union[str, List[str]]):
        """
        This filtering method will select only observations included in the fields specified.

        :param str/List[str] allowed_fields: The fields or field types (or list of fields or field types)
            that you wish to be let through the filter.
        """

        # Convert field types or a singular field name into a list of field name(s)
        fields = self._check_chos_fields(fields=fields)

        # Updating the chosen_field attribute
        self.chosen_fields = fields

        # Selecting all Obs_IDs from each field
        field_obs_ids = EROSITA_CALPV_INFO.loc[EROSITA_CALPV_INFO["Field_Name"].isin(fields), "ObsID"].tolist()

        # Uses the Pandas isin functionality to find the rows of the overall observation table that match the input
        #  ObsIDs. This outputs a boolean array.
        sel_obs_mask = self._obs_info['ObsID'].isin(field_obs_ids).values
        # Said boolean array can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array*sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter
    
    @_lock_check
    def filter_on_obs_ids(self, allowed_obs_ids: Union[str, List[str]]):
        """
        This filtering method will select only observations with IDs specified by the allowed_obs_ids argument.

        Please be aware that filtering methods are cumulative, so running another method will not remove the
        filtering that has already been applied, you can use the reset_filter method for that.

        :param str/List[str] allowed_obs_ids: The ObsID (or list of ObsIDs) that you wish to be let
            through the filter.
        """
        # Had to overwrite this function from BaseMission since there is an issue with an eROSITA obs_id
        if not isinstance(allowed_obs_ids, list):
            allowed_obs_ids = [allowed_obs_ids]

        # Accounting for the wrong ObsID being written on the eROSITA website
        if '700195' in allowed_obs_ids:
            allowed_obs_ids.remove('700195')
            allowed_obs_ids.append('700199')
            allowed_obs_ids.append('700200')
            warn("The ObsID '700195' is misstyped on the eROSITA early data release website. It has"
                 " been replaced with '700199', '700200' which are the true ObsIDs associated with the "
                 "Puppis A galactic field.", stacklevel=2)

        super().filter_on_obs_ids(allowed_obs_ids)

    def download(self, num_cores: int = NUM_CORES, download_products: bool = True):
        """
        A method to acquire and download the eROSITA Calibration and Performance Validation data that 
        have not been filtered out (if a filter has been applied, otherwise all data will be downloaded). 
        Fields (or field types) specified by the chosen_fields property will be downloaded, which is set 
        either on declaration of the class instance or by passing a new value to the chosen_fields property. 
        Downloaded data is then filtered according to Instruments specified by the chosen_instruments property
        (set in the same manner as chosen_fields).

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: UNLIKE MOST MISSIONS, this does not actually change what is downloaded, but
            rather changes the DAXA classification of the downloaded event lists from raw to raw+preprocessed. This
            means they would be included in the processed data storage structure of an archive.
        """
        # Ensures that a directory to store the 'raw' eROSITACalPV data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        if download_products:
            # We can only download raw event lists for eROSITACalPV, but I'm going to mark it as raw+preprocessed so
            #  that they can be copied into the processed data structure, as the eROSITA flaring is pretty low it
            #  should be safe
            self._download_type = "raw+preprocessed"
        else:
            # In this case the classification of the downloaded data is just 'raw', so that processing will be possible
            self._download_type = 'raw'

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + o) for o in self.filtered_obs_ids]):
            self._download_done = True

        # Getting all the obs_ids that haven't already been downloaded
        obs_to_download = list(set(self.filtered_obs_ids) - set(os.listdir(stor_dir)))
        # Getting all the unique download links (since the CalPV data is downloaded in whole fields, rather than
        #  individual obs_ids)
        download_links = list(set(EROSITA_CALPV_INFO.loc[EROSITA_CALPV_INFO['ObsID'].isin(
            obs_to_download), 'download']))
    
        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(download_links), desc="Downloading "
                                                          "{} data".format(self._pretty_miss_name)) as download_prog:
                    for link in download_links:
                        # Use the internal static method I set up which both downloads and unpacks the
                        #  eROSITACalPV data
                        self._download_call(raw_dir=self.raw_data_path, link=link)
                        # Update the progress bar
                        download_prog.update(1)

            elif num_cores > 1:
                # List to store any errors raised during download tasks
                raised_errors = []

                # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                with tqdm(total=len(download_links), desc="Downloading {} data".format(self._pretty_miss_name)) \
                        as download_prog, Pool(num_cores) as pool:

                    # The callback function is what is called on the successful completion of a _download_call
                    def callback(download_conf: Any):
                        """
                        Callback function for the apply_async pool method, gets called when a download task finishes
                        without error.

                        :param Any download_conf: The Null value confirming the operation is over.
                        """
                        nonlocal download_prog  # The progress bar will need updating
                        download_prog.update(1)

                    # The error callback function is what happens when an exception is thrown during a _download_call
                    def err_callback(err):
                        """
                        The callback function for errors that occur inside a download task running in the pool.

                        :param err: An error that occurred inside a task.
                        """
                        nonlocal raised_errors
                        nonlocal download_prog

                        if err is not None:
                            # Rather than throwing an error straight away I append them all to a list for later.
                            raised_errors.append(err)
                        download_prog.update(1)

                    # Again nested for loop through ObsIDs and instruments
                    for link in download_links:
                        # Add each download task to the pool
                        pool.apply_async(self._download_call,
                                         kwds={'raw_dir': self.raw_data_path, 'link': link},
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                # Raise all the download errors at once, if there are any
                if len(raised_errors) != 0:
                    raise DAXADownloadError(str(raised_errors))

            else:
                raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

            # Rearranging the obs_id eventlists into the directory format DAXA expects
            self._directory_formatting()

            # Only doing the instrument filtering step if not all the instruments have been chosen
            if len(self.chosen_instruments) != 7:
                # Getting all the path for each eventlist corresponding to an obs_id for the
                #  _inst_filtering function later
                fits_paths = [self.get_evt_list_path(o) for o in self.filtered_obs_ids]

                # Filtering out any events from the raw data that arent from the selected instruments
                if num_cores == 1:
                    with tqdm(total=len(self), desc="Selecting EventLists from "
                                                    "{}".format(self.chosen_instruments)) as inst_filter_prog:
                        for path in fits_paths:
                            self._inst_filtering(insts=self.chosen_instruments, evlist_path=path)
                            # Update the progress bar
                            inst_filter_prog.update(1)

                elif num_cores > 1:
                    # List to store any errors raised during download tasks
                    raised_errors = []

                    # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(self.chosen_instruments)) \
                            as inst_filter_prog, Pool(num_cores) as pool:

                        # The callback function is what is called on the successful completion of a _download_call
                        def callback(download_conf: Any):
                            """
                            Callback function for the apply_async pool method, gets called when a download task
                            finishes without error.

                            :param Any download_conf: The Null value confirming the operation is over.
                            """
                            nonlocal inst_filter_prog  # The progress bar will need updating
                            inst_filter_prog.update(1)

                        # The error callback function is what happens when an exception is thrown
                        #  during a _download_call
                        def err_callback(err):
                            """
                            The callback function for errors that occur inside a download task running in the pool.

                            :param err: An error that occurred inside a task.
                            """
                            nonlocal raised_errors
                            nonlocal inst_filter_prog

                            if err is not None:
                                # Rather than throwing an error straight away I append them all to a list for later.
                                raised_errors.append(err)
                            inst_filter_prog.update(1)

                        # Again nested for loop through each Obs_ID
                        for path in fits_paths:
                            # Add each download task to the pool
                            pool.apply_async(self._inst_filtering, kwds={'insts': self.chosen_instruments,
                                                                         'evlist_path': path},
                                             error_callback=err_callback, callback=callback)
                        pool.close()  # No more tasks can be added to the pool
                        pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                    # Raise all the download errors at once, if there are any
                    if len(raised_errors) != 0:
                        raise DAXADownloadError(str(raised_errors))

                else:
                    raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

            self._download_done = True
        
        else:
            warn("The raw data for this mission have already been downloaded.", stacklevel=2)

    def get_evt_list_path(self, obs_id: str, inst: str = None) -> str:
        """
        A get method that provides the path to a downloaded pre-generated event list for the current mission (if
        available). This method will not work if pre-processed data have not been downloaded.

        :param str obs_id: The ObsID of the event list.
        :param str inst: The instrument of the event list (if applicable).
        :return: The requested event list path.
        :rtype: str
        """
        # Just setting the instrument to a known instrument - it doesn't matter for eROSITA because they're all
        #  shipped in the same files - this is the reason this method overrides the base implementation. Sort of wish
        #  I'd done all of them like this...
        inst = self.chosen_instruments[0]

        rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_evt_name.format(oi=obs_id))
        # This performs certain checks to make sure the file exists, and fill in any wildcards
        rel_pth = self._get_prod_path_post_checks(rel_pth, obs_id, inst, 'event list')

        return rel_pth

    def assess_process_obs(self, obs_info: dict):
        """
        A slightly unusual method which will allow the eROSITACalPV mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        
        insts = list(obs_info.keys())

        # The dictionary which will be set back will have top level instrument dictionaries with the 
        # following keys lower level sub keys:
        #   usable --> this dependent on the filter_wheel setting 
        #   included --> this is to indicate whether this instrument is included in the chosen instruments
        # We start off by assuming all the filters are set to OPEN and all the instruments are included
        to_return = {inst: {'usable': True} for inst in insts}

        for inst in to_return:
            rel_info = obs_info[inst]

            # Want to check that the observation was taken when the filter wheel was on OPEN or FILTER
            if rel_info['filter'] not in ['OPEN', 'FILTER']:
                to_return[inst]['usable'] = False
        
        return to_return

    def ident_to_obsid(self, ident: str):
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
        return ident[:6]


class eRASS1DE(BaseMission):
    """
    The mission class for the first data release of the German half of the eROSITA All-Sky Survey

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """
    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
        The mission class for the first data release of the German half of the eROSITA All-Sky Survey

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        # Call the init of parent class with the required information
        super().__init__()

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['ra_min', 'ra_max', 'dec_min', 'dec_max', 'neigh_obs']

        # Runs the method which fetches information on all available eROSITACalPV observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()

        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

        # Sets the default instruments
        if insts is None:
            insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']

        # Setting all the possible instruments that can be associated with eROSITA data
        self._miss_poss_insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        # Setting the user specified instruments
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for ROSAT images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        self._template_en_trans = {Quantity(0.2, 'keV'): {Quantity(0.6, 'keV'): "1",
                                                          Quantity(2.3, 'keV'): "4",
                                                          Quantity(0.5, 'keV'): "5"},
                                   Quantity(0.6, 'keV'): {Quantity(2.3, 'keV'): "2"},
                                   Quantity(2.3, 'keV'): {Quantity(5.0, 'keV'): "3"},
                                   Quantity(0.5, 'keV'): {Quantity(1.0, 'keV'): "6"},
                                   Quantity(1, 'keV'): {Quantity(2.0, 'keV'): "7"}}

        # We set up the eROSITA file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        # The wildcards are needed because that second character describes the 'owner' of the tile - m=MPE,
        #  c=calibration, b=MPE+IKE - guess we could have populated that from the observation info table but ah well
        #  if it works...
        self._template_evt_name = "EXP_010/e*01_{oi}_020_EventList_c010.fits"
        self._template_img_name = "EXP_010/e*01_{oi}_02{eb}_Image_c010.fits"
        self._template_exp_name = "DET_010/e*01_{oi}_02{eb}_ExposureMap_c010.fits"
        self._template_bck_name = "DET_010/e*01_{oi}_02{eb}_BackgrImage_c010.fits"

        # Call the name property to set up the name and pretty name attributes
        self.name

        # Runs the method which fetches information on all available RASS observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

        # We now will read in the previous state, if there is one to be read in.
        if save_file_path is not None:
            self._load_state(save_file_path)

    # Defining properties first
    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission.

        :return: The mission name
        :rtype: str
        """
        # This is defined here (as well as in the init of BaseMission) because I want people to just copy this
        #  property if they're making a new subclass, then replace None with the name of the mission.
        self._miss_name = "erosita_all_sky_de_dr1"
        # Used for things like progress bar descriptions
        self._pretty_miss_name = "eRASS DE:1"
        return self._miss_name

    @property
    def chosen_instruments(self) -> List[str]:
        """
        Property getter for the names of the currently selected instruments associated with this mission which
        will be processed into an archive by DAXA functions. Overwritten here because there
        are custom behaviours for eROSITA.

        :return: A list of instrument names
        :rtype: List[str]
        """
        return self._chos_insts

    @chosen_instruments.setter
    @_lock_check
    def chosen_instruments(self, new_insts: List[str]):
        """
        Property setter for the instruments associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value. Overwritten here because there
        are custom behaviours for eROSITA.

        :param List[str] new_insts: The new list of instruments associated with this mission which should
            be processed into the archive.
        """
        new_insts = self.check_inst_names(new_insts)

        if (len(self._chos_insts) == 0 or np.array(deepcopy(new_insts)).sort() !=
                np.array(deepcopy(self._chos_insts)).sort()):
            changed = True
            self._chos_insts = new_insts
        else:
            changed = False

        # Checking if the data has already been downloaded:
        if self._download_done and changed:
            # TODO I don't think this is a safe way of doing this - not sure we should be doing it at all
            # Only doing the instrument filtering if not all the instruments have been chosen
            if len(new_insts) != 7:
                # Getting all the path for each eventlist corresponding to an obs_id for the
                #  _inst_filtering function later
                fits_paths = [self.get_evt_list_path(o) for o in self.filtered_obs_ids]

                # Filtering out any events from the raw data that aren't from the selected instruments
                if NUM_CORES == 1:
                    with tqdm(total=len(self), desc="Selecting EventLists from "
                                                    "{}".format(new_insts)) as inst_filter_prog:
                        for path in fits_paths:
                            self._inst_filtering(insts=new_insts, evlist_path=path)
                            # Update the progress bar
                            inst_filter_prog.update(1)

                elif NUM_CORES > 1:
                    # List to store any errors raised during download tasks
                    raised_errors = []

                    # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(new_insts)) \
                            as inst_filter_prog, Pool(NUM_CORES) as pool:

                        # The callback function is what is called on the successful completion of a _download_call
                        def callback(download_conf: Any):
                            """
                            Callback function for the apply_async pool method, gets called when a download task
                            finishes without error.

                            :param Any download_conf: The Null value confirming the operation is over.
                            """
                            nonlocal inst_filter_prog  # The progress bar will need updating
                            inst_filter_prog.update(1)

                        # The error callback function is what happens when an exception is thrown
                        #  during a _download_call
                        def err_callback(err):
                            """
                            The callback function for errors that occur inside a download task running in the pool.

                            :param err: An error that occurred inside a task.
                            """
                            nonlocal raised_errors
                            nonlocal inst_filter_prog

                            if err is not None:
                                # Rather than throwing an error straight away I append them all to a list for later.
                                raised_errors.append(err)
                            inst_filter_prog.update(1)

                        # Again nested for loop through each Obs_ID
                        for path in fits_paths:
                            # Add each download task to the pool
                            pool.apply_async(self._inst_filtering, kwds={'insts': new_insts, 'evlist_path': path},
                                             error_callback=err_callback, callback=callback)
                        pool.close()  # No more tasks can be added to the pool
                        pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                    # Raise all the download errors at once, if there are any
                    if len(raised_errors) != 0:
                        raise DAXADownloadError(str(raised_errors))

                else:
                    raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission.

        :return: The coordinate frame of the RA-Dec.
        :rtype: BaseRADecFrame
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        # FK5 is an assumption because I can't find anything to contradict it - doesn't really matter anyway
        self._miss_coord_frame = FK5
        return self._miss_coord_frame

    @property
    def id_regex(self) -> str:
        """
        Property getter for the regular expression (regex) pattern for observation IDs of this mission.

        :return: The regex pattern for observation IDs.
        :rtype: str
        """
        # The ObsID regular expression is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        # TODO THIS WILL NEED TO BE CHECKED WHEN I GET ACCESS TO DR1
        self._id_format = '^[0-9]{6}$'
        return self._id_format

    @property
    def fov(self) -> Union[Quantity, dict]:
        """
        Property getter for the approximate field of view set for this mission. This is the radius/half-side-length of
        the field of view. In cases where the field of view is not square/circular, it is the half-side-length of
        the longest side.

        :return: The approximate field of view(s) for the mission's instrument(s). In cases with multiple instruments
            then this may be a dictionary, with keys being instrument names.
        :rtype: Union[Quantity, dict]
        """
        # The approximate field of view is defined here because I want to force implementation for each
        #  new mission class.
        self._approx_fov = Quantity(1.8, 'degree')
        return self._approx_fov

    @property
    def all_obs_info(self) -> pd.DataFrame:
        """
        A property getter that returns the base dataframe containing information about all the observations available
        for an instance of a mission class.

        :return: A pandas dataframe with (at minimum) the following columns; 'ra', 'dec', 'ObsID', 'science_usable',
            'start', 'duration'
        :rtype: pd.DataFrame
        """
        return self._obs_info

    @all_obs_info.setter
    def all_obs_info(self, new_info: pd.DataFrame):
        """
        Property setter that allows the setting of a new all-observation-information dataframe. This is the dataframe
        that contains information on every possible observation for a mission.

        :param pd.DataFrame new_info: The new dataframe to update the all observation information.
        """
        self._obs_info_checks(new_info)
        self._obs_info = new_info

    # Then define user-facing methods
    def _fetch_obs_info(self):
        """
        This method uses the hard coded csv file to pull information on all German eRASS:1 observations.
        The data are processed into a Pandas dataframe and stored.
        """
        # Hard coded this information and saved it to the erass_de_dr1_info.csv file in /files
        # Making a copy so that ERASS_DE_DR1_INFO remains unchanged
        erass_dr1_copy = ERASS_DE_DR1_INFO.copy()

        # I prefer lowercase column names, so I make sure they are
        erass_dr1_copy = erass_dr1_copy.rename(columns={cn: cn.lower() for cn in erass_dr1_copy.columns})
        # Apart from ObsID of course, I prefer that camel case, because why be consistent?
        erass_dr1_copy = erass_dr1_copy.rename(columns={'obsid': 'ObsID'})

        # Converting the start and end time columns to datetimes
        erass_dr1_copy['start'] = pd.to_datetime(erass_dr1_copy['start'], utc=False, format="%Y-%m-%dT%H:%M:%S",
                                                 errors='coerce')
        erass_dr1_copy['end'] = pd.to_datetime(erass_dr1_copy['end'], utc=False, format="%Y-%m-%dT%H:%M:%S",
                                               errors='coerce')
        erass_dr1_copy['duration'] = erass_dr1_copy['end'] - erass_dr1_copy['start']

        # Have to assume this for all of them for now
        erass_dr1_copy['science_usable'] = True

        # I want to keep the information about which ObsIDs are neighbours to the one in each row, but not in separate
        #  columns, so I join them all into a string and stick them in one column
        field_cols = erass_dr1_copy.columns[erass_dr1_copy.columns.str.contains('field')]
        erass_dr1_copy['neigh_obs'] = erass_dr1_copy[field_cols].agg(','.join, axis=1)

        # Including the relevant information for the final all_obs_info DataFrame
        obs_info_pd = erass_dr1_copy[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration',
                                      'ra_min', 'ra_max', 'dec_min', 'dec_max', 'neigh_obs']]
        # Finally, setting the all_obs_info property with our dataframe
        self.all_obs_info = obs_info_pd

    @staticmethod
    def _inst_filtering(insts: List[str], evlist_path: str):
        """
        Method to filter event lists for eRASS1DE data based on instrument choice.

        :param List[str] insts: The self.chosen_instruments attribute.
        :param str evlist_path: This is the file path to the raw event list for a certain ObsID
            that has NOT been filtered for the users instrument choice yet.
        """

        # Getting a string of TM numbers to add to the end of the file name
        insts_str = ''.join(sorted(re.findall(r'\d+', ''.join(insts))))

        # Checking that this combination of instruments has not been filtered for before for this Obsid
        #  this is done by checking that there is no file with the _if_{}.fits ending where {} is the
        # number(s) of the TM(s) that the user has specified when declaring the mission.
        # Indexing the string at [:-5] removes the .fits part of the file path
        if os.path.exists(evlist_path[:-5] + '_if_{}.fits'.format(insts_str)):
            pass
        else:
            # Reading in the file
            with fits.open(evlist_path) as fits_file:
                # Selecting the telescope module number column
                data = fits_file[1].data
                t_col = data["TM_NR"]

                # Putting inst names into correct format to search in t_col for
                gd_insts = [int(re.sub('[^0-9]', '', tscope)) for tscope in insts]

                # Getting the indexes of events with the chosen insts
                gd_insts_indx = np.where(np.isin(t_col, gd_insts))[0]

                # Filtering the data on those telescopes
                filtered_data = data[gd_insts_indx]

                # Replacing unfiltered event list in the fits file with the new ones
                fits_file[1].data = filtered_data

                # Writing this to a new file (the if is for instrument filtered)
                fits_file.writeto(evlist_path[:-5] + '_if_{}.fits'.format(insts_str))

    @staticmethod
    def _download_call(obs_id: str, raw_dir: str, download_products: bool, pipeline_version: str = None):
        """
        This internal static method is purely to enable parallelised downloads of data, as defining
        an internal function within download causes issues with pickling for multiprocessing.

        :param str obs_id: The ObsID (RRRDDD, where RRR is an integer representation of the central RA, and DDD is an
            integer representation of the central Dec) of the eROSITA All-Sky Survey 1 data to be downloaded.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        :param bool download_products: Controls whether pre-generated images and exposure maps are included in the
            download of this eRASS1DE ObsID data.
        :param str pipeline_version: The processing pipeline version used to generate the data that is to be
            downloaded. The default is None, in which case the latest available will be used.
        """

        # This, once formatted, is the link to the top-level directory of the specified ObsID - that directory
        #  contains sub-directories with various other processed products that people might want to download.
        top_url = 'https://erosita.mpe.mpg.de/dr1/erodat/data/download/{D}/{R}/'

        # Splitting the ObsID into the RA and Dec components, we shall need them for the link
        rrr = obs_id[:3]
        ddd = obs_id[3:]

        # The populated link, the directory it leads too should contain the following subdirectories;
        #  DET_010/ EXP_010/ SOU_010/ UPP_010/
        obs_url = top_url.format(D=ddd, R=rrr)

        # Relevant directories that we check for are defined here, by the choice of whether to download products\
        if not download_products:
            req_dir = REQUIRED_DIRS['erosita_all_sky_de_dr1']['all']
        else:
            req_dir = REQUIRED_DIRS['erosita_all_sky_de_dr1']['products']

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory
        all_web_data = [en['href'] for en in BeautifulSoup(session.get(obs_url).text, "html.parser").find_all("a")]
        top_data = [en for en in all_web_data if any([rd in en for rd in req_dir])]

        # The directory names indicate the version of the processing pipeline that was used to generate the data, and
        #  as the user can specify the version (and as we want to use the latest version if they didn't) we need to
        #  see what is available
        vers = list(set([td.split('_')[-1].replace('/', '') for td in top_data]))
        if pipeline_version is not None and pipeline_version not in vers:
            raise ValueError("The specified pipeline version ({p}) is not available for "
                             "{oi}".format(p=pipeline_version, oi=obs_id))
        else:
            pipeline_version = vers[np.argmax([int(pv) for pv in vers])]
        
        # Final check that the online archive directory that we're pointing at does actually contain the data
        #  directories we expect it too. Every mission I've implemented I seem to have done this in a slightly
        #  different way, but as eROSITA is an active project things are more liable to change and I think this
        #  should be able to fail fairly informatively if that does happen
        req_dir = [(rd + '_' + pipeline_version + '/') for rd in req_dir]
        req_dir_missing = [rd for rd in req_dir if rd not in all_web_data]
        if len(req_dir_missing) > 0:
            raise FileNotFoundError("The archive data directory for {o} does not contain the following required "
                                    "directories; {rq}".format(o=obs_id, rq=", ".join(req_dir_missing)))

        for rd in req_dir:
            if 'EXP_' in rd and not download_products:
                down_patt = ['_EventList_']
            elif 'EXP_' in rd:
                down_patt = ['_EventList_', '_Image_']
            elif 'DET_' in rd:
                down_patt = ['_ExposureMap_', '_BackgrImage_']

            # This is the directory to which we will be saving this archive directories files
            local_dir = raw_dir + '/{}/'.format(obs_id) + rd
            # Make sure that the local directory is created
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # Set up the url for the current required directory
            cur_url = obs_url + rd + '/'

            # Then use beautiful soup to find out what files are present at that directory, and reduce those entries
            #  to just things with 'fits' in
            all_files = [en['href'] for en in BeautifulSoup(session.get(cur_url).text, "html.parser").find_all("a")
                         if 'fits' in en['href']]

            # Finally we strip anything that doesn't match the file pattern defined by whether the user wants
            #  pre-generated products or not
            to_down = [f for patt in down_patt for f in all_files if patt in f]
            # Now we cycle through the files and download them
            for down_file in to_down:
                down_url = cur_url + down_file
                with session.get(down_url, stream=True) as acquiro:
                    with open(local_dir + down_file, 'wb') as writo:
                        copyfileobj(acquiro.raw, writo)

                # There are a few compressed fits files in each archive
                if '.gz' in down_file:
                    # Open and decompress the events file
                    with gzip.open(local_dir + down_file, 'rb') as compresso:
                        # Open a new file handler for the decompressed data, then funnel the decompressed events there
                        with open(local_dir + down_file.split('.gz')[0], 'wb') as writo:
                            copyfileobj(compresso, writo)
                    # Then remove the tarred file to minimize storage usage
                    os.remove(local_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, download_products: bool = True, pipeline_version: int = None):
        """
        A method to acquire and download the German eROSITA All-Sky Survey DR1 data that
        have not been filtered out (if a filter has been applied, otherwise all data will be downloaded).
        Downloaded data is then filtered according to Instruments specified by the chosen_instruments property
        (set in the same manner as chosen_fields).

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: This controls whether the data downloaded include the images and exposure maps
            generated by the eROSITA team and included in the first data release. The default is True.
        :param int pipeline_version: The processing pipeline version used to generate the data that is to be
            downloaded. The default is None, in which case the latest available will be used.
        """
        # Ensures that a directory to store the 'raw' eRASS1DE data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        # We store the type of data that was downloaded
        if download_products:
            self._download_type = "raw+preprocessed"
        else:
            self._download_type = "raw"

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + o) for o in self.filtered_obs_ids]):
            self._download_done = True

        # Getting all the obs_ids that haven't already been downloaded
        obs_to_download = list(set(self.filtered_obs_ids) - set(os.listdir(stor_dir)))

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(obs_to_download), desc="Downloading {} "
                                                           "data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in obs_to_download:
                        self._download_call(obs_id=obs_id, raw_dir=self.raw_data_path,
                                            download_products=download_products, pipeline_version=pipeline_version)
                        # Update the progress bar
                        download_prog.update(1)

            elif num_cores > 1:
                # List to store any errors raised during download tasks
                raised_errors = []

                # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                with tqdm(total=len(obs_to_download), desc="Downloading {} data".format(self._pretty_miss_name)) \
                        as download_prog, Pool(num_cores) as pool:

                    # The callback function is what is called on the successful completion of a _download_call
                    def callback(download_conf: Any):
                        """
                        Callback function for the apply_async pool method, gets called when a download task finishes
                        without error.

                        :param Any download_conf: The Null value confirming the operation is over.
                        """
                        nonlocal download_prog  # The progress bar will need updating
                        download_prog.update(1)

                    # The error callback function is what happens when an exception is thrown during a _download_call
                    def err_callback(err):
                        """
                        The callback function for errors that occur inside a download task running in the pool.

                        :param err: An error that occurred inside a task.
                        """
                        nonlocal raised_errors
                        nonlocal download_prog

                        if err is not None:
                            # Rather than throwing an error straight away I append them all to a list for later.
                            raised_errors.append(err)
                        download_prog.update(1)

                    # Again nested for loop through ObsIDs and instruments
                    for obs_id in obs_to_download:
                        # Add each download task to the pool
                        pool.apply_async(self._download_call,
                                         kwds={'raw_dir': self.raw_data_path, 'obs_id': obs_id,
                                               'download_products': download_products,
                                               'pipeline_version': pipeline_version},
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                # Raise all the download errors at once, if there are any
                if len(raised_errors) != 0:
                    raise DAXADownloadError(str(raised_errors))

            else:
                raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

            # Only doing the instrument filtering step if not all the instruments have been chosen
            if len(self.chosen_instruments) != 7:
                # Getting all the path for each event list corresponding to an obs_id for the
                #  _inst_filtering function later
                fits_paths = [self.get_evt_list_path(o) for o in self.filtered_obs_ids]

                # Filtering out any events from the raw data that aren't from the selected instruments
                if num_cores == 1:
                    with tqdm(total=len(self), desc="Selecting EventLists from "
                                                    "{}".format(self.chosen_instruments)) as inst_filter_prog:
                        for path in fits_paths:
                            self._inst_filtering(insts=self.chosen_instruments, evlist_path=path)
                            # Update the progress bar
                            inst_filter_prog.update(1)

                elif num_cores > 1:
                    # List to store any errors raised during download tasks
                    raised_errors = []

                    # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(self.chosen_instruments)) \
                            as inst_filter_prog, Pool(num_cores) as pool:

                        # The callback function is what is called on the successful completion of a _download_call
                        def callback(download_conf: Any):
                            """
                            Callback function for the apply_async pool method, gets called when a download task
                            finishes without error.

                            :param Any download_conf: The Null value confirming the operation is over.
                            """
                            nonlocal inst_filter_prog  # The progress bar will need updating
                            inst_filter_prog.update(1)

                        # The error callback function is what happens when an exception is thrown
                        #  during a _download_call
                        def err_callback(err):
                            """
                            The callback function for errors that occur inside a download task running in the pool.

                            :param err: An error that occurred inside a task.
                            """
                            nonlocal raised_errors
                            nonlocal inst_filter_prog

                            if err is not None:
                                # Rather than throwing an error straight away I append them all to a list for later.
                                raised_errors.append(err)
                            inst_filter_prog.update(1)

                        # Again nested for loop through each Obs_ID
                        for path in fits_paths:
                            # Add each download task to the pool
                            pool.apply_async(self._inst_filtering, kwds={'insts': self.chosen_instruments,
                                                                         'evlist_path': path},
                                             error_callback=err_callback, callback=callback)
                        pool.close()  # No more tasks can be added to the pool
                        pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                    # Raise all the download errors at once, if there are any
                    if len(raised_errors) != 0:
                        raise DAXADownloadError(str(raised_errors))

                else:
                    raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

            self._download_done = True

        else:
            warn("The raw data for this mission have already been downloaded.", stacklevel=2)

    def get_evt_list_path(self, obs_id: str, inst: str = None) -> str:
        """
        A get method that provides the path to a downloaded pre-generated event list for the current mission (if
        available). This method will not work if pre-processed data have not been downloaded.

        :param str obs_id: The ObsID of the event list.
        :param str inst: The instrument of the event list (if applicable).
        :return: The requested event list path.
        :rtype: str
        """
        # Just setting the instrument to a known instrument - it doesn't matter for eROSITA because they're all
        #  shipped in the same files - this is the reason this method overrides the base implementation. Sort of wish
        #  I'd done all of them like this...
        inst = self.chosen_instruments[0]

        rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_evt_name.format(oi=obs_id))
        # This performs certain checks to make sure the file exists, and fill in any wildcards
        rel_pth = self._get_prod_path_post_checks(rel_pth, obs_id, inst, 'event list')

        return rel_pth

    def get_image_path(self, obs_id: str, lo_en: Quantity = None, hi_en: Quantity = None, inst: str = None) -> str:
        """
        A get method that provides the path to a downloaded pre-generated image for the current mission (if
        available). This method will not work if pre-processed data have not been downloaded.

        :param str obs_id: The ObsID of the image.
        :param Quantity lo_en: The lower energy bound of the image.
        :param Quantity hi_en: The upper energy bound of the image.
        :param str inst: The instrument of the image (if applicable).
        :return: The requested image file path.
        :rtype: str
        """
        # Just setting the instrument to a known instrument - it doesn't matter for eROSITA because they're all
        #  shipped in the same files - this is the reason this method overrides the base implementation. Sort of wish
        #  I'd done all of them like this...
        inst = self.chosen_instruments[0]

        if lo_en is not None:
            # We make sure that the provided energy bounds are in keV
            lo_en = lo_en.to('keV')
            hi_en = hi_en.to('keV')

        # Run the pre-checks to make sure inputs are valid and the mission is compatible with the request
        inst, en_bnd_trans, file_inst, lo_en, hi_en = self._get_prod_path_checks(obs_id, inst, lo_en, hi_en)

        # If this quantity is still None by now, it means that the chosen instrument has multiple energy bands
        #  available and the pre-processing method could not fill in the energy range
        if lo_en is None:
            rel_bands = self.preprocessed_energy_bands[inst]
            # Joining the available energy bands into a string for the energy message
            eb_strs = [str(eb[0].value) + "-" + str(eb[1].value) for eb_ind, eb in enumerate(rel_bands)]
            al_eb = ", ".join(eb_strs) + "keV"
            raise ValueError("The 'lo_en' and 'hi_en' arguments cannot be None, as {m}-{i} has multiple energy "
                             "bands available for pre-processed products; {eb} are "
                             "available".format(m=self.pretty_name, i=inst, eb=al_eb))

        # This fishes out the relevant energy-bounds-to-identifying string translation
        bnd_ident = en_bnd_trans[lo_en][hi_en]

        rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_img_name.format(oi=obs_id, i=file_inst,
                                                                                          eb=bnd_ident))

        # This performs certain checks to make sure the file exists, and fill in any wildcards
        rel_pth = self._get_prod_path_post_checks(rel_pth, obs_id, inst, 'image')

        return rel_pth

    def get_expmap_path(self, obs_id: str, lo_en: Quantity = None, hi_en: Quantity = None, inst: str = None) -> str:
        """
        A get method that provides the path to a downloaded pre-generated exposure map for the current mission (if
        available). This method will not work if pre-processed data have not been downloaded.

        :param str obs_id: The ObsID of the exposure map.
        :param Quantity lo_en: The lower energy bound of the exposure map.
        :param Quantity hi_en: The upper energy bound of the exposure map.
        :param str inst: The instrument of the exposure map (if applicable).
        :return: The requested exposure map file path.
        :rtype: str
        """
        # Just setting the instrument to a known instrument - it doesn't matter for eROSITA because they're all
        #  shipped in the same files - this is the reason this method overrides the base implementation. Sort of wish
        #  I'd done all of them like this...
        inst = self.chosen_instruments[0]

        if lo_en is not None:
            # We make sure that the provided energy bounds are in keV
            lo_en = lo_en.to('keV')
            hi_en = hi_en.to('keV')

        # Run the pre-checks to make sure inputs are valid and the mission is compatible with the request
        inst, en_bnd_trans, file_inst, lo_en, hi_en = self._get_prod_path_checks(obs_id, inst, lo_en, hi_en)

        # If this quantity is still None by now, it means that the chosen instrument has multiple energy bands
        #  available and the pre-processing method could not fill in the energy range
        if lo_en is None:
            rel_bands = self.preprocessed_energy_bands[inst]
            # Joining the available energy bands into a string for the energy message
            eb_strs = [str(eb[0].value) + "-" + str(eb[1].value) for eb_ind, eb in enumerate(rel_bands)]
            al_eb = ", ".join(eb_strs) + "keV"
            raise ValueError("The 'lo_en' and 'hi_en' arguments cannot be None, as {m}-{i} has multiple energy "
                             "bands available for pre-processed products; {eb} are "
                             "available".format(m=self.pretty_name, i=inst, eb=al_eb))

        # This fishes out the relevant energy-bounds-to-identifying string translation
        bnd_ident = en_bnd_trans[lo_en][hi_en]

        rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_exp_name.format(oi=obs_id, i=file_inst,
                                                                                          eb=bnd_ident))

        # This performs certain checks to make sure the file exists, and fill in any wildcards
        rel_pth = self._get_prod_path_post_checks(rel_pth, obs_id, inst, 'exposure map')

        return rel_pth

    def get_background_path(self, obs_id: str, lo_en: Quantity = None, hi_en: Quantity = None, inst: str = None) -> str:
        """
        A get method that provides the path to a downloaded pre-generated background map for the current mission (if
        available). This method will not work if pre-processed data have not been downloaded.

        :param str obs_id: The ObsID of the background map.
        :param Quantity lo_en: The lower energy bound of the background map.
        :param Quantity hi_en: The upper energy bound of the background map.
        :param str inst: The instrument of the background map (if applicable).
        :return: The requested background map file path.
        """
        # Just setting the instrument to a known instrument - it doesn't matter for eROSITA because they're all
        #  shipped in the same files - this is the reason this method overrides the base implementation. Sort of wish
        #  I'd done all of them like this...
        inst = self.chosen_instruments[0]

        if lo_en is not None:
            # We make sure that the provided energy bounds are in keV
            lo_en = lo_en.to('keV')
            hi_en = hi_en.to('keV')

        # Run the pre-checks to make sure inputs are valid and the mission is compatible with the request
        inst, en_bnd_trans, file_inst, lo_en, hi_en = self._get_prod_path_checks(obs_id, inst, lo_en, hi_en)

        # If this quantity is still None by now, it means that the chosen instrument has multiple energy bands
        #  available and the pre-processing method could not fill in the energy range
        if lo_en is None:
            rel_bands = self.preprocessed_energy_bands[inst]
            # Joining the available energy bands into a string for the energy message
            eb_strs = [str(eb[0].value) + "-" + str(eb[1].value) for eb_ind, eb in enumerate(rel_bands)]
            al_eb = ", ".join(eb_strs) + "keV"
            raise ValueError("The 'lo_en' and 'hi_en' arguments cannot be None, as {m}-{i} has multiple energy "
                             "bands available for pre-processed products; {eb} are "
                             "available".format(m=self.pretty_name, i=inst, eb=al_eb))

        # This fishes out the relevant energy-bounds-to-identifying string translation
        bnd_ident = en_bnd_trans[lo_en][hi_en]

        rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_bck_name.format(oi=obs_id, i=file_inst,
                                                                                          eb=bnd_ident))

        # This performs certain checks to make sure the file exists, and fill in any wildcards
        rel_pth = self._get_prod_path_post_checks(rel_pth, obs_id, inst, 'background map')

        return rel_pth

    def assess_process_obs(self, obs_info: dict):
        """
        A slightly unusual method which will allow the eRASS1DE mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """

        insts = list(obs_info.keys())

        # The dictionary which will be set back will have top level instrument dictionaries with the
        # following keys lower level sub keys:
        #   usable --> this dependent on the filter_wheel setting
        #   included --> this is to indicate whether this instrument is included in the chosen instruments
        # We start off by assuming all the filters are set to OPEN and all the instruments are included
        to_return = {inst: {'usable': True} for inst in insts}

        for inst in to_return:
            rel_info = obs_info[inst]

            # Want to check that the observation was taken when the filter wheel was on OPEN or FILTER
            if rel_info['filter'] not in ['OPEN', 'FILTER']:
                to_return[inst]['usable'] = False

        return to_return

    def ident_to_obsid(self, ident: str):
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
        # Most eROSITA unique identifiers are just ObsID, but it is also possible that TM idents are included when
        #  we deal with pre-processed data using preprocessed_in_archive. That may change if I decide on a more
        #  elegant way of doing that, but in the meantime we know that all eRASS idents are 6 digits
        return ident[:6]



        




