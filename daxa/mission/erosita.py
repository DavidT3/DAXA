import os
import shutil
import tarfile
import requests
from typing import List, Union, Any
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from tqdm import tqdm

from .base import BaseMission
from .base import _lock_check
from .. import NUM_CORES
from ..config import CalPV_info
from ..exceptions import DAXADownloadError

class eROSITACalPV(BaseMission):
    """
    The mission class for the eROSITA early data release observations made during the Calibration and Performance 
    Verification program. 

    :param [str]/str fields: The fields or field type that the user is choosing to download/process data from.
    :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
    """
    def __init__(self, insts: Union[List[str], str] = None, fields: Union[List[str], str] = None):
        """
        The mission class for the eROSITA early data release observations made during the Calibration and Performance 
        Verification program. 

        :param [str]/str fields: The fields or field types that the user is choosing to download/process data from.
        :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
        """
        # Call the init of parent class with the required information
        super().__init__() 

        # All the allowed names of fields 
        self._miss_poss_fields = CalPV_info["Field_Name"].tolist()
        # All the allowed types of field, ie. survey, magellanic cloud, galactic field, extragalactic field
        self._miss_poss_field_types = CalPV_info["Field_Type"].unique().tolist()

        # Sets the default fields
        if fields is None:
            self.chosen_fields = self._miss_poss_fields
        else:
            # Using the property setter because it calls the internal _check_chos_fields function
            #  which deals with the fields being given as a name or field type
            self.chosen_fields = fields
            # Applying filters on obs_ids so that only obs_ids associated with the chosen fields are included
            self._filter_on_fields(fields)

        # JESS_TODO what is going on with crab

        # Sets the default instruments
        if insts is None:
            insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        else:
            # Make sure everything is uppercase
            insts = [i.upper() for i in insts]
        
        self._miss_poss_insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        self.chosen_instruments = insts

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
        self._pretty_miss_name = "eROSITA Calibration and Performance Verification"
        return self._miss_name
    
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
        
    # DAVID_QUESTION not sure what id_regex is

    @property
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
        return self._miss_poss_fields
    
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

        :param List[str] new_insts: The new list of fields or field types associated with this mission which should
            be processed into the archive.
        """
        self._chos_fields = self._check_chos_fields(new_fields)
    
    def _field_dict_generator(self): 
        """
        Returns a dictionary with all eROSITACalPV Obs_IDs as keys and their field as the value.

        :param str/List[str] allowed_fields: The fields or field types (or list of fields or field types) 
            that you wish to be let through the filter.
        :return: The field dictionary.
        :rtype: dict
        """
        # Creating a dictionary to store obs_ids as keys and their field name as values
        field_dict = {}
        for field in self._miss_poss_fields:
            obs = CalPV_info["Obs_ID"].loc[CalPV_info["Field_Name"] == field].values[0]
            if "," in obs:
                # This checks if multiple observations are associated with one field
                indv_obs = obs.split(", ")
                for ind_ob in indv_obs:
                    field_dict[ind_ob] = field
            else:
                field_dict[obs] = field
        
        return field_dict

    
    @_lock_check
    def _filter_on_fields(self, fields: Union[str, List[str]]):
        """
        This filtering method will select only observations included in the fields specified.

        :param str/List[str] allowed_fields: The fields or field types (or list of fields or field types) 
            that you wish to be let through the filter.
        """

        # Convert field types or a singular field name into a list of field name(s)
        fields = self._check_chos_fields(fields=fields)

        # Creating a dictionary that store obs_ids as keys and their field name as values
        field_dict = self._field_dict_generator(fields)
        
        # Selecting all Obs_IDs from each field
        field_obs_ids = [obs for field in fields for obs in field_dict if field_dict[obs] == field]

        # Uses the Pandas isin functionality to find the rows of the overall observation table that match the input
        #  ObsIDs. This outputs a boolean array.
        sel_obs_mask = self._obs_info['ObsID'].isin(field_obs_ids)
        # Said boolean array can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array*sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter
    
    # Then define user-facing methods
    def fetch_obs_info(self):
        """
        This method uses the hard coded csv file in to pull information on all eROSITACalPV observations. 
        The data are processed into a Pandas dataframe and stored.
        """
        # DF has 'Obs_ID', 'start', 'duration, 'end'
        # Getting a list of the observations
        obs_unformatted = CalPV_info["Obs_ID"].tolist()

        obs_formatted = []
        for obs in obs_unformatted:
            if "," in obs:
                # This checks if multiple observations are associated with one field
                indv_obs = obs.split(", ")
                for ind_ob in indv_obs:
                    obs_formatted.append(ind_ob)
            else:
                obs_formatted.append(obs)

        # Creating a column for the Obs_IDs
        obs_info_pd = pd.DataFrame(obs_formatted, columns=["Obs_ID"])

        #JESS_TODO need to input the start end and duration deets

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

        # Storing the input fields original format so that the ValueError later will clearer for the user
        input_fields = fields

        # Converting to upper case to match the entries in CalPV_info 
        fields = [field.upper() for field in fields]
        
        if all(field in self._miss_poss_field_types for field in fields):
            # Checking if the fields were given as a field type 
            # Then collecting all the fields associated with that field type/s
            updated_fields = CalPV_info.loc[CalPV_info["Field_Type".isin(fields), "Field_Name"]].tolist()
        elif all(field in self._miss_poss_fields for field in fields):
            # Checking if the field names are valid
            updated_fields = fields
        else:
            # Checking if any of the fields entered are not valid field names or types
            field_test = [field in self.miss_poss_field_types or self._miss_poss_fields for field in fields]
            if not all(field_test):
                bad_fields = np.array(input_fields)[~np.array(field_test)]
                raise ValueError("Some field names or field types {bf} are not associated with this mission, please"
                        "choose from the following fields; {gf} or field types; {gft}".format(
                        bf=",".join(bad_fields), gf=",".join(self._miss_poss_fields), gft=",".join(self._miss_poss_field_types)))

        # Return the chosen fields 
        return updated_fields
    
    @staticmethod
    def _download_call(self, field: str):
        """
        This internal static method is purely to enable parallelised downloads of data, as defining
        an internal function within download causes issues with pickling for multiprocessing.

        :param str field: The field name of the particular field to be downloaded.
        :param str filename: The directory under which to save the downloaded tar.gz.
        :return: A None value.
        :rtype: Any
        """
         # JESS_TODO i think somewhere i need a check that I haven't done the download before

        # Since you can't download a single observation for a field, you have to download them all in one tar file,
        #  I am making a temporary directory to download the tar file and unpack it in, then move the observations 
        #  to their own directory afterwards in the _directory_formatting function
        temp_dir = os.path.join(self.raw_data_path, "temp_download")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Download the requested data
        r = requests.get(CalPV_info["download"].loc[CalPV_info["Field_Name"] == field].values[0])
        if field in {"EFEDS", "ETA_CHA"}:
             # The eFEDS and eta_cha data come in a folder inside the tar file 
             #  instead of files - like the rest of the CalPV data
            tarname = os.path.join(temp_dir, "{f}.tar.gz".format(f=field))
            open(tarname, "wb").write(r.content)
            # unzipping the tar file
            tar = tarfile.open(tarname, "r:gz")
            tar.extractall(temp_dir)
            tar.close()
            os.remove(tarname)
        else: 
            field_dir = os.path.join(temp_dir, "{f}".format(f=field))
            os.makedirs(field_dir)
            open(field_dir + "{f}.tar.gz".format(f=field), "wb").write(r.content)
            # unzipping the tar file
            tarname = field_dir + "{f}.tar.gz".format(f=field)
            tar = tarfile.open(tarname, "r:gz")
            tar.extractall(field_dir)
            tar.close()
            os.remove(tarname)

        return None
    
    def _directory_formatting(self):
        """
        Internal method to rearrange the downloaded files from field names into the Obs_ID top layer 
        directory structure for consistency with other missions in DAXA. To be called after the initial 
        download of the fields has been completed.
        """

        if os.path.exists(os.path.join(self.raw_data_path, "temp_download")):
            # Creating a dictionary that stores obs_ids as keys and their field name as values
            field_dict = self._field_dict_generator()

            # Moving the eventlist for each obs_id from its downloaded path the path DAXA expects
            for o in self.filtered_obs_ids:
                if not os.path.exists(self.raw_data_path + '{o}'.format(o=o)):
                    os.makedirs(self.raw_data_path + '{o}'.format(o=o))
                    # The path to the obs_id directory
                    obs_dir = os.path.join(self.raw_data_path, '{o}'.format(o=o))
                    # The field the obs_id was downloaded with
                    field_name = field_dict[o]
                    # The path to where the obs_id was initially downloaded
                    field_dir = os.path.join(self.raw_data_path, "temp_download", field_name)
                    # All the obs_id files 
                    all_files = os.listdir(os.path.join(self.raw_data_path, "temp_download", field_name))
                    # Selecting the eventlist for the obs_id
                    obs_file_name =  [obs_file for obs_file in all_files if o in obs_file and "eRO" not in obs_file ][0]
                    source = os.path.join(field_dir, obs_file_name)
                    dest = os.path.join(obs_dir, obs_file_name)
                    shutil.move(source, dest)
        
            # Deleting temp_download directory containing the extra files that were not the obs_id eventlists
            shutil.rmtree(os.path.join(self.raw_data_path, "temp_download"))


    def download(self, num_cores: int = NUM_CORES):
        """
        A method to acquire and download the eROSITA CalPV data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Fields (or field types), which is set either 
        on declaration of the class instance or by passing a new value to the chosen_instruments property.
        """
        # Ensures that a directory to store the 'raw' pointed XMM data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.top_level_path + self.name + '_raw'):
            os.makedirs(self.top_level_path + self.name + '_raw')
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + '{o}'.format(o=o)) for o in self.filtered_obs_ids]):
            self._download_done = True
        
        # Since multiple Obs_IDs are downloaded with one field, we are first downloading all fields needed.
        # This dictionary contains which obs_ids are associated with which field
        field_dict = self._field_dict_generator()
        # A list of fields needed to get the required obs_ids (with no duplicate fields)
        required_fields = list(set([field_dict[o] for o in self.filtered_obs_ids]))
        
        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for field in required_fields:
                        # Use the internal static method I set up which both downloads and unpacks the eROSITACalPV data
                        self._download_call(field)
                        # Update the progress bar
                        download_prog.update(1)

            elif num_cores > 1:
                # List to store any errors raised during download tasks
                raised_errors = []

                # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) \
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
                    for field in required_fields:
                        # Add each download task to the pool
                        pool.apply_async(self._download_call,
                                            kwds={'field': field},
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
        
            self._download_done = True





        




