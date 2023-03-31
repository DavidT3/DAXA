#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 28/03/2023, 09:51. Copyright (c) The Contributors

import os
import re
import shutil
import tarfile
from multiprocessing import Pool
from shutil import copyfileobj
from typing import List, Union, Any
from warnings import warn

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from tqdm import tqdm

from .base import BaseMission
from .base import _lock_check
from .. import NUM_CORES
from ..config import CALPV_INFO
from ..exceptions import DAXADownloadError


class eROSITACalPV(BaseMission):
    """
    The mission class for the eROSITA early data release observations made during the Calibration and Performance 
    Verification program. 

    :param List[str]/str fields: The eROSITA calibration field name(s) or type to download/process data from. 
    :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
    """
    def __init__(self, insts: Union[List[str], str] = None, fields: Union[List[str], str] = None):
        """
        The mission class for the eROSITA early data release observations made during the Calibration and Performance 
        Verification program. 

        :param List[str]/str fields: The eROSITA calibration field name(s) or type to download/process data from.
        :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
        """
        # Call the init of parent class with the required information
        super().__init__() 

        # All the allowed names of fields 
        self._miss_poss_fields = CALPV_INFO["Field_Name"].tolist()
        # All the allowed types of field, ie. survey, magellanic cloud, galactic field, extragalactic field
        self._miss_poss_field_types = CALPV_INFO["Field_Type"].unique().tolist()

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['Field_Name', 'Field_Type']
        
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
        # Call the name property to set up the name and pretty name attributes
        self.name

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
    def chosen_instruments(self) -> List[str]:
        """
        Property getter for the names of the currently selected instruments associated with this mission which
        will be processed into an archive by DAXA functions. Overwritten here because I want to use a custom
        version of _check_chos_insts for eROSITA.

        :return: A list of instrument names
        :rtype: List[str]
        """
        return self._chos_insts

    @chosen_instruments.setter
    @_lock_check
    def chosen_instruments(self, new_insts: List[str]):
        """
        Property setter for the instruments associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value. Overwritten here because I want
        to use a custom version of _check_chos_insts for eROSITA.

        :param List[str] new_insts: The new list of instruments associated with this mission which should
            be processed into the archive.
        """
        self._chos_insts = self._check_chos_insts(new_insts)
        
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
    

    def _check_chos_insts(self, insts: Union[List[str], str]):
        """
        An internal function to check and peform event list filtering for instruments for eROSITA. This
        overwrites the version of this method declared in BaseMission, though it does call the super method.
        This sub-class of BaseMission re-implements this method so that setting chosen instruments also 
        filters event lists for user specified instruments, as eROSITA observations contain all instruments.

        :param List[str]/str insts:
        :return: The list of instruments (possibly altered to match formats expected by this module).
        :rtype: List
        """
        insts = super()._check_chos_insts(insts)

        # Checking if the data has already been downloaded:
        if all([os.path.exists(self.raw_data_path + '{o}'.format(o=obs)) for obs in self.filtered_obs_ids]):
            # Only doing the instrument filtering if not all the instruments have been chosen
            if len(insts) != 7:
                # Getting all the path for each eventlist corresponding to an obs_id for the _inst_filtering function later
                fits_paths = [self._get_evlist_path_from_obs(obs=o) for o in self.filtered_obs_ids]

                # Filtering out any events from the raw data that arent from the selected instruments
                if NUM_CORES == 1:
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(insts)) as inst_filter_prog:
                        for path in fits_paths:
                            self._inst_filtering(insts=insts, evlist_path=path)
                            # Update the progress bar
                            inst_filter_prog.update(1)

                elif NUM_CORES > 1:
                    # List to store any errors raised during download tasks
                    raised_errors = []

                    # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(insts)) \
                            as inst_filter_prog, Pool(NUM_CORES) as pool:

                        # The callback function is what is called on the successful completion of a _download_call
                        def callback(download_conf: Any):
                            """
                            Callback function for the apply_async pool method, gets called when a download task finishes
                            without error.

                            :param Any download_conf: The Null value confirming the operation is over.
                            """
                            nonlocal inst_filter_prog  # The progress bar will need updating
                            inst_filter_prog.update(1)

                        # The error callback function is what happens when an exception is thrown during a _download_call
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
                            pool.apply_async(self._inst_filtering, kwds={'insts': insts, 
                                            'evlist_path': path}, 
                                            error_callback=err_callback, callback=callback)
                        pool.close()  # No more tasks can be added to the pool
                        pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                    # Raise all the download errors at once, if there are any
                    if len(raised_errors) != 0:
                        raise DAXADownloadError(str(raised_errors))

                else:
                    raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")
        
        return insts
    
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

    @_lock_check
    def filter_on_fields(self, fields: Union[str, List[str]]):
        """
        This filtering method will select only observations included in the fields specified.

        :param str/List[str] allowed_fields: The fields or field types (or list of fields or field types) 
            that you wish to be let through the filter.
        """

        # Convert field types or a singular field name into a list of field name(s)
        fields = self._check_chos_fields(fields=fields)

        # Updating the chosen_field attribute
        self.chosen_fields = fields

        # Selecting all Obs_IDs from each field
        field_obs_ids = CALPV_INFO.loc[CALPV_INFO["Field_Name"].isin(fields), "ObsID"].tolist()

        # Uses the Pandas isin functionality to find the rows of the overall observation table that match the input
        #  ObsIDs. This outputs a boolean array.
        sel_obs_mask = self._obs_info['ObsID'].isin(field_obs_ids).values
        # Said boolean array can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array*sel_obs_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter
    
    # Then define user-facing methods
    def _fetch_obs_info(self):
        """
        This method uses the hard coded csv file to pull information on all eROSITACalPV observations. 
        The data are processed into a Pandas dataframe and stored.
        """
        # Hard coded this information and saved it to the CALPV_INFO.csv in /files
        # Making a copy so that CALPV_INFO remains unchanged
        calpv_copy = CALPV_INFO

        # Need to split the times since they go to milisecond precision, 
        #  which is a pain to translate to a datetime object, and is superfluous information anyway
        calpv_copy['start'] = [str(time).split('.', 1)[0] for time in calpv_copy['start']]
        calpv_copy['end'] = [str(time).split('.', 1)[0] for time in calpv_copy['end']]
        calpv_copy['start'] = pd.to_datetime(calpv_copy['start'], utc=False, format="%Y-%m-%dT%H:%M:%S", errors='coerce')
        calpv_copy['end'] = pd.to_datetime(calpv_copy['end'], utc=False, format="%Y-%m-%dT%H:%M:%S", errors='coerce')

        # Including the relevant information for the final all_obs_info DataFrame
        obs_info_pd = calpv_copy[['ra', 'dec', 'ObsID', 'usable', 'start', 'end', 'duration', 'Field_Name', 'Field_Type']]

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
        #  with underscores to match the entries in CALPV_INFO 
        fields = [re.sub("[-()+/. ]", "_", field.upper()) for field in fields]

        # In case people use roman numerals or dont include the brackets in their input
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
            raise ValueError("Some field names or field types {bf} are not associated with this mission, please"
                            " choose from the following fields; {gf} or field types; {gft}".format(
                            bf=",".join(bad_fields), gf=",".join(self._miss_poss_fields), gft=",".join(self._miss_poss_field_types)))
        
        # Extracting the alt_fields from fields
        alt_fields = [field for field in fields if field in poss_alt_field_names]
        # Making a list of the alt_fields DAXA compatible name
        alt_fields_proper_name = [poss_alt_field_names[field] for field in alt_fields]
        # Seeing if someone just input 'crab' into the fields argument
        if 'CRAB' in fields:
            crab = ['CRAB_1', 'CRAB_2', 'CRAB_3', 'CRAB_4']
        else:
            crab = []
        # Then the extracting the field_types
        field_types = [field for field in fields if field in self._miss_poss_field_types]
        # Turning the field_types into field_names
        field_types_proper_name = CALPV_INFO.loc[CALPV_INFO["Field_Type"].isin(field_types), "Field_Name"].tolist()
        # Then extracting the field names from fields
        field_names = [field for field in fields if field in self._miss_poss_fields]
    
        # Adding all these together to make the final fields list
        updated_fields = alt_fields_proper_name + crab + field_types_proper_name + field_names
        # Removing the duplicates from updated_fields
        updated_fields = list(set(updated_fields))

        # Return the chosen fields 
        return updated_fields

    @staticmethod
    def _inst_filtering(insts: List[str], evlist_path: str):
        """
        Method to filter event lists for eROSITACalPV data based on instrument choice.
        
        :param List[str] insts: The self.chosen_instruments attribute.
        :param str evlist_path: This is the file path to the raw eventlist for a certain ObsID
         that has NOT been filtered for the users instrument choice yet.
        """

        # Getting a string of TM numbers to add to the end of the file name
        insts_str = ''.join(sorted(re.findall(r'\d+', ''.join(insts))))

        # Checking that this combination of instruments has not been filtered for before for this Obsid
        #  this is done by checking that there is no file with the _if_{}.fits ending where {} is the 
        #  number(s) of the TM(s) that the user has specified when declaring the mission.
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
                
                # Filtering the data on those tscopes
                filtered_data = data[gd_insts_indx]

                # Replacing unfiltered eventlist in the fits file with the new ones
                fits_file[1].data = filtered_data

                # Writing this to a new file (the if is for instrument filtered)
                fits_file.writeto(evlist_path[:-5] + '_if_{}.fits'.format(insts_str))
        
    @staticmethod
    def _download_call(raw_data_path: str, link: str):
        """
        This internal static method is purely to enable parallelised downloads of data, as defining
        an internal function within download causes issues with pickling for multiprocessing.

        :param str raw_data_path: This is the self.raw_data_path attribute.
        :param str link: The download_link of the particular field to be downloaded.
        :return: A None value.
        :rtype: Any
        """
        # Since you can't download a single observation for a field, you have to download them all in one tar file,
        #  I am making a temporary directories to download the tar file and unpack it in, then move the observations 
        #  to their own directory afterwards in the _directory_formatting function

        # Getting the field name associated with the download link for directory naming purposes
        field_name = CALPV_INFO.loc[CALPV_INFO['download'].isin([link]), 'Field_Name'].tolist()[0]
        # The temporary 
        temp_dir = os.path.join(raw_data_path, "temp_download", field_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        # Download the requested data
        with requests.get(link, stream=True) as r:
            field_dir = os.path.join(temp_dir, field_name)
            os.makedirs(field_dir)
            with open(field_dir + "{f}.tar.gz".format(f=field_name), "wb") as writo:
                copyfileobj(r.raw, writo)
                # unzipping the tar file
                tarname = field_dir + "{f}.tar.gz".format(f=field_name)
                with tarfile.open(tarname, "r:gz") as tar:
                    tar.extractall(field_dir)
                    os.remove(tarname)

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
            field_name = CALPV_INFO["Field_Name"].loc[CALPV_INFO["ObsID"] ==  obs_id].values[0]
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
                        all_files = [f for f in os.listdir(os.path.join(field_dir, second_field_dir)) if not f.startswith('.')]
                        # redefining field_dir so in the later block, the source is correct
                        field_dir = os.path.join(field_dir, second_field_dir)

                        # Some of the fields are in another folder, so need to perform the same check again (pretty sure this only applies to efeds and eta cha)
                        if len(all_files) == 1:
                            third_field_dir = all_files[0]
                            # redefining all_files so it lists the files in the folder
                            all_files = os.listdir(os.path.join(field_dir, third_field_dir))
                            # redefining field_dir so in the later block, the source is correct
                            field_dir = os.path.join(field_dir, third_field_dir)

                    # Selecting the eventlist for the obs_id
                    obs_file_name =  [obs_file for obs_file in all_files if obs_id in obs_file and "eRO" not in obs_file ][0]
                    source = os.path.join(field_dir, obs_file_name)
                    dest = os.path.join(obs_dir, obs_file_name)
                    shutil.move(source, dest)
            
            else:
                pass

        # Deleting temp_download directory containing the field_name directories that contained
        #  extra files that were not the obs_id eventlists
        temp_dir = os.path.join(self.raw_data_path, "temp_download")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        

    def _get_evlist_path_from_obs(self, obs: str):
        '''
        Internal method to get the unfiltered, downloaded event list path for a given
        obs id, for use in the download method. 

        :param str obs: The obs id for the event list required.
        :return: The path of the event list.
        :rtype: str
        '''
        all_files = os.listdir(os.path.join(self.raw_data_path + obs))

         # This directory could have instrument filtered files in as well as the eventlist
         #  so selecting the eventlist by chosing the one with 4 hyphens in
        file_name = [file for file in all_files if len(re.findall('_', file)) == 4][0]

        ev_list_path = os.path.join(self.raw_data_path + obs, file_name)

        return ev_list_path
    
    def download(self, num_cores: int = NUM_CORES):
        """
        A method to acquire and download the eROSITA Calibration and Performance Validation data that 
        have not been filtered out (if a filter has been applied, otherwise all data will be downloaded). 
        Fields (or field types) specified by the chosen_fields property will be downloaded, which is set 
        either on declaration of the class instance or by passing a new value to the chosen_fields property. 
        Donwloaded data is then filtered according to Instruments specified by the chosen_instruments property 
        (set in the same manner as chosen_fields).

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        """
        # Ensures that a directory to store the 'raw' eROSITACalPV data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + o) for o in self.filtered_obs_ids]):
            self._download_done = True

        # Getting all the obs_ids that havent already been downloaded
        obs_to_download = list(set(self.filtered_obs_ids) - set(os.listdir(stor_dir)))
        # Getting all the unique download links (since the CalPV data is downloaded in whole fields, rather than individual obs_ids)
        download_links = list(set(CALPV_INFO.loc[CALPV_INFO['ObsID'].isin(obs_to_download), 'download']))
    
        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(download_links), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for link in download_links:
                        # Use the internal static method I set up which both downloads and unpacks the eROSITACalPV data
                        self._download_call(raw_data_path=self.raw_data_path, link=link)
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
                                            kwds={'raw_data_path': self.raw_data_path, 'link': link},
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

            # Only doing the instrument filtering step if not all the instruments have been chosen
            if len(self.chosen_instruments) != 7:
                # Getting all the path for each eventlist corresponding to an obs_id for the _inst_filtering function later
                fits_paths = [self._get_evlist_path_from_obs(obs=o) for o in self.filtered_obs_ids]

                # Filtering out any events from the raw data that arent from the selected instruments
                if num_cores == 1:
                    with tqdm(total=len(self), desc="Selecting EventLists from {}".format(self.chosen_instruments)) as inst_filter_prog:
                        for path in fits_paths:
                            self._inst_filtering(insts=self.chosen_instruments, evlist_path=path)
                            # Update the progress bar
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
                            Callback function for the apply_async pool method, gets called when a download task finishes
                            without error.

                            :param Any download_conf: The Null value confirming the operation is over.
                            """
                            nonlocal inst_filter_prog  # The progress bar will need updating
                            inst_filter_prog.update(1)

                        # The error callback function is what happens when an exception is thrown during a _download_call
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
            warn("The raw data for this mission have already been downloaded.")






        




