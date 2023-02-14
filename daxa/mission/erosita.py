import os.path
import tarfile
import requests
from typing import List, Union
from warnings import warn

import numpy as np
import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from tqdm import tqdm

from .base import BaseMission
from .base import _lock_check
from .. import NUM_CORES
from ..config import CalPV_info

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
        self._moss_poss_field_types = CalPV_info["Field_Type"].unique().tolist()

        # Sets the default fields
        if fields is None:
            self.chosen_fields = self._miss_poss_fields
        else:
            # Using the property setter because it calls the internal _check_chos_fields function
            #  which deals with the fields being given as a name or field type
            self.chosen_fields = fields

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
    def _download_call(field: str, filename: str):
        """
        This internal static method is purely to enable parallelised downloads of data, as defining
        an internal function within download causes issues with pickling for multiprocessing.

        :param str field: The field name of the particular field to be downloaded.
        :param str filename: The directory under which to save the downloaded tar.gz.
        :return: A None value.
        :rtype: Any
        """
        # Another part of the very unsophisticated method I currently have for checking whether a raw XMM data
        #  download has already been performed (see issue #30). If the ObsID directory doesn't exist then
        #  an attempt will be made.
        if not os.path.exists(filename):
            os.makedirs(filename)
            # Download the requested data
            r = requests.get(CalPV_info["download"].loc[CalPV_info["Field_Name"] == field].values[0])
            # The full file path
            file_path = filename + "/{}.tar.gz".format(field)
            # Saving the requested data
            open(file_path, "wb").write(r.content)
            # Unpacking the tar file
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall()
            tar.close()
            # Then remove the original compressed tar to save space
            os.remove(file_path)

        return None


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
        if all([os.path.exists(stor_dir + '{f}'.format(f=f)) for f in self._miss_poss_fields]):
            self._download_done = True
        
        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for field in self._chos_fields:
                        # Use the internal static method I set up which both downloads and unpacks the XMM data
                        self._download_call(field, filename=stor_dir + '/{f}'.format(f=field))
                        # Update the progress bar
                        download_prog.update(1)

            else:
                raise NotImplementedError("The download for {} currently only works on one core".format(self._pretty_miss_name))
        
            self._download_done = True

        else:
            warn("The raw data for this mission have already been downloaded.")






        




