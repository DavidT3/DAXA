import numpy as np
import pandas as pd
import pkg_resources
from typing import List, Union

from .base import BaseMission

class eROSITACalPV(BaseMission):
    """
    The mission class for the eROSITA early data release observations made during the Calibration and Performance 
    Verification program. 

    :param [str]/str fields: The fields or field type that the user is choosing to download/process data from.
    :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
    """
    def __init__(self, fields: Union[List[str], str] = None, insts: Union[List[str], str] = None):
        """
        The mission class for the eROSITA early data release observations made during the Calibration and Performance 
        Verification program. 

        :param [str]/str fields: The fields or field types that the user is choosing to download/process data from.
        :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
        """
        # Call the init of parent class with the required information
        super().__init__() 

        # Reading in what data is available in the Cal-PV release
        CalPV_info = pd.read_csv(pkg_resources.resource_filename(__name__, "files/eROSITACalPV_info.csv"), header="infer")
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

        # DAVID_QUESTION what is going on with crab
        # DAVID_QUESTION what shall we do about catalogues

        # Sets the default instruments
        if insts is None:
            insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        else:
            # Make sure everything is uppercase
            insts = [i.upper() for i in insts]
        
        self._miss_poss_insts = ['TM1', 'TM2', 'TM3', 'TM4', 'TM5', 'TM6', 'TM7']
        self.chosen_instruments = insts
    
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
    # DAVID_QUESTION do i need to do a lock check here?
    #@_lock_check
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
        # Reading in what data is available in the Cal-PV release
        CalPV_info = pd.read_csv(pkg_resources.resource_filename(__name__, "files/eROSITACalPV_info.csv"), header="infer")
        
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





        




