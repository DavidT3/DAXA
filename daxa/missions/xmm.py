#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/11/2022, 18:42. Copyright (c) The Contributors
from datetime import datetime
from typing import List, Union
from warnings import warn

import numpy as np
import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astroquery import log
from astroquery.esa.xmm_newton import XMMNewton as AQXMMNewton

from .base import BaseMission

log.setLevel(0)


class XMMPointed(BaseMission):
    """
    The mission class for pointed XMM observations (i.e. slewing observations are NOT included in the data accessed
    and collected by instances of this class). The available observation information is fetched from the XMM Science
    Archive using AstroQuery, and data are downloaded with the same module.

    :param str output_archive_name: The name under which the eventual processed archive will be stored.
    :param str output_path: The top-level path where an archive directory will be created. If this is set to None
        then the class will default to the value specified in the configuration file.
    """
    def __init__(self, output_archive_name: str, output_path: str = None, insts: Union[List[str], str] = None):
        """
        The mission class init for pointed XMM observations (i.e. slewing observations are NOT included in the data
        accessed and collected by instances of this class). The available observation information is fetched from
        the XMM Science Archive using AstroQuery, and data are downloaded with the same module.

        :param str output_archive_name: The name under which the eventual processed archive will be stored.
        :param str output_path: The top-level path where an archive directory will be created. If this is set to None
            then the class will default to the value specified in the configuration file.
        :param List[str]/str insts:
        """
        # Call the init of parent class with the required information
        super().__init__(output_archive_name, output_path)

        # Sets the default instruments - #TODO Perhaps update these to include RGS and OM, once they're supported
        if insts is None:
            insts = ['M1', 'M2', 'PN']
        else:
            # Makes sure everything is uppercase
            insts = [i.upper() for i in insts]

        self._miss_poss_insts = ['M1', 'M2', 'PN', 'OM', 'R1', 'R2']
        # The chosen_instruments property setter (see below) will use these to convert possible contractions
        #  of XMM instrument names to the names that the module expects. The M1, M2 etc. form is not one I favour,
        #  but is what the download function provided by astroquery wants, so that's what I'm going to use
        self._alt_miss_inst_names = {'MOS1': 'M1', 'MOS2': 'M2', 'RGS1': 'R1', 'RGS2': 'R2'}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['proprietary_end_date', 'usable_proprietary', 'usable_science']

        # Runs the method which fetches information on all available pointed XMM observations and stores that
        #  information in the all_obs_info property
        self.fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

    # Defining properties first
    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission.

        :return: The mission name
        :rtype: str
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_name = "XMM Pointed"
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
    def id_regex(self) -> str:
        """
        Property getter for the regular expression (regex) pattern for observation IDs of this mission.

        :return: The regex pattern for observation IDs.
        :rtype: str
        """
        # The ObsID regular expression is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._id_format = '^[0-9]{10}$'
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

    # Then define user-facing methods
    def fetch_obs_info(self):
        """
        This method uses the AstroQuery table access protocol implemented for the XMM Science Archive to pull
        down information on all of the pointed XMM observations which are stored in XSA. The data are processed
        into a Pandas dataframe and stored.
        """
        # First of all I want to know how many entries there are in the 'all observations' table, because I need to
        #  specify the number of rows to select in my ADQL (Astronomical Data Query Language) command for reasons
        #  I'll explain in a second
        count_tab = AQXMMNewton.query_xsa_tap('select count(observation_id) from xsa.v_all_observations')
        # Then I round up to the nearest 1000, probably unnecessary but oh well
        num_obs = np.ceil(count_tab['count'].tolist()[0] / 1000).astype(int) * 1000
        # Now I have to be a bit cheesy - If I used select * (which is what I would normally do in an SQL-derived
        #  language to grab every row) it actually only returns the top 2000. I think that * is replaced with TOP 2000
        #  before the query is sent to the server. However if I specify a TOP N, where N is greater than 2000, then it
        #  works as intended. I hope this is a stable behaviour!
        # TODO Might want to grab footprint_fov, stc_s at some point
        obs_info = AQXMMNewton.query_xsa_tap("select TOP {} ra, dec, observation_id, start_utc, with_science, "
                                             "duration, proprietary_end_date from v_all_observations".format(num_obs))
        # The above command has gotten some basic information; central coordinates, observation ID, start time
        #  and duration, whether the data are proprietary etc. Now this Astropy table object is turned into a
        #  Pandas dataframe (which I much prefer working with).
        obs_info_pd: pd.DataFrame = obs_info.to_pandas()

        # Convert the string representation of proprietary period ending into a datetime object. I have to use
        #  errors='coerce' here because for some reason some proprietary end times are set ~1000 years in
        #  the future, which Pandas implementation of datetime does not like. Errors coerce means that such
        #  datetimes are just set to NaT (not a time) rather than erroring everything out.
        obs_info_pd['proprietary_end_date'] = pd.to_datetime(obs_info_pd['proprietary_end_date'], utc=False,
                                                             errors='coerce')
        # Convert the start time to a datetime
        obs_info_pd['start_utc'] = pd.to_datetime(obs_info_pd['start_utc'], utc=False, errors='coerce')
        # Grab the current date and time
        today = datetime.today()

        # This adds a column that describes whether the data are out of their proprietary period, and thus
        #  usable by the general community. Can just use less than or equal to operator because everything involved
        #  is now a datetime object.
        obs_info_pd['usable_proprietary'] = obs_info_pd['proprietary_end_date'].apply(
            lambda x: ((x <= today) & (pd.notnull(x)))).astype(bool)

        # Just renaming some of the columns
        obs_info_pd = obs_info_pd.rename(columns={'observation_id': 'ObsID', 'with_science': 'usable_science',
                                                  'start_utc': 'start'})

        # Converting the duration column to a timedelta object, which can then be directly added to the start column
        #  which should be a datetime object itself
        obs_info_pd['duration'] = pd.to_timedelta(obs_info_pd['duration'], 's')
        # Now creating an end column by adding duration to start
        obs_info_pd['end'] = obs_info_pd.apply(lambda x: x.start + x.duration, axis=1)

        # This checks for NaN values of RA or Dec, which for some reason do appear sometimes??
        obs_info_pd['radec_good'] = obs_info_pd.apply(lambda x: np.isfinite(x['ra']) & np.isfinite(x['dec']), axis=1)
        # Throws a warning if there are some.
        if len(obs_info_pd) != obs_info_pd['radec_good'].sum():
            warn("{ta} of the {tot} observations located for this mission have been removed due to NaN "
                 "RA or Dec values".format(ta=len(obs_info_pd)-obs_info_pd['radec_good'].sum(), tot=len(obs_info_pd)),
                 stacklevel=2)
        # Cut the total information down to just those that don't have NaN positions. I've done it this way rather
        #  than adding the radec_good column as another input to the usable column (see below) because having NaN
        #  positions really screws up the filter_on_positions method in BaseMission
        obs_info_pd = obs_info_pd[obs_info_pd['radec_good']]
        # Create a combined usable column from usable_science and usable_proprietary - this overall usable column
        #  is required by the BaseMission superclass and governs whether an observation will be considered from the
        #  outset.
        obs_info_pd['usable'] = obs_info_pd['usable_science'] * obs_info_pd['usable_proprietary']
        # Don't really care about this column now so remove.
        del obs_info_pd['radec_good']

        self.all_obs_info = obs_info_pd

#     def download(self, num_cores: int = 1):
#
#         #
#         #     for cmd_ind, cmd in enumerate(all_run):
#         #         # These are just the relevant entries in all these lists for the current command
#         #         # Just defined like this to save on line length for apply_async call.
#         #         exp_type = all_type[cmd_ind]
#         #         exp_path = all_path[cmd_ind]
#         #         ext = all_extras[cmd_ind]
#         #         src = source_rep[cmd_ind]
#         #         pool.apply_async(execute_cmd, args=(str(cmd), str(exp_type), exp_path, ext, src),
#         #                          error_callback=err_callback, callback=callback)
#         #     pool.close()  # No more tasks can be added to the pool
#         #     pool.join()  # Joins the pool, the code will only move on once the pool is empty.
#
#         if num_cores == 1:
#             with tqdm(total=len(self)*len(self.chosen_instruments), desc="Downloading XMM data") as download_prog:
#                 for obs_id in self.filtered_obs_ids:
#                     for inst in self.chosen_instruments:
#                         AQXMMNewton.download_data(obs_id, instname=inst, level='ODF',
#                                                   filename='testo_{o}'.format(o=obs_id))
#
#
# # with tqdm(total=len(all_run), desc="Generating products of type(s) " + prod_type_str,
#         #           disable=disable) as gen, Pool(cores) as pool:
#         #     def callback(results_in: Tuple[BaseProduct, str]):
#         #         """
#         #         Callback function for the apply_async pool method, gets called when a task finishes
#         #         and something is returned.
#         #         :param Tuple[BaseProduct, str] results_in: Results of the command call.
#         #         """
#         #         nonlocal gen  # The progress bar will need updating
#         #         nonlocal results  # The dictionary the command call results are added to
#         #         if results_in[0] is None:
#         #             gen.update(1)
#         #             return
#         #         else:
#         #             prod_obj, rel_src = results_in
#         #             results[rel_src].append(prod_obj)
#         #             gen.update(1)
#         #
#         #     def err_callback(err):
#         #         """
#         #         The callback function for errors that occur inside a task running in the pool.
#         #         :param err: An error that occurred inside a task.
#         #         """
#         #         nonlocal raised_errors
#         #         nonlocal gen
#         #
#         #         if err is not None:
#         #             # Rather than throwing an error straight away I append them all to a list for later.
#         #             raised_errors.append(err)
#         #         gen.update(1)

