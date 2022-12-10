#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 10/12/2022, 15:42. Copyright (c) The Contributors
import os.path
import tarfile
from datetime import datetime
from multiprocessing import Pool
from typing import List, Union, Any
from warnings import warn

import numpy as np
import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astroquery import log
from astroquery.esa.xmm_newton import XMMNewton as AQXMMNewton
from tqdm import tqdm

from .base import BaseMission
from .. import NUM_CORES
from ..exceptions import DAXADownloadError

log.setLevel(0)


class XMMPointed(BaseMission):
    """
    The mission class for pointed XMM observations (i.e. slewing observations are NOT included in the data accessed
    and collected by instances of this class). The available observation information is fetched from the XMM Science
    Archive using AstroQuery, and data are downloaded with the same module.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
    """
    def __init__(self, insts: Union[List[str], str] = None):
        """
        The mission class init for pointed XMM observations (i.e. slewing observations are NOT included in the data
        accessed and collected by instances of this class). The available observation information is fetched from
        the XMM Science Archive using AstroQuery, and data are downloaded with the same module.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from.
        """
        # Call the init of parent class with the required information
        super().__init__()

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
        self._required_mission_specific_cols = ['proprietary_end_date', 'usable_proprietary', 'usable_science',
                                                'revolution']

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
        #  the BaseMission superclass. Suggest keeping this in a format that would be good for a unix
        #  directory name (i.e. lowercase + underscores), because it will be used as a directory name
        self._miss_name = "xmm_pointed"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "XMM-Newton Pointed"
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
                                             "duration, proprietary_end_date, revolution "
                                             "from v_all_observations".format(num_obs))
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

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], level: str, filename: str):
        """
        This internal static method is purely to enable parallelised downloads of XMM data, as defining
        an internal function within download causes issues with pickling for multiprocessing.

        :param str observation_id: The ObsID of the particular observation to be downloaded.
        :param List[str] insts: The names of instruments to be retained - currently all instruments ODFs
            must be downloaded, and then irrelevant instruments must be deleted. The names must be
            in the two-character format expected by the XSA AIO URLs (i.e. PN, M1, R1, OM, etc.)
        :param str level: The level of data to be downloaded. Either ODF or PPS is supported.
        :param str filename: The filename under which to save the downloaded tar.gz.
        :return: A None value.
        :rtype: Any
        """
        # Another part of the very unsophisticated method I currently have for checking whether a raw XMM data
        #  download has already been performed (see issue #30). If the ObsID directory doesn't exist then
        #  an attempt will be made.
        if not os.path.exists(filename):
            # Set this again here because otherwise its annoyingly verbose
            log.setLevel(0)
            # Download the requested data
            AQXMMNewton.download_data(observation_id=observation_id, level=level, filename=filename)
            # As the above function downloads the data as compressed tars, we need to decompress them
            with tarfile.open(filename+'.tar.gz') as zippo:
                zippo.extractall(filename)

            # Then remove the original compressed tar to save space
            os.remove(filename+'.tar.gz')

            # Finally, the actual telescope data is in another .tar, so we expand that as well, first making
            #  sure that there is only one tar in the observations folder
            rel_tars = [f for f in os.listdir(filename) if "{o}.tar".format(o=observation_id) in f.lower()]

            # Checks to make sure there is only one tarred file (otherwise I don't know what this will be
            #  unzipping)
            if len(rel_tars) == 0 or len(rel_tars) > 1:
                raise ValueError("Multiple tarred ODFs were detected for {o}, and cannot be "
                                 "unpacked".format(o=observation_id))

            # Variable to store the name of the tarred file (included revolution number and ObsID, hence why
            #  I don't just construct it myself, don't know the revolution number a priori)
            to_untar = filename + '/{}'.format(rel_tars[0])

            # Open and untar the file
            with tarfile.open(to_untar) as tarro:
                # untar_path = to_untar.split('.')[0] + '/'
                untar_path = filename + '/'
                tarro.extractall(untar_path)
            # Then remove the tarred file to minimise storage usage
            os.remove(to_untar)

            # This part removes ODFs which belong to instruments the user hasn't requested, but we have
            #  to make sure to add the code 'SC' otherwise spacecraft information files will get removed
            to_keep = insts + ['SC']
            throw_away = [f for f in os.listdir(untar_path) if 'MANIFEST' not in f
                          and f.split(observation_id+'_')[1][:2] not in to_keep]
            for for_removal in throw_away:
                os.remove(untar_path + for_removal)

        return None

    def download(self, num_cores: int = NUM_CORES):
        """
        A method to acquire and download the pointed XMM data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        """

        # Ensures that a directory to store the 'raw' pointed XMM data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.top_level_path + self.name + '_raw'):
            os.makedirs(self.top_level_path + self.name + '_raw')
        # Just make a shorthand variable for the storage path
        stor_dir = self.top_level_path + self.name + '_raw/'

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + '{o}'.format(o=o)) for o in self.filtered_obs_ids]):
            self._download_done = True

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in self.filtered_obs_ids:
                        # Use the internal static method I set up which both downloads and unpacks the XMM data
                        self._download_call(obs_id, insts=self.chosen_instruments, level='ODF',
                                            filename=stor_dir + '{o}'.format(o=obs_id))
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
                    for obs_id in self.filtered_obs_ids:
                        # Add each download task to the pool
                        pool.apply_async(self._download_call,
                                         kwds={'observation_id': obs_id, 'insts': self.chosen_instruments,
                                               'level': 'ODF', 'filename': stor_dir + '{o}'.format(o=obs_id)},
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                # Raise all the download errors at once, if there are any
                if len(raised_errors) != 0:
                    raise DAXADownloadError(str(raised_errors))

            else:
                raise ValueError("The value of NUM_CORES must be greater than or equal to 1.")

            # This is set to True once the download is done, and is used by archives to tell if data have been
            #  downloaded for a particular mission or not
            self._download_done = True

        else:
            warn("The raw data for this mission have already been downloaded.")

