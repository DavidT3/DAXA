#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 12/03/2023, 21:49. Copyright (c) The Contributors
import gzip
import io
import os
from datetime import datetime
from multiprocessing import Pool
from shutil import copyfileobj
from typing import List, Union, Any
from warnings import warn

import pandas as pd
import requests
from astropy.coordinates import BaseRADecFrame, ICRS
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from bs4 import BeautifulSoup
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission

# Unlike NuSTAR, we should only need one directory to be present to download the unprocessed Chandra observations
REQUIRED_DIRS = ['secondary/']
GOOD_FILE_PATTERNS = ['evt1.fits', 'mtl1.fits', 'bias0.fits', 'pbk0.fits', 'flt1.fits', 'bpix1.fits', 'msk1.fits']


class Chandra(BaseMission):
    """
    The mission class for Chandra observations. The available observation information is fetched from the HEASArc
    CHANMASTER table, and data are downloaded from the HEASArc https access to their FTP server. Proprietary data
    are not currently supported by this class.

    This will be the only Chandra mission class, as they do not appear to take data during slewing (like XMM
    and NuSTAR).

    Selecting particular instruments will effectively be putting a filter on the observations, as due to the design
    of Chandra ACIS and HRC instruments cannot observe simultaneously.

    Functionally, this class is very similar to NuSTARPointed.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include ACIS-I, ACIS-S, HRC-I, and HRC-S.
    """

    def __init__(self, insts: Union[List[str], str] = None):
        """
        The mission class for Chandra observations. The available observation information is fetched from the HEASArc
        CHANMASTER table, and data are downloaded from the HEASArc https access to their FTP server. Proprietary data
        are not currently supported by this class.

        This will be the only Chandra mission class, as they do not appear to take data during slewing (like XMM
        and NuSTAR).

        Selecting particular instruments will effectively be putting a filter on the observations, as due to the design
        of Chandra ACIS and HRC instruments cannot observe simultaneously.

        Functionally, this class is very similar to NuSTARPointed.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include ACIS-I, ACIS-S, HRC-I, and HRC-S.
        """
        super().__init__()

        # Sets the default instruments - This is slightly complicated by the fact that the gratings (HETG and LETG)
        #  are configurable too but don't count as an instrument in the CHANMASTER table. They are in a separate
        #  grating column
        if insts is None:
            insts = ['ACIS-I', 'ACIS-S', 'HRC-I', 'HRC-S']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # TODO Remove this once RGS is supported, not sure OM should even be here tbh
        if 'HETG' in insts or 'LETG' in insts:
            raise NotImplementedError("The RGS and OM instruments are not currently supported by this class.")

        # These are the allowed instruments for this mission - Chandra has two sets of instruments (HRC and
        #  ACIS), each with two sets of detectors (one for imaging one for grating spectroscopy). It also has
        #  two choices of grating spectroscopy (HETG and LETG).
        # TODO FIGURE IT OUT - DON'T KNOW THAT THE DETECTOR SKEWS COME IN DIFFERENT EVENT LISTS (PRESUMABLY DO THOUGH)
        self._miss_poss_insts = ['ACIS-I', 'ACIS-S', 'HRC-I', 'HRC-S', 'HETG', 'LETG']
        self._alt_miss_inst_names = {}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['proprietary_end_date', 'target_category', 'detector', 'grating',
                                                'data_mode']

        # Runs the method which fetches information on all available pointed NuSTAR observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

    @property
    def name(self) -> str:
        """
        Property getter for the name of this mission

        :return: The mission name.
        :rtype: str
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass. Suggest keeping this in a format that would be good for a unix
        #  directory name (i.e. lowercase + underscores), because it will be used as a directory name
        self._miss_name = "chandra"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "Chandra"
        return self._miss_name

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission.
        Chandra is the only one so far to actually use ICRS! (or at least it does for its source lists and
        image WCS headers).

        :return: The coordinate frame of the RA-Dec.
        :rtype: BaseRADecFrame
        """
        # The name is defined here because this is the pattern for this property defined in
        #  the BaseMission superclass
        self._miss_coord_frame = ICRS
        return self._miss_coord_frame

    @property
    def id_regex(self) -> str:
        """
        Property getter for the regular expression (regex) pattern for observation IDs of this mission.

        :return: The regex pattern for observation IDs.
        :rtype: str
        """
        # This implementation is slightly different from other mission classes. As Chandra doesn't used consistent
        #  lengths for its ObsIDs, the max number of digits allowed is set dynamically from the table pulled
        #  from HEASArc.
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
        # Frankly I'm not really sure why I made this an abstract method, but possibly because I thought some
        #  missions might need extra checks run on their observation information dataframes?
        # This _obs_info_checks method is defined in BaseMission, and uses the ObsID regex defined near the top of
        #  this class to ensure that the dataframe's ObsID column contains legal values.
        self._obs_info_checks(new_info)
        self._obs_info = new_info
        self.reset_filter()

    def _fetch_obs_info(self):
        """
        This method adapts the 'browse_extract.pl' script (a copy of which can be found in daxa/files for the proper
        credit) to acquire the 'CHANMASTER' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available, proprietary, and scheduled Chandra observations, with important information such as pointing
        coordinates, ObsIDs, and exposure.
        """
        # This is the web interface for querying NASA HEASArc catalogues
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        # This returns the requested information in a FITS format - the idea being I will stream this into memory
        #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited, as we don't know how many NuSTAR observations there are, and the number will
        #  increase with time (so long as the telescope doesn't break...)
        result_max = "&ResultMax=0"
        # This just tells the interface it's a query (I think?)
        action = "&Action=Query"
        # Tells the interface that I want to retrieve from the CHANMASTER (Chandra Master) observation catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20chanmaster"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/all/chanmaster.html)
        which_cols = ['RA', 'DEC', 'TIME', 'OBSID', 'STATUS', 'DETECTOR', 'GRATING', 'EXPOSURE', 'TYPE', 'DATA_MODE',
                      'CLASS', 'PUBLIC_DATE']
        # This is what will be put into the URL to retrieve just those data fields - there are some more, but I
        #  curated it to only those I think might be useful for DAXA
        fields = '&Fields=' + '&varon=' + '&varon='.join(which_cols)

        # The full URL that we will pull the data from, with all the components we have previously defined
        fetch_url = host_url + table_head + action + result_max + down_form + fields

        # Opening that URL, we can access the results of our request!
        with requests.get(fetch_url, stream=True) as urlo:
            # This opens the data as using the astropy fits interface (using io.BytesIO() to stream it into memory
            #  first so that fits.open can access it as an already opened file handler).
            with fits.open(io.BytesIO(urlo.content)) as full_fits:
                # Then convert the data in that fits file just into an astropy table object, and from there to a DF
                full_chandra = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_chandra.select_dtypes(['object']).columns:
                    full_chandra[col] = full_chandra[col].apply(lambda x: x.strip())

            # The ObsID regular expression is defined here because this is the pattern for this property defined in
            #  the BaseMission superclass - Chandra ObsIDs start at 1 and start incrementing - there isn't a
            #  pre-defined length that XMM and NuSTAR have.
            self._id_format = '^[0-9]{1,' + str(len(str(full_chandra['OBSID'].max()))) + '}$'

        # I want the ObsID column to be a string (even though Pandas hates strings) because thats what it is in
        #  other mission classes and being consistent makes everything easier.
        full_chandra['OBSID'] = full_chandra['OBSID'].astype(str)

        # Important first step, select only those observations which have actually been observed (the CHANMASTER
        #  descriptor website defines all the possible status values, but I shall select archived and observed, even
        #  though observed aren't public yet).
        rel_chandra = full_chandra[full_chandra['STATUS'].isin(['processed', 'archived'])]

        # Lower-casing all the column names (personal preference largely).
        rel_chandra = rel_chandra.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects
        rel_chandra = rel_chandra.rename(columns={'obsid': 'ObsID', 'time': 'start',
                                                  'public_date': 'proprietary_end_date',
                                                  'class': 'target_category', 'exposure': 'duration'})
        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        rel_chandra['start'] = pd.to_datetime(Time(rel_chandra['start'].astype(float).values, format='mjd',
                                                   scale='utc').to_datetime())
        # Then make a duration column from the exposure, which is the only thing I have to go on as this table
        #  does not include an end column. This is then used to calculate the end column, as this is required
        rel_chandra['duration'] = pd.to_timedelta(rel_chandra['duration'], 's')
        rel_chandra['end'] = rel_chandra['start'] + rel_chandra['duration']

        # Slightly more complicated with the public release dates, as I have seen some HEASArc tables set them to
        #  0 MJD, which makes the conversion routine quite upset (0 is not valid) - as such I convert only those
        #  which aren't 0, then replace the 0 valued ones with Pandas' Not a Time (NaT) value. Note that I didn't
        #  actually see any such 0 values for Chandra when I created this method, but I'm keeping this in anyway
        val_end_dates = rel_chandra['proprietary_end_date'] != 0
        rel_chandra.loc[val_end_dates, 'proprietary_end_date'] = pd.to_datetime(
            Time(rel_chandra.loc[val_end_dates, 'proprietary_end_date'].values,
                 format='mjd', scale='utc').to_datetime())

        rel_chandra.loc[~val_end_dates, 'proprietary_end_date'] = pd.NaT
        # Grab the current date and time
        today = datetime.today()
        # Create a boolean column that describes whether the data are out of their proprietary period yet
        rel_chandra['usable_proprietary'] = rel_chandra['proprietary_end_date'].apply(
            lambda x: ((x <= today) & (pd.notnull(x)))).astype(bool)

        # Whether the data are public or not is the only criteria for acceptable Chandra data for DAXA at the moment
        rel_chandra['usable'] = rel_chandra['usable_proprietary']

        # Convert the categories of target that are present in the dataframe to the DAXA taxonomy
        conv_dict = {'AGN UNCLASSIFIED': 'AGN', 'EXTENDED GALACTIC OR EXTRAGALACTIC': 'EGE', 'X-RAY BINARY': 'GS',
                     'CV': 'GS', 'Calibration Observations': 'CAL', 'SNR': 'SNR', 'NON-ACTIVE GALAXY': 'NGS',
                     'CLUSTER OF GALAXIES': 'GCL', 'Non-Pointing data': 'MISC'}
        # Obviously some of these have the same key and value, just wanted to demonstrate what I was doing with them. I
        #  am counting director's discretionary time as targets of opportunity for this
        obs_type_dict = {'TOO': 'TOO', 'DDT': 'TOO', 'CAL': 'CAL'}
        # I wanted to define the dictionaries separately, but then just add them together for neatness
        conv_dict.update(obs_type_dict)

        # I don't want to assume that the types I've seen Chandra list will stay forever, as such I construct
        #  a mask that tells me which entries have a recognised description - any that don't will be set to
        #  the 'MISC' code
        type_recog = rel_chandra['target_category'].isin(list(conv_dict.keys()))
        # The recognized target category descriptions are converted to DAXA taxonomy
        rel_chandra.loc[type_recog, 'target_category'] = rel_chandra.loc[type_recog, 'target_category'].apply(
            lambda x: conv_dict[x])
        # Now I set any unrecognized target category descriptions to MISC - there are none at the time of writing,
        #  but that could well change
        rel_chandra.loc[~type_recog, 'target_category'] = 'MISC'

        # Doing things slightly differently for Chandra - as it has a type of observation column and a lot of 'misc'
        #  entries, I'm going to check to see if any of them can be labelled as calibration or Target of oppurtunity,
        #  just to give the user more to work with.
        misc_mask = rel_chandra['target_category'] == 'MISC'
        # For context, GO means General Observer, GTO means Guaranteed Time Observation, and CCT means Chandra
        #  Cool Target (meaning a science target that allows the observatory to cool off)
        rel_chandra.loc[misc_mask, 'target_category'] = rel_chandra[misc_mask].apply(
            lambda x: x.target_category if x.type in ['GO', 'GTO', 'CCT'] else conv_dict[x.type], axis=1)

        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_chandra = rel_chandra[['ra', 'dec', 'ObsID', 'usable', 'start', 'end', 'duration',
                                   'proprietary_end_date', 'target_category', 'detector', 'grating', 'data_mode']]

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_chandra = rel_chandra.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_chandra

    @staticmethod
    def _download_call(observation_id: str, raw_dir: str):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        """
        # The Chandra data are stored in observatories that are named to correspond with the last digit of
        #  the particular observation's ObsID, so we shall extract that for later
        init_id = observation_id[-1]

        # This is the path to the HEASArc data directory for this ObsID
        obs_dir = "/FTP/chandra/data/byobsid/{ii}/{oid}/".format(ii=init_id, oid=observation_id)
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the three directories that I need to download unprocessed Chandra data are present
        # The 'secondary' data products are the L1 unprocessed products that we want
        top_data = [en['href'] for en in BeautifulSoup(session.get(top_url).text, "html.parser").find_all("a")
                    if en['href'] in REQUIRED_DIRS]
        # If the lengths of top_data and REQUIRED_DIRS are different, then one or more of the expected dirs
        #  is not present
        if len(top_data) != len(REQUIRED_DIRS):
            # This list comprehension figures out what directory is missing and reports it
            missing = [rd for rd in REQUIRED_DIRS if rd not in top_data]
            raise FileNotFoundError("The archive data directory for {o} does not contain the following required "
                                    "directories; {rq}".format(o=observation_id, rq=", ".join(missing)))

        # The lower level URL of the directory we're currently looking at
        rel_url = top_url + 'secondary/'
        # This is the directory to which we will be saving this archive directories files
        local_dir = raw_dir + '/'
        # Make sure that the local directory is created
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # We explore the contents of said directory, making sure to clean any useless HTML guff left over - these
        #  are the files we shall be downloading
        to_down = [en['href'] for en in BeautifulSoup(session.get(rel_url).text, "html.parser").find_all("a")
                   if '?' not in en['href'] and obs_dir not in en['href']]

        # This cleans the list of files further, down to only files matching the patterns defined in the constant
        #  Those patterns are designed to grab the files that this page
        #  (https://cxc.cfa.harvard.edu/ciao/data_products_guide/) claims we need for re-processing
        to_down = [f for f in to_down for fp in GOOD_FILE_PATTERNS if fp in f]

        # Every file will need to be unzipped, as they all appear to be gunzipped when I've looked in
        #  the HEASARC directories

        for down_file in to_down:
            down_url = rel_url + down_file
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
                # Then remove the tarred file to minimise storage usage
                os.remove(local_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES):
        """
        A method to acquire and download the pointed Chandra data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        """
        # Ensures that a directory to store the 'raw' Chandra data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.top_level_path + self.name + '_raw'):
            os.makedirs(self.top_level_path + self.name + '_raw')
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        # A very unsophisticated way of checking whether raw data have been downloaded before
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
                        self._download_call(obs_id, raw_dir=stor_dir + '{o}'.format(o=obs_id))
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
                                         kwds={'observation_id': obs_id, 'raw_dir': stor_dir + '{o}'.format(o=obs_id)},
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
