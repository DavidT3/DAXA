#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/08/2024, 11:15. Copyright (c) The Contributors
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
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from bs4 import BeautifulSoup
from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission
from tqdm import tqdm

# Don't require that the event_cl directory be present (cleaned events), as we download the level-1 data (event_uf)
#  and process it ourselves - THAT IS UNLESS the user wants to download the processed data
REQUIRED_DIRS = {'raw': ['auxil/', 'event_uf/', 'hk/'],
                 'processed': ['auxil/', 'event_uf/', 'event_cl/', 'hk/']}


class NuSTARPointed(BaseMission):
    """
    The mission class for pointed NuSTAR observations (i.e. slewing observations are NOT included in the data accessed
    and collected by instances of this class), nor are observations for which the spacecraft mode was 'STELLAR'.
    The available observation information is fetched from the HEASArc NuMASTER table, and data are downloaded from
    the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include FPMA and FPMB (the default
        is both).
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
        The mission class for pointed NuSTAR observations (i.e. slewing observations are NOT included in the data
        accessed and collected by instances of this class), nor are observations for which the spacecraft mode was
        'STELLAR'. The available observation information is fetched from the HEASArc NuMASTER table, and data are
        downloaded from the HEASArc https access to their FTP server. Proprietary data are not currently supported
        by this class.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include FPMA and FPMB (the default
            is both).
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        super().__init__()

        # Sets the default instruments - both instruments that are on NuSTAR
        if insts is None:
            insts = ['FPMA', 'FPMB']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - NuSTAR has two telescopes, and each has its own
        #  Focal Plane Module (FPMx)
        self._miss_poss_insts = ['FPMA', 'FPMB']
        # The chosen_instruments property setter (see below) will use these to convert possible contractions
        #  of NuSTAR names to the names that the module expects. I'm not that familiar with NuSTAR currently, so
        #  I've just put in FA and FB without any real expectation that anyone would use them.
        self._alt_miss_inst_names = {'FA': 'FPMA', 'FB': 'FPMB'}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for ROSAT images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        self._template_en_trans = {Quantity(3., 'keV'): {Quantity(78, 'keV'): ""}}
        self._template_inst_trans = {'FPMA': 'A', 'FPMB': 'B'}

        # We set up the ROSAT file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        self._template_evt_name = "event_cl/nu{oi}{i}01_cl.evt"
        self._template_img_name = "event_cl/nu{oi}{i}01_sk.img"
        self._template_exp_name = None
        self._template_bck_name = None

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['proprietary_end_date', 'exposure_a', 'exposure_b', 'ontime_a',
                                                'ontime_b', 'nupsdout', 'issue_flag', 'target_category',
                                                'proprietary_usable']

        # Runs the method which fetches information on all available pointed NuSTAR observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

        # We now will read in the previous state, if there is one to be read in.
        if save_file_path is not None:
            self._load_state(save_file_path)

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
        self._miss_name = "nustar_pointed"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "NuSTAR Pointed"
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
        #  the BaseMission superclass - NuSTAR observations have a unique 11-digit ObsID, the construction of
        #  which is discussed here (https://heasarc.gsfc.nasa.gov/W3Browse/nustar/numaster.html#obsid)
        self._id_format = '^[0-9]{11}$'
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
        #  new mission class - found conflicting values of either 12 or 13 arcmin across, so I went with half
        #  of 13 to be on the safe side.
        self._approx_fov = Quantity(6.5, 'arcmin')
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
        credit) to acquire the 'numaster' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available, proprietary, and scheduled NuSTAR observations, with important information such as pointing
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
        # Tells the interface that I want to retrieve from the numaster (NuSTAR Master) catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20numaster"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/nustar/numaster.html)
        # The INSTRUMENT_MODE is acquired here even though they say that it is unlikely any observations will be made
        #  in 'normal' mode, just so I can exclude those observations because frankly I don't know the difference
        # SPACECRAFT_MODE is acquired because the 'STELLAR' mode might not be suitable for science so may be excluded
        which_cols = ['RA', 'DEC', 'TIME', 'OBSID', 'STATUS', 'EXPOSURE_A', 'OBSERVATION_MODE', 'PUBLIC_DATE',
                      'ISSUE_FLAG', 'END_TIME', 'EXPOSURE_B', 'INSTRUMENT_MODE', 'NUPSDOUT', 'ONTIME_A', 'ONTIME_B',
                      'SPACECRAFT_MODE', 'SUBJECT_CATEGORY', 'OBS_TYPE']
        # This is what will be put into the URL to retrieve just those data fields - there are quite a few more
        #  but I curated it to only those I think might be useful for DAXA
        fields = '&Fields=' + '&varon=' + '&varon='.join(which_cols)

        # The full URL that we will pull the data from, with all the components we have previously defined
        fetch_url = host_url + table_head + action + result_max + down_form + fields

        # Opening that URL, we can access the results of our request!
        with requests.get(fetch_url, stream=True) as urlo:
            # This opens the data as using the astropy fits interface (using io.BytesIO() to stream it into memory
            #  first so that fits.open can access it as an already opened file handler).
            with fits.open(io.BytesIO(urlo.content)) as full_fits:
                # Then convert the data in that fits file just into an astropy table object, and from there to a DF
                full_nustar = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_nustar.select_dtypes(['object']).columns:
                    full_nustar[col] = full_nustar[col].apply(lambda x: x.strip())

        # Important first step, select only 'science mode' observations, slew observations will be dealt with in
        #  another class - this includes excluding observations with 'STELLAR' spacecraft mode, as they are likely
        #  rotating. Also select only those observations which have actually been taken (this table contains
        #  planned observations as well).
        rel_nustar = full_nustar[(full_nustar['OBSERVATION_MODE'] == 'SCIENCE') &
                                 (full_nustar['SPACECRAFT_MODE'] == 'INERTIAL') &
                                 (full_nustar['STATUS'].isin(['processed', 'archived']))]

        # Lower-casing all the column names (personal preference largely).
        rel_nustar = rel_nustar.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects
        rel_nustar = rel_nustar.rename(columns={'obsid': 'ObsID', 'time': 'start', 'end_time': 'end',
                                                'public_date': 'proprietary_end_date',
                                                'subject_category': 'target_category'})

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        rel_nustar['start'] = pd.to_datetime(Time(rel_nustar['start'].values, format='mjd', scale='utc').to_datetime())
        rel_nustar['end'] = pd.to_datetime(Time(rel_nustar['end'].values, format='mjd', scale='utc').to_datetime())
        # Then make a duration column by subtracting one from t'other - there are also exposure and ontime columns
        #  which I've acquired, but I think total duration is what I will go with here.
        rel_nustar['duration'] = rel_nustar['end']-rel_nustar['start']

        # Converting the exposure times and ontimes to Pandas time deltas
        for col in rel_nustar.columns[(rel_nustar.columns.str.contains('exposure')) |
                                      rel_nustar.columns.str.contains('ontime')]:
            rel_nustar[col] = pd.to_timedelta(rel_nustar[col], 's')

        # Slightly more complicated with the public release dates, as some of them are set to 0 MJD, which makes the
        #  conversion routine quite upset (0 is not valid) - as such I convert only those which aren't 0, then
        #  replace the 0 valued ones with Pandas' Not a Time (NaT) value
        val_end_dates = rel_nustar['proprietary_end_date'] != 0

        # We make a copy of the proprietary end dates to work on because pandas will soon not allow us to set what
        #  was previously an int column with a datetime - so we need to convert it but retain the original
        #  data as well
        prop_end_datas = rel_nustar['proprietary_end_date'].copy()
        # Convert the original column to datetime
        rel_nustar['proprietary_end_date'] = rel_nustar['proprietary_end_date'].astype('datetime64[ns]')
        rel_nustar.loc[val_end_dates, 'proprietary_end_date'] = pd.to_datetime(
            Time(prop_end_datas[val_end_dates.values], format='mjd', scale='utc').to_datetime())
        rel_nustar.loc[~val_end_dates, 'proprietary_end_date'] = pd.NaT
        # Grab the current date and time
        today = datetime.today()
        # Create a boolean column that describes whether the data are out of their proprietary period yet
        rel_nustar['proprietary_usable'] = rel_nustar['proprietary_end_date'].apply(lambda x:
                                                                                    ((x <= today) &
                                                                                     (pd.notnull(x)))).astype(bool)

        # I was going to use the 'issue_flag' column as a way of deciding scientific viability, but over 1500
        #  observations are marked '1' (for an issue) and I don't really want to exclude that many out of hand so
        #  I will just make everything scientifically usable for now.
        rel_nustar['science_usable'] = True

        # Convert the categories of target that are present in the dataframe to the DAXA taxonomy
        conv_dict = {'Active galaxies and Quasars': 'AGN', 'Non-Proposal ToOs': 'TOO',
                     'Galactic Compact Sources': 'GS', 'Solar System Objects': 'SOL',
                     'Proposed ToOs and Directors Discretionary Time': 'TOO', 'Calibration Observations': 'CAL',
                     'Non-ToO Supernovae, Supernova Remnants, and Galactic diffuse': 'SNR', 'Normal galaxies': 'NGS',
                     'Galaxy clusters and extragalactic diffuse objects': 'GCL', 'Non-Pointing data': 'MISC'}

        # I don't want to assume that the descriptions I've seen looking at the whole NuSTAR master list as it is now
        #  is how it will stay forever, as such I construct a mask that tells me which entries have a recognised
        #  description - any that don't will be set to the 'MISC' code
        type_recog = rel_nustar['target_category'].isin(list(conv_dict.keys()))
        # The recognized target category descriptions are converted to DAXA taxonomy
        rel_nustar.loc[type_recog, 'target_category'] = rel_nustar.loc[type_recog, 'target_category'].apply(
            lambda x: conv_dict[x])
        # Now I set any unrecognized target category descriptions to MISC - there are none at the time of writing,
        #  but that could well change
        rel_nustar.loc[~type_recog, 'target_category'] = 'MISC'

        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_nustar = rel_nustar[['ra', 'dec', 'ObsID', 'science_usable', 'proprietary_usable', 'start', 'end',
                                 'duration', 'proprietary_end_date', 'target_category', 'exposure_a', 'exposure_b',
                                 'ontime_a', 'ontime_b', 'nupsdout', 'issue_flag']]

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_nustar = rel_nustar.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_nustar

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], raw_dir: str, download_products: bool):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param List[str] insts: The instruments which the user wishes to acquire data for.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        :param bool download_products: This controls whether the data downloaded include the pre-processed event lists
            and images stored by HEASArc, or whether they are the original raw event lists. Default is to download
            raw data.
        """
        if download_products:
            req_dir = REQUIRED_DIRS['processed']
        else:
            req_dir = REQUIRED_DIRS['raw']

        # This two digit code identifies the program type (00 assigned to the first 2-year primary mission, and
        #  then 01, 02, 03 ... increment for each additional year of observations. Useful here to get to the
        #  correct directory to find our ObsID
        prog_id = observation_id[1:3]

        # This identifies the type of source that was being observed, useful here to get to the right directory
        src_cat = observation_id[0]

        # This is the path to the HEASArc data directory for this ObsID
        obs_dir = "/FTP/nustar/data/obs/{pid}/{sc}/{oid}/".format(pid=prog_id, sc=src_cat, oid=observation_id)
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the three directories that I need to download unprocessed NuStar data are present
        top_data = [en['href'] for en in BeautifulSoup(session.get(top_url).text, "html.parser").find_all("a")
                    if en['href'] in req_dir]
        # If the lengths of top_data and REQUIRED_DIRS are different, then one or more of the expected dirs
        #  is not present
        if len(top_data) != len(req_dir):
            # This list comprehension figures out what directory is missing and reports it
            missing = [rd for rd in req_dir if rd not in top_data]
            raise FileNotFoundError("The archive data directory for {o} does not contain the following required "
                                    "directories; {rq}".format(o=observation_id, rq=", ".join(missing)))

        for dat_dir in top_data:
            # The lower level URL of the directory we're currently looking at
            rel_url = top_url + dat_dir + '/'
            # This is the directory to which we will be saving this archive directories files
            local_dir = raw_dir + '/' + dat_dir + '/'
            # Make sure that the local directory is created
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            # We explore the contents of said directory, making sure to clean any useless HTML guff left over - these
            #  are the files we shall be downloading
            to_down = [en['href'] for en in BeautifulSoup(session.get(rel_url).text, "html.parser").find_all("a")
                       if '?' not in en['href'] and obs_dir not in en['href']]

            # As we allow the user to select a single instrument, if they don't want both (though goodness knows why
            #  on earth anyone would do that), the event_uf/cl directory gets an extra check. The last character of
            #  the instrument is either A or B, and that is what I am using to identify the relevant event lists.
            if dat_dir == 'event_uf/':
                to_down = [td for td in to_down for inst in insts if observation_id+inst[-1]+"_uf" in td]
            elif dat_dir == 'event_cl/':
                to_down = [td for td in to_down for inst in insts if observation_id+inst[-1] in td]

            for down_file in to_down:
                down_url = rel_url + down_file
                with session.get(down_url, stream=True) as acquiro:
                    with open(local_dir + down_file, 'wb') as writo:
                        copyfileobj(acquiro.raw, writo)

                # There are a few compressed fits files in each archive, but I think I'm only going to decompress the
                #  event lists, as they're more likely to be used
                if 'evt.gz' in down_file or 'img.gz' in down_file:
                    # Open and decompress the events file
                    with gzip.open(local_dir + down_file, 'rb') as compresso:
                        # Open a new file handler for the decompressed data, then funnel the decompressed events there
                        with open(local_dir + down_file.split('.gz')[0], 'wb') as writo:
                            copyfileobj(compresso, writo)
                    # Then remove the tarred file to minimise storage usage
                    os.remove(local_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, credentials: Union[str, dict] = None,
                 download_products: bool = True):
        """
        A method to acquire and download the pointed NuSTAR data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param dict/str credentials: The path to an ini file containing credentials, a dictionary containing 'user'
            and 'password' entries, or a dictionary of ObsID top level keys, with 'user' and 'password' entries
            for providing different credentials for different observations.
        :param bool download_products: This controls whether the data downloaded include the pre-processed event lists
            and images stored by HEASArc, or whether they are the original raw event lists. Default is True.
        """

        if credentials is not None and not self.filtered_obs_info['proprietary_usable'].all():
            raise NotImplementedError("Support for credentials for proprietary data is not yet implemented.")
        elif not self.filtered_obs_info['proprietary_usable'].all() and credentials is None:
            warn("Proprietary data have been selected, but no credentials provided; as such the proprietary data have "
                 "been excluded from download and further processing.", stacklevel=2)
            new_filter = self.filter_array * self.all_obs_info['proprietary_usable'].values
            self.filter_array = new_filter

        # Ensures that a directory to store the 'raw' pointed NuSTAR data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.top_level_path + self.name + '_raw'):
            os.makedirs(self.top_level_path + self.name + '_raw')
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
        if all([os.path.exists(stor_dir + '{o}'.format(o=o)) for o in self.filtered_obs_ids]):
            self._download_done = True

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in self.filtered_obs_ids:
                        # Use the internal static method I set up which both downloads and unpacks the NuSTAR data
                        self._download_call(obs_id, insts=self.chosen_instruments,
                                            raw_dir=stor_dir + '{o}'.format(o=obs_id),
                                            download_products=download_products)
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
                                               'raw_dir': stor_dir + '{o}'.format(o=obs_id),
                                               'download_products': download_products},
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
            warn("The raw data for this mission have already been downloaded.", stacklevel=2)

    def assess_process_obs(self, obs_info: dict):
        """
        A slightly unusual method which will allow the NuSTARPointed mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        raise NotImplementedError("The check_process_obs method has not yet been implemented for NuSTARPointed, as "
                                  "we need to see what detailed information are available once processing downloaded "
                                  "data has begun.")

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
        # raise NotImplementedError("The check_process_obs method has not yet been implemented for {n}, as it isn't yet"
        #                           "clear to me what form the unique identifiers will take once we start processing"
        #                           "{n} data ourselves.".format(n=self.pretty_name))
        # NuSTAR ObsIDs are always 11 digits, so we just retrieve the first 11
        return ident[:11]
