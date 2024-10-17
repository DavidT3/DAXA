#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 24/09/2024, 17:00. Copyright (c) The Contributors

import gzip
import io
import os
from multiprocessing import Pool
from shutil import copyfileobj
from typing import List, Union, Any
from warnings import warn

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from bs4 import BeautifulSoup
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission

# The Swift directories have their data split into directories named the same as the instruments (i.e. uvot, xrt,
#  bat) - those will be added to the required dirs list at the download phase, depending on what the user has selected.
# Not all the directories will always be present, as it depends on observing modes etc.
# I'm using this - https://swift.gsfc.nasa.gov/archive/archiveguide1/node5.html#SECTION00521000000000000000 - guide
#  to determine which directories are needed
# Some directories (e.g. events) hold the cleaned and uncleaned event lists, so download method filtering of files
#  will have to be done there
REQUIRED_DIRS = {'all': ['auxil/'],
                 'raw': {'uvot': ['hk/', 'event/', 'image/'],
                         'xrt': ['event/', 'hk/', 'image/'],
                         'bat': ['event/', 'rate/', 'survey/']},
                 'processed': {'uvot': ['hk/', 'event/', 'image/', 'products/'],
                               'xrt': ['event/', 'hk/', 'image/', 'products/'],
                               'bat': ['event/', 'rate/', 'survey/']}}


class Swift(BaseMission):
    """
    The mission class for observations by the Neil Gehrels Swift Observatory.
    The available observation information is fetched from the HEASArc SWIFTMASTR table, and data are downloaded from
    the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include XRT, BAT, and UVOT (the default
        is both XRT and BAT).
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
        The mission class for observations by the Neil Gehrels Swift Observatory.
        The available observation information is fetched from the HEASArc SWIFTMASTR table, and data are downloaded
        from the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include XRT, BAT, and UVOT (the default
            is both XRT and BAT).
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        super().__init__()

        # Sets the default instruments - the two X-ray (though BAT sort of tends towards low energy gamma rays as
        #  well) instruments on Swift.
        # TODO decide whether UV data should be acquired as default considering this module focuses on X-rays
        if insts is None:
            insts = ['XRT', 'BAT']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - Swift has a focusing X-ray telescope (XRT), the burst
        #  alert telescope (BAT) which observes in the hard X-ray (15-150keV) and up to 500keV for non-imaging
        #  studies, and a UV telescope very similar to the optical monitor on XMM (but designed better).
        self._miss_poss_insts = ['XRT', 'BAT', 'UVOT']
        # As far as I know there aren't any other common names for the instruments on Swift
        self._alt_miss_inst_names = {}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for ROSAT images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        self._template_en_trans = {'XRT': {Quantity(0.01, 'keV'): {Quantity(10.23, 'keV'): ""}},
                                   'BAT': None,
                                   'UVOT': {Quantity(0, 'keV'): {Quantity(0, 'keV'): ""}}}
        self._template_inst_trans = None

        # We set up the ROSAT file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        # I added wildcards before the ObsID (and I hope this isn't going to break things) because irritatingly they
        #  fill in zeroes before shorted ObsIDs I think - could add that functionality to the general get methods #
        #  but this could be easier
        self._template_evt_name = {'XRT': "xrt/event/sw{oi}xpc*po_cl.evt", "UVOT": None,
                                   'BAT': "sw{oi}msbevshsp uf.evt"}
        self._template_img_name = {'XRT': "xrt/products/sw{oi}xpc_sk.img", "UVOT": "uvot/products/sw{oi}u_sk.img",
                                   "BAT": None}
        self._template_exp_name = {'XRT': "xrt/products/sw{oi}xpc_ex.img", "UVOT": "uvot/products/sw{oi}u_ex.img",
                                   "BAT": None}
        self._template_bck_name = None

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['target_category', 'xrt_exposure', 'bat_exposure', 'uvot_exposure']

        # Runs the method which fetches information on all available Swift observations and stores that
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
        self._miss_name = "swift"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "Swift"
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
        #  the BaseMission superclass - Swift observations have a unique 11-digit ObsID, the construction of
        #  which is discussed here (https://heasarc.gsfc.nasa.gov/w3browse/swift/swiftmastr.html#ObsID)
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
        #  new mission class. XRT is described here (https://swift.gsfc.nasa.gov/about_swift/xrt_desc.html),
        #  UVOT is described here (https://swift.gsfc.nasa.gov/about_swift/uvot_desc.html), and BAT is described
        #  here (https://swift.gsfc.nasa.gov/about_swift/bat_desc.html).
        # BAT is somewhat complicated, because the half-coded region (which can do imaging) has a 100x60deg FoV, so I
        #  have gone with half the long side
        self._approx_fov = {'XRT': Quantity(11.8, 'arcmin'), 'BAT': Quantity(50, 'deg'),
                            'UVOT': Quantity(8.5, 'arcmin')}
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
        credit) to acquire the 'swiftmastr' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available Swift observations, with important information such as pointing coordinates, ObsIDs, and exposure.
        """
        # This is the web interface for querying NASA HEASArc catalogues
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        # This returns the requested information in a FITS format - the idea being I will stream this into memory
        #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited, as we don't know how many Swift observations there are, and the number will
        #  increase with time (so long as the telescope doesn't break...)
        result_max = "&ResultMax=0"
        # This just tells the interface it's a query (I think?)
        action = "&Action=Query"
        # Tells the interface that I want to retrieve from the swiftmastr (Swift Master) catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20swiftmastr"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/w3browse/swift/swiftmastr.html)

        # Could poll the individual instrument observing times for different modes to determine the mode of the
        #  instrument - but that would be a LOT of columns
        which_cols = ['RA', 'DEC', 'Roll_Angle', 'ObsID', 'Start_Time', 'Stop_Time', 'XRT_Exposure', 'UVOT_Exposure',
                      'BAT_Exposure']
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
                full_swift = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_swift.select_dtypes(['object']).columns:
                    full_swift[col] = full_swift[col].apply(lambda x: x.strip())

        # Important first step, making any global cuts to the dataframe to remove entries that are not going to be
        #  useful. For Swift, as information in the table is pretty limited, I have elected to remove any ObsID with
        #  zero exposure in all three instruments - further cuts may be made later.
        rel_swift = full_swift[(full_swift['XRT_EXPOSURE'] != 0.0) | (full_swift['BAT_EXPOSURE'] != 0.0) |
                               (full_swift['UVOT_EXPOSURE'] != 0.0)]
        # We throw a warning that some number of the Swift observations are dropped because it doesn't seem that they
        #  will be at all useful
        if len(rel_swift) != len(full_swift):
            warn("{ta} of the {tot} observations located for Swift have been removed due to all instrument exposures "
                 "being zero.".format(ta=len(full_swift)-len(rel_swift), tot=len(full_swift)), stacklevel=2)

        # This removes any ObsIDs that have zero exposure time for the currently selected instruments - if all three
        #  instruments are selected then this won't do anything because it is the same as what we did a couple of
        #  lines up - but if a subset have been selected then it might well do something
        pre_inst_exp_check_num = len(rel_swift)
        rel_swift = rel_swift[np.logical_or.reduce([rel_swift[inst+'_EXPOSURE'] != 0
                                                    for inst in self.chosen_instruments])]
        # I warn the user if their chosen instruments have observations that have been removed because the chosen
        #  instruments are all zero exposure
        if len(rel_swift) != pre_inst_exp_check_num:
            warn("{ta} of the {tot} observations located for Swift have been removed due to all chosen instrument "
                 "({ci}) exposures being zero.".format(ta=pre_inst_exp_check_num-len(rel_swift), tot=len(full_swift),
                                                       ci=", ".join(self.chosen_instruments)), stacklevel=2)

        # Lower-casing all the column names (personal preference largely).
        rel_swift = rel_swift.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects
        rel_swift = rel_swift.rename(columns={'obsid': 'ObsID', 'start_time': 'start', 'stop_time': 'end'})

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        rel_swift['start'] = pd.to_datetime(Time(rel_swift['start'].values.astype(float), format='mjd',
                                                 scale='utc').to_datetime())
        rel_swift['end'] = pd.to_datetime(Time(rel_swift['end'].values.astype(float), format='mjd',
                                               scale='utc').to_datetime())
        # Then make a duration column by subtracting the start MJD from the end MJD - not an exposure time but just
        #  how long the observation window was
        rel_swift['duration'] = rel_swift['end'] - rel_swift['start']
        # Cycle through the exposure columns and make sure that they are time objects
        for col in rel_swift.columns[rel_swift.columns.str.contains('exposure')]:
            rel_swift[col] = pd.to_timedelta(rel_swift[col], 's')

        # Not really much information in the table that I can use to decide how to populate this
        rel_swift['science_usable'] = True

        # There are no Swift target categories in the catalogue unfortunately
        # The recognized target category descriptions are converted to DAXA taxonomy
        rel_swift['target_category'] = 'MISC'

        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_swift = rel_swift[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration',  'target_category',
                               'xrt_exposure', 'bat_exposure', 'uvot_exposure', 'roll_angle']]

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_swift = rel_swift.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_swift

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], start_year: str, start_month: str, raw_dir: str,
                       download_products: bool):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param List[str] insts: The instruments which the user wishes to acquire data for.
        :param str start_year: The start year of the observation to be downloaded - this is necessary as
            the HEASArc Swift data are split into yyyy-mm directories.
        :param str start_month: The start month of the observation to be downloaded - this is necessary as
            the HEASArc Swift data are split into yyyy-mm directories.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        :param bool download_products: This controls whether the data downloaded include the pre-processed event lists
            and images stored by HEASArc, or whether they are the original raw event lists. Default is to download
            raw data.
        """
        insts = [inst.lower() for inst in insts]
        req_dir = REQUIRED_DIRS['all'] + [inst + '/' for inst in insts]
        if download_products:
            dir_lookup = REQUIRED_DIRS['processed']
        else:
            dir_lookup = REQUIRED_DIRS['raw']

        # The data on HEASArc are stored in subdirectories named after the year-month that they were taken, so
        #  we first need to construct that to setup the URL we need
        date_id = start_year + '_' + start_month.zfill(2)

        # This is the path to the HEASArc data directory for this ObsID
        obs_dir = "/FTP/swift/data/obs/{did}/{oid}/".format(did=date_id, oid=observation_id)
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the directories that I need to download unprocessed Swift data are present
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
            rel_url = top_url + dat_dir
            # This is the directory to which we will be saving this archive directories files
            local_dir = raw_dir + '/' + dat_dir
            # Make sure that the local directory is created
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            if dat_dir == 'auxil/':
                # All the files in the auxiliary directory are downloaded
                to_down = [en['href'] for en in BeautifulSoup(session.get(rel_url).text, "html.parser").find_all("a")
                           if '?' not in en['href'] and obs_dir not in en['href']]
            else:
                # The way the Swift archives are laid out, each instrument directory has subdirectories that
                #  we need to decide whether to download or not - we also need to make some distinctions on
                #  a file level (i.e. cleaned event lists won't be downloaded if the user doesn't want to download
                #  pre-processed data).
                rel_req_dir = dir_lookup[dat_dir[:-1]]
                to_down = []
                # Here we cycle through the directories that we have found at the instrument URL for this ObsID
                for en in BeautifulSoup(session.get(rel_url).text, "html.parser").find_all("a"):
                    # This is what happens in most cases for a genuine directory - we don't deal with any
                    #  directory named 'event' here though
                    if '?' not in en['href'] and obs_dir not in en['href'] and en['href'] in rel_req_dir and \
                            en['href'] != 'event/':
                        low_rel_url = rel_url + en['href']
                        files = [en['href'] + '/' + fil['href'] for fil in BeautifulSoup(session.get(low_rel_url).text,
                                                                                         "html.parser").find_all("a")
                                 if '?' not in fil['href'] and obs_dir not in fil['href']]
                    # 'event' directories get their own treatment because we have to decide whether to download
                    #  cleaned event lists or not, depending whether the user has requested them
                    elif '?' not in en['href'] and obs_dir not in en['href'] and en['href'] in rel_req_dir and \
                            en['href'] == 'event/':
                        low_rel_url = rel_url + en['href']
                        files = [en['href'] + '/' + fil['href'] for fil in BeautifulSoup(session.get(low_rel_url).text,
                                                                                         "html.parser").find_all("a")
                                 if '?' not in fil['href'] and obs_dir not in fil['href'] and
                                 ('cl.evt' not in fil['href'] or download_products)]
                    else:
                        files = []

                    # If the current subdirectory of the current instrument of the current ObsID has got files that
                    #  we want to download, then we make sure that the subdirectory exists locally
                    if len(files) != 0 and not os.path.exists(local_dir + en['href']):
                        os.makedirs(local_dir + en['href'])
                    # And add the current list of files to the overall downloading list for this instrument
                    to_down += files

            # Now we cycle through the files and download them
            for down_file in to_down:
                down_url = rel_url + down_file
                with session.get(down_url, stream=True) as acquiro:
                    with open(local_dir + down_file, 'wb') as writo:
                        copyfileobj(acquiro.raw, writo)

                # There are a few compressed fits files in each archive, but I think I'm only going to decompress the
                #  event lists, as they're more likely to be used
                if 'evt.gz' in down_file or 'img.gz':
                    # Open and decompress the events file
                    with gzip.open(local_dir + down_file, 'rb') as compresso:
                        # Open a new file handler for the decompressed data, then funnel the decompressed events there
                        with open(local_dir + down_file.split('.gz')[0], 'wb') as writo:
                            copyfileobj(compresso, writo)
                    # Then remove the tarred file to minimise storage usage
                    os.remove(local_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, download_products: bool = True):
        """
        A method to acquire and download the Swift data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: This controls whether the data downloaded include the pre-processed event lists
            and images stored by HEASArc, or whether they are the original raw event lists. Default is True.
        """

        # Ensures that a directory to store the 'raw' Swift data in exists - once downloaded and unpacked
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

        # We store the type of data that was downloaded
        if download_products:
            self._download_type = "raw+preprocessed"
        else:
            self._download_type = "raw"

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for row_ind, row in self.filtered_obs_info.iterrows():
                        obs_id = row['ObsID']
                        # While the user may have chosen multiple instruments, it is possible for Swift to have an
                        #  instrument switched off for a given observation, in which case an expected instrument
                        #  directory that the internal download method checks for will be missing. As such we ensure
                        #  that we only request instruments that have a non-zero exposure
                        rel_insts = [ci for ci in self.chosen_instruments
                                     if row[ci.lower() + '_exposure'].total_seconds() != 0]

                        # Use the internal static method I set up which both downloads and unpacks the Swift data
                        self._download_call(obs_id, insts=rel_insts, start_year=str(row['start'].year),
                                            start_month=str(row['start'].month),
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
                    for row_ind, row in self.filtered_obs_info.iterrows():
                        obs_id = row['ObsID']

                        # While the user may have chosen multiple instruments, it is possible for Swift to have an
                        #  instrument switched off for a given observation, in which case an expected instrument
                        #  directory that the internal download method checks for will be missing. As such we ensure
                        #  that we only request instruments that have a non-zero exposure
                        rel_insts = [ci for ci in self.chosen_instruments
                                     if row[ci.lower() + '_exposure'].total_seconds() != 0]

                        # Add each download task to the pool
                        pool.apply_async(self._download_call,
                                         kwds={'observation_id': obs_id, 'insts': rel_insts,
                                               'start_year': str(row['start'].year),
                                               'start_month': str(row['start'].month),
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
        A slightly unusual method which will allow the Swift mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        raise NotImplementedError("The check_process_obs method has not yet been implemented for Swift, as "
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
        # Swift ObsIDs are always 11 digits, so we just retrieve the first 11
        return ident[:11]
