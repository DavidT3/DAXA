#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 24/04/2024, 10:27. Copyright (c) The Contributors

import gzip
import io
import os
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
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError, PreProcessedNotSupportedError
from daxa.mission.base import BaseMission

# There are a lot of directories in the ASCA archives it would seem. There also isn't a great explanation of them
#  that I can find anywhere. This is the best I found:
#  https://heasarc.gsfc.nasa.gov/docs/asca/abc/node3.html#SECTION00350000000000000000
# I am very much erring on the side of caution and downloading more rather than less here
REQUIRED_DIRS = {'raw': ['aux/', 'calib/', 'telem/', 'unscreened/'],
                 'processed': ['aux/', 'calib/', 'telem/', 'unscreened/', 'images/', 'spectra/', 'lcurves/',
                               'screened/']}


class ASCA(BaseMission):
    """
    The mission class for ASCA observations, both from the GIS AND SIS instruments.
    The available observation information is fetched from the HEASArc ASCAMASTER table, and data are downloaded from
    the HEASArc https access to their FTP server.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include SIS0, SIS1, GIS2, and GIS3 (the
            default is all of them).
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
        The mission class for ASCA observations, both from the GIS AND SIS instruments.
        The available observation information is fetched from the HEASArc ASCAMASTER table, and data are downloaded from
        the HEASArc https access to their FTP server.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
                pass either a single string value or a list of strings. They may include SIS0, SIS1, GIS2, and GIS3 (the
                default is all of them).
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        super().__init__()

        # Sets the default instruments - all the instruments on ASCA
        if insts is None:
            insts = ['SIS0', 'SIS1', 'GIS2', 'GIS3']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - they all have their own telescopes
        self._miss_poss_insts = ['SIS0', 'SIS1', 'GIS2', 'GIS3']
        # The chosen_instruments property setter (see below) will use these to convert possible contractions
        #  to the names that the module expects. I'm not that familiar with ASCA currently, so
        #  I've just put in X0, X1, ... without any real expectation that anyone would use them.
        self._alt_miss_inst_names = {'S0': 'SIS0', 'S1': 'SIS1', 'G2': 'GIS2', 'G3': 'GIS3'}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for ROSAT images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        # GIS all -> 0-1023 PI, LO -> 0-170 PI, HI -> 170-1024 PI
        # SIS all -> 0-2047 PI, LO -> 0-547 PI, HI -> 547-2048 PI
        # Apparently low is below 2 keV, high is above 2 keV, all is 'everything', might choose 0.4-10 keV for
        #  SIS, and 0.7-10 keV for GIS
        self._template_en_trans = {"SIS0": {Quantity(0.4, 'keV'): {Quantity(2, 'keV'): 'lo',
                                                                   Quantity(10.0, 'keV'): 'all'},
                                            Quantity(2, 'keV'): {Quantity(10.0, 'keV'): 'hi'}},
                                   "SIS1": {Quantity(0.4, 'keV'): {Quantity(2, 'keV'): 'lo',
                                                                   Quantity(10.0, 'keV'): 'all'},
                                            Quantity(2, 'keV'): {Quantity(10.0, 'keV'): 'hi'}},
                                   "GIS2": {Quantity(0.7, 'keV'): {Quantity(2, 'keV'): 'lo',
                                                                   Quantity(10.0, 'keV'): 'all'},
                                            Quantity(2, 'keV'): {Quantity(10.0, 'keV'): 'hi'}},
                                   "GIS3": {Quantity(0.7, 'keV'): {Quantity(2, 'keV'): 'lo',
                                                                   Quantity(10.0, 'keV'): 'all'},
                                            Quantity(2, 'keV'): {Quantity(10.0, 'keV'): 'hi'}},
                                   }
        # This does strip the specific instrument number from the name, which doesn't work for the event lists
        #  because they are provided individually, but it DOES work for the images and exposure maps as they exist
        #  as combinations of the paired instruments. That means we only have to override the get_evt_list_path method
        self._template_inst_trans = {'GIS2': 'gis', 'GIS3': 'gis', 'SIS0': 'sis', 'SIS1': 'sis'}

        # We set up the ROSAT file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        # TODO This isn't completely adequate - there can be different bit rates (h, l, m), AND INDEED MULTIPLE
        #  sub-exposures of the same bit rate!
        self._template_evt_name = "screened/ad{oi}{i}*h.evt"
        self._template_img_name = "images/ad{oi}{i}*_{eb}.totsky"
        self._template_exp_name = "images/ad{oi}{i}*.totexpo"
        self._template_bck_name = None

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['target_category', 'sis_exposure', 'gis_exposure']

        # Runs the method which fetches information on all available ASCA observations and stores that
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
        self._miss_name = "asca"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "ASCA"
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
        #  the BaseMission superclass - ASCA observations seem to have a unique 8-digit ObsID, though I can find
        #  no discussion of whether there is extra information in the ObsID (i.e. target type).
        self._id_format = '^[0-9]{8}$'
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
        # The instruments have different field of views but do observe simultaneously.
        # Used information from https://heasarc.gsfc.nasa.gov/docs/asca/asca.html
        # SIS instruments had a square 22x22', and GIS instruments had a 50' diameter circular FoV
        self._approx_fov = {'SIS0': Quantity(11, 'arcmin'), 'SIS1': Quantity(11, 'arcmin'),
                            'GIS2': Quantity(25, 'arcmin'), 'GIS3': Quantity(25, 'arcmin')}
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
        credit) to acquire the 'ascamaster' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available ASCA observations, with important information such as pointing coordinates, ObsIDs, and exposure.
        """
        # This is the web interface for querying NASA HEASArc catalogues
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        # This returns the requested information in a FITS format - the idea being I will stream this into memory
        #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited
        result_max = "&ResultMax=0"
        # This just tells the interface it's a query (I think?)
        action = "&Action=Query"
        # Tells the interface that I want to retrieve from the ascamaster (ASCA Master) catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20ascamaster"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/asca/ascamaster.html)
        # All the proprietary periods for ASCA data have passed, so we don't need to download them at this point
        #  like we do with some other missions
        which_cols = ['RA', 'DEC', 'SEQUENCE_NUMBER', 'TIME', 'END_TIME', 'Subject_Category', 'Status', 'GIS_EXPOSURE',
                      'SIS_EXPOSURE']
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
                full_asca = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_asca.select_dtypes(['object']).columns:
                    full_asca[col] = full_asca[col].apply(lambda x: x.strip())

        # Important first step, making any global cuts to the dataframe to remove entries that are not going to be
        #  useful. For ASCA I have elected to remove any ObsID with zero exposure SIS AND GIS exposure
        rel_asca = full_asca[(full_asca['SIS_EXPOSURE'] != 0.0) | (full_asca['GIS_EXPOSURE'] != 0.0)]
        # We throw a warning that some number of the ASCA observations are dropped because it doesn't seem that they
        #  will be at all useful
        if len(rel_asca) != len(full_asca):
            warn("{ta} of the {tot} observations located for ASCA have been removed due to all instrument exposures "
                 "being zero.".format(ta=len(full_asca) - len(rel_asca), tot=len(full_asca)), stacklevel=2)

        # Lower-casing all the column names (personal preference largely).
        rel_asca = rel_asca.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects
        rel_asca = rel_asca.rename(columns={'sequence_number': 'ObsID', 'time': 'start', 'end_time': 'end',
                                            'subject_category': 'target_category'})

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        rel_asca['start'] = pd.to_datetime(Time(rel_asca['start'].values.astype(float), format='mjd',
                                                scale='utc').to_datetime())
        rel_asca['end'] = pd.to_datetime(Time(rel_asca['end'].values.astype(float), format='mjd',
                                              scale='utc').to_datetime())
        # Then make a duration column by subtracting one from t'other - there are also exposure and ontime columns
        #  which I've acquired, but I think total duration is what I will go with here.
        rel_asca['duration'] = rel_asca['end'] - rel_asca['start']

        # Converting the exposure times to Pandas time deltas
        for col in rel_asca.columns[rel_asca.columns.str.contains('exposure')]:
            rel_asca[col] = pd.to_timedelta(rel_asca[col], 's')

        # No clear way of defining this from the tables, so we're going to assume that they all are
        rel_asca['science_usable'] = True

        # print(rel_asca['target_category'].value_counts().index.values)
        # stop

        # Convert the categories of target that are present in the dataframe to the DAXA taxonomy
        # The ASCA categories are here:
        #  https://heasarc.gsfc.nasa.gov/W3Browse/asca/ascamaster.html#Subject_category
        # These translations are pretty hand-wavey honestly
        conv_dict = {'AGN': 'AGN',
                     'SUPERNOVA REMNANTS AND GALACTIC DIFFUSE EMISSION': 'SNR',
                     'CLUSTERS OF GALAXIES AND SUPERCLUSTERS': 'GCL',
                     'STARS': 'GS',
                     'NORMAL STARS': 'GS',
                     'X RAY BINARIES': 'XRB',
                     'COSMIC X-RAY BACKGROUND AND DEEP SURVEYS AND OTHER': 'EGS',
                     'NORMAL GALAXIES': 'NGS',
                     'CATACLYSMIC VARIABLES': 'CV'}

        # I construct a mask that tells me which entries have a recognised description - any that don't will be set
        #  to the 'MISC' code
        type_recog = rel_asca['target_category'].isin(list(conv_dict.keys()))
        # The recognized target category descriptions are converted to DAXA taxonomy
        rel_asca.loc[type_recog, 'target_category'] = rel_asca.loc[type_recog, 'target_category'].apply(
            lambda x: conv_dict[x])
        # Now I set any unrecognized target category descriptions to MISC - there are none at the time of writing,
        #  but that could well change
        rel_asca.loc[~type_recog, 'target_category'] = 'MISC'
        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_asca = rel_asca[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration', 'target_category',
                             'sis_exposure', 'gis_exposure']]

        # As it stands, the ObsID column is an integer, but we want them as strings!
        rel_asca['ObsID'] = rel_asca['ObsID'].astype(str)

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_asca = rel_asca.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_asca

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
        insts = [inst.lower() for inst in insts]

        if download_products:
            req_dir = REQUIRED_DIRS['processed']
        else:
            req_dir = REQUIRED_DIRS['raw']

        # This is the path to the HEASArc data directory for this ObsID
        obs_dir = "/FTP/asca/data/rev2/{oid}/".format(oid=observation_id)
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the three directories that I need to download unprocessed ASCA data are present
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

            # As we allow the user to select a single instrument we have to check event list and product files
            #  for matching names, so we only download for those instruments the user has chosen
            if dat_dir in ['unscreened/', 'spectra/', 'lcurves/', 'screened/']:
                # Unhelpfully some of the products have shortened versions of the individual instrument
                #  IDs in the file names
                short_inst = [inst[0]+inst[-1] for inst in insts]
                to_down = [td for td in to_down for inst in short_inst if observation_id + inst in td]
            elif dat_dir == 'images/':
                # Unhelpfully some of the products don't have the individual instrument IDs in the names, just the
                #  'GIS' or 'SIS' type
                just_inst_type = list(set([inst[:-1] for inst in insts]))
                to_down = [td for td in to_down for inst in just_inst_type if observation_id + inst in td]

            for down_file in to_down:
                down_url = rel_url + down_file
                with session.get(down_url, stream=True) as acquiro:
                    with open(local_dir + down_file, 'wb') as writo:
                        copyfileobj(acquiro.raw, writo)

                # There are a few compressed fits files in each archive, but I think I'm only going to decompress the
                #  event lists, as they're more likely to be used
                if 'evt.gz' in down_file or 'totsky.gz' in down_file or 'totexpo.gz' in down_file:
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
        A method to acquire and download the ASCA data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: This controls whether the data downloaded include the pre-processed event lists
            and images stored by HEASArc, or whether they are the original raw event lists. Default is True.
        """

        # Ensures that a directory to store the 'raw' ASCA data in exists - once downloaded and unpacked
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
                        # Use the internal static method I set up which both downloads and unpacks the ASCA data
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
            warn("The raw data for this mission have already been downloaded.")

    def get_evt_list_path(self, obs_id: str, inst: str = None) -> str:
        """
        A get method that provides the path to a downloaded pre-generated event list for the current mission (if
        available). This method will not work if pre-processed data have not been downloaded.

        :param str obs_id: The ObsID of the event list.
        :param str inst: The instrument of the event list (if applicable).
        :return: The requested event list path.
        :rtype: str
        """

        inst, en_bnd_trans, file_inst, lo_en, hi_en = self._get_prod_path_checks(obs_id, inst)

        # The reason we needed to override the base get method for event list path - the naming scheme for event lists
        #  is different from images + exposure maps, as those are shipped with SIS0+1 and GIS2+3 added together.
        if file_inst == 'sis':
            file_inst = 's' + inst[-1]
        elif file_inst[0] == 'gis':
            file_inst = 'g' + inst[-1]

        # The template path can take two forms, one is a straight string and can just be filled in, but the
        #  other is a dictionary where the keys are instrument names and the values are the string file templates. We
        #  need to check which is applicable to this mission and treat it accordingly
        if isinstance(self._template_evt_name, str):
            rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_evt_name.format(oi=obs_id, i=file_inst))
        # In some cases the instrument name will have to be supplied, otherwise we will not be able to
        #  create a path
        elif isinstance(self._template_evt_name, dict) and inst is None:
            raise ValueError("The 'inst' argument cannot be None for this mission, as the different instruments have "
                             "differently formatted pre-processed file names.")
        # It is possible for only some instruments of a mission to have images, so we check
        elif isinstance(self._template_evt_name, dict) and self._template_evt_name[inst] is None:
            raise PreProcessedNotSupportedError("This mission ({m}) does not support the download of pre-processed "
                                                "event lists for the {i} instrument, so a path cannot be "
                                                "provided.".format(m=self.pretty_name, i=inst))
        elif isinstance(self._template_evt_name, dict):
            rel_pth = os.path.join(self.raw_data_path, obs_id, self._template_evt_name[inst].format(oi=obs_id,
                                                                                                    i=file_inst))

        # This performs certain checks to make sure the file exists, and fill in any wildcards
        rel_pth = self._get_prod_path_post_checks(rel_pth, obs_id, inst, 'event list')

        return rel_pth

    def assess_process_obs(self, obs_info: dict):
        """
        A slightly unusual method which will allow the ASCA mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        raise NotImplementedError("The check_process_obs method has not yet been implemented for ASCA, as "
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
        # All ASCA ObsIDs are 8 digits, so that is what we select
        return ident[:8]
