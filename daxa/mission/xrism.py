#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/02/2025, 15:38. Copyright (c) The Contributors

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
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission

REQUIRED_DIRS = {'all': ['auxil/', 'resolve/', 'xtend/'],
                 'raw': {'resolve': ['event_uf/', 'hk/', 'products/'],
                         'xtend': ['event_uf/', 'hk/', 'products/']},
                 'processed': {'resolve': ['event_uf/', 'event_cl/', 'hk/', 'products/'],
                               'xtend': ['event_uf/', 'event_cl/', 'hk/', 'products/']}}


class XRISMPointed(BaseMission):
    """
    The mission class for pointed XRISM observations, supporting the acquisition of data from both the RESOLVE
    high-resolution spectrometer, and the XTEND wide-field CCD imager.
    The available observation information is fetched from the HEASArc XRISMMASTER table, and data are downloaded from
    the HEASArc https access to their FTP server.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include RESOLVE and XTEND.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
    The mission class for XRISM observations, supporting the acquisition of data from both the RESOLVE
    high-resolution spectrometer, and the XTEND wide-field CCD imager.
    The available observation information is fetched from the HEASArc XRISMMASTER table, and data are downloaded from
    the HEASArc https access to their FTP server.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include RESOLVE and XTEND.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
    """
        super().__init__()

        # Sets the default instruments - both instruments on XRISM (maybe the anti-coincidence detector will be
        #  added at some point)
        if insts is None:
            insts = ['RESOLVE', 'XTEND']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - both RESOLVE and XTEND have their own XRTs
        self._miss_poss_insts = ['RESOLVE', 'XTEND']
        # The chosen_instruments property setter (see below) will use these to convert possible contractions
        #  to the names that the module expects. Technically XTEND is the XMA + SXI
        self._alt_miss_inst_names = {}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for XRISM images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        self._template_en_trans = {'RESOLVE': {Quantity(1.7, 'keV'): {Quantity(12, 'keV'): ""}},
                                   'XTEND': {Quantity(0.4, 'keV'): {Quantity(20, 'keV'): ""}}}

        # TODO THIS MIGHT BE RELEVANT AGAIN AT SOME POINT - THESE ENERGY BANDS ARE FOR THE GIF PREVIEW IMAGES BUT
        #  THEY DON'T NECESSARILY HAVE FITS COUNTERPARTS
        # self._template_en_trans = {'RESOLVE': {Quantity(0.5, 'keV'): {Quantity(1, 'keV'): "0p5to1keV",
        #                                                                       Quantity(2, 'keV'): "0p5to2keV"},
        #                                                Quantity(2.0, 'keV'): {Quantity(10, 'keV'): "2to10keV"}},
        #                                    'XTEND': {Quantity(0.5, 'keV'): {Quantity(2, 'keV'): "0p5to2keV"},
        #                                              Quantity(2.0, 'keV'): {Quantity(10, 'keV'): "2to10keV"},
        #                                              Quantity(0.4, 'keV'): {Quantity(20, 'keV'): "2to10keV"}}}

        self._template_inst_trans = {}

        # We set up the XRISM file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        # The wild-carded string in XTEND event lists is the 'mode' of the detector
        # The RESOLVE 'px1000' part is the open-filter mode (all others are for calibration)
        self._template_evt_name = {'RESOLVE': 'resolve/event_cl/xa{oi}rsl_p0px1000_cl.evt.gz',
                                   'XTEND': 'xtend/event_cl/xa{oi}xtd_p*_cl.evt.gz'}
        self._template_img_name = "xis/products/ae{oi}{i}_*_sk.img"
        self._template_img_name = {'RESOLVE': 'resolve/products/xa{oi}rsl_p0px1000_cl.sky.img.gz',
                                   'XTEND': 'xtend/products/xa{oi}xtd_p*_cl.sky.img.gz'}

        self._template_exp_name = None
        self._template_bck_name = None

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['target_category', 'exposure', 'xtend_exposure', 'resolve_mode',
                                                'xtend_mode1-2', 'xtend_mode3-4', 'xtend_dataclass1-2',
                                                'xtend_dataclass3-4']

        # Runs the method which fetches information on all available XRISM observations and stores that
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
        self._miss_name = "xrism_pointed"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "XRISM Pointed"
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
        #  the BaseMission superclass - XRISM observations have a unique 9-digit ObsID, though I can find
        #  no discussion of whether there is extra information in the ObsID (i.e. target type).
        self._id_format = '^[0-9]{9}$'
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
        #  new mission class - XTEND is huge, RESOLVE is tiny (both are square?)
        self._approx_fov = {'XTEND': Quantity(19, 'arcmin'),
                            'RESOLVE': Quantity(1.5, 'arcmin')}
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
        credit) to acquire the 'xrismmastr' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available pointed XRISM observations, with important information such as pointing coordinates, ObsIDs, and
        exposure.
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
        # Tells the interface that I want to retrieve from the xrismmastr (XRISM Master) catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20xrismmastr"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/xrism/xrismmastr.html)
        which_cols = ['RA', 'DEC', 'OBSID', 'TIME', 'END_TIME', 'Duration', 'Exposure', 'Xtd_Expo', 'Subject_Category',
                      'Rsl_Datamode', 'Xtd_Datamode1', 'Xtd_Datamode2', 'Xtd_Dataclas1', 'Xtd_Dataclas2', 'Public_Date']
        # This is what will be put into the URL to retrieve just those data fields - there are quite a few more,
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
                full_xrism = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_xrism.select_dtypes(['object']).columns:
                    full_xrism[col] = full_xrism[col].apply(lambda x: x.strip())

        # Important first step, making any global cuts to the dataframe to remove entries with zero exposure, which
        #  I think in this case should be data that haven't been taken. The 'exposure' column will preferentially be
        #  RESOLVE exposure times, but if RESOLVE disabled will be XTEND
        rel_xrism = full_xrism[(full_xrism['EXPOSURE'] != 0.0)]
        # We throw a warning that some number of the XRISM observations are dropped because it doesn't seem that they
        #  will be at all useful
        if len(rel_xrism) != len(full_xrism):
            warn(
                "{ta} of the {tot} observations located for XRISM Pointed have been removed due to all instrument "
                "exposures being zero.".format(ta=len(full_xrism) - len(rel_xrism), tot=len(full_xrism)), stacklevel=2)

        # Lower-casing all the column names (personal preference largely).
        rel_xrism = rel_xrism.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects
        rel_xrism = rel_xrism.rename(columns={'obsid': 'ObsID', 'time': 'start', 'end_time': 'end',
                                              'subject_category': 'target_category', 'xtd_expo': 'xtend_exposure',
                                              'rsl_datamode': 'resolve_mode',
                                              'xtd_datamode1': 'xtend_mode1-2',
                                              'xtd_datamode2': 'xtend_mode3-4',
                                              'xtd_dataclas1': 'xtend_dataclass1-2',
                                              'xtd_dataclas2': 'xtend_dataclass3-4',
                                              'public_date': 'proprietary_end_date'})

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        rel_xrism['start'] = pd.to_datetime(Time(rel_xrism['start'].values.astype(float), format='mjd',
                                                 scale='utc').to_datetime())
        rel_xrism['end'] = pd.to_datetime(Time(rel_xrism['end'].values.astype(float), format='mjd',
                                               scale='utc').to_datetime())
        # Then make a duration column by subtracting one from t'other - there are also exposure and ontime columns
        #  which I've acquired, but I think total duration is what I will go with here.
        rel_xrism['duration'] = pd.to_timedelta(rel_xrism['duration'], 's')

        # Converting the exposure times to Pandas time deltas
        for col in rel_xrism.columns[rel_xrism.columns.str.contains('expo')]:
            rel_xrism[col] = pd.to_timedelta(rel_xrism[col], 's')

        # Slightly more complicated with the public release dates, as some of them are set to 0 MJD, which makes the
        #  conversion routine quite upset (0 is not valid) - as such I convert only those which aren't 0, then
        #  replace the 0 valued ones with Pandas' Not a Time (NaT) value
        # The '88068' value is in 2099 (I think), which is the value set for public date for some entries
        val_end_dates = (rel_xrism['proprietary_end_date'] != 0) & (rel_xrism['proprietary_end_date'] != 88068)

        # We make a copy of the proprietary end dates to work on because pandas will soon not allow us to set what
        #  was previously an int column with a datetime - so we need to convert it but retain the original
        #  data as well
        prop_end_datas = rel_xrism['proprietary_end_date'].copy()
        # Convert the original column to datetime
        rel_xrism['proprietary_end_date'] = rel_xrism['proprietary_end_date'].astype('datetime64[ns]')
        rel_xrism.loc[val_end_dates, 'proprietary_end_date'] = pd.to_datetime(
            Time(prop_end_datas[val_end_dates.values], format='mjd', scale='utc').to_datetime())
        rel_xrism.loc[~val_end_dates, 'proprietary_end_date'] = pd.NaT
        # Grab the current date and time
        today = datetime.today()
        # Create a boolean column that describes whether the data are out of their proprietary period yet
        rel_xrism['proprietary_usable'] = rel_xrism['proprietary_end_date'].apply(lambda x:
                                                                                  ((x <= today) &
                                                                                   (pd.notnull(x)))).astype(bool)

        # No clear way of defining this from the tables, so we're going to assume that they all are
        rel_xrism['science_usable'] = True

        # Convert the categories of target that are present in the dataframe to the DAXA taxonomy
        # The XRISM categories are here:
        #  https://heasarc.gsfc.nasa.gov/W3Browse/xrism/xrismmastr.html#subject_category
        conv_dict = {'Non-Jet-Dominated AGN': 'AGN',
                     'Blazars and Other Jet-Dominated AGN': 'BLZ', 'Solar System': 'SOL',
                     'Supernova Remnants': 'SNR',
                     'Normal Galaxies: Integrated Emission': 'NGS',
                     'Groups and Clusters of Galaxies': 'GCL',
                     'Extragalactic Diffuse Emission and Surveys': 'EGS',
                     'BH and NS Binaries': 'NS',
                     'Non-Accreting Neutron Stars': 'NS',
                     'Galactic Diffuse Emission and Surveys': 'GS',
                     'Extragalactic Transients: SNe, GRBs, GW Events, and TDEs': 'EGS',
                     'Astrophysical Processes': 'MISC',
                     'Stellar Coronae, Winds, and Young Stars': 'STR',
                     'Accreting White Dwarfs and Novae': 'STR'}

        # I construct a mask that tells me which entries have a recognised description - any that don't are set
        #  to the 'MISC' code
        type_recog = rel_xrism['target_category'].isin(list(conv_dict.keys()))
        # The recognized target category descriptions are converted to DAXA taxonomy
        rel_xrism.loc[type_recog, 'target_category'] = rel_xrism.loc[type_recog, 'target_category'].apply(
            lambda x: conv_dict[x])
        # Now I set any unrecognized target category descriptions to MISC - there are none at the time of writing,
        #  but that could well change
        rel_xrism.loc[~type_recog, 'target_category'] = 'MISC'

        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_xrism = rel_xrism[['ra', 'dec', 'ObsID', 'science_usable', 'proprietary_usable', 'start', 'end',
                               'duration', 'target_category', 'exposure', 'xtend_exposure', 'resolve_mode',
                               'xtend_mode1-2', 'xtend_mode3-4', 'xtend_dataclass1-2', 'xtend_dataclass3-4']]

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_xrism = rel_xrism.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_xrism

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

        req_dir = REQUIRED_DIRS['all']
        if download_products:
            dir_lookup = REQUIRED_DIRS['processed']
        else:
            dir_lookup = REQUIRED_DIRS['raw']

        # The data on HEASArc are stored in subdirectories that have the first digit of the ObsID as their name
        cat_id = observation_id[0]

        # This is the path to the HEASArc data directory for this ObsID
        obs_dir = "/FTP/xrism/data/obs/{cid}/{oid}/".format(cid=cat_id, oid=observation_id)
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the directories that I need to download unprocessed XRISM data are present
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
                # The way the XRISM archives are laid out, XIS has its own directory, and sub-directories that we
                #  need to decide whether to download or not
                rel_req_dir = dir_lookup[dat_dir[:-1]]
                to_down = []
                # Here we cycle through the directories that we have found at the instrument URL for this ObsID
                for en in BeautifulSoup(session.get(rel_url).text, "html.parser").find_all("a"):
                    # We have to check that the 'en' isn't some HTML guff that we don't need, and that the
                    #  subdirectories actually should be downloaded (i.e. we won't download event_cl and products
                    #  when the user doesn't want pre-processed data).
                    if '?' not in en['href'] and obs_dir not in en['href'] and en['href'] in rel_req_dir:
                        low_rel_url = rel_url + en['href']
                        files = [en['href'] + '/' + fil['href']
                                 for fil in BeautifulSoup(session.get(low_rel_url).text, "html.parser").find_all("a")
                                 if '?' not in fil['href'] and obs_dir not in fil['href']]
                    else:
                        files = []

                    # If the current subdirectory has got files that  we want to download, then we make sure that
                    #  the subdirectory exists locally
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

                # There are a few compressed fits files in each archive, but I think I'm going to decompress the
                #  event lists, gifs so people can have a quick look if they so desire, and the fits images
                if 'evt.gz' in down_file or 'gif.gz' in down_file or 'img.gz' in down_file:
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
        A method to acquire and download the XRISM data that have not been filtered out (if a filter
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

        # Ensures that a directory to store the 'raw' XRISM data in exists - once downloaded and unpacked
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
                        # Use the internal static method I set up which both downloads and unpacks the XRISM data
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
        A slightly unusual method which will allow the XRISM mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        raise NotImplementedError("The check_process_obs method has not yet been implemented for XRISM, as "
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
        # XRISM ObsIDs are always 9 digits, so we just retrieve the first 9
        return ident[:9]

