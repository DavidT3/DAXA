#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 24/04/2024, 10:27. Copyright (c) The Contributors

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

# The HEASArc archive SCW directories don't have any subdirectories, so that is simpler. There are a bunch of files
#  in there that I can't really find a definition of anywhere unfortunately, but the limited mention of the SCW
#  storage structure that I could find is here:
#  https://www.isdc.unige.ch/integral/download/osa/doc/11.2/osa_um_intro/node28.html
# I don't know if excluding certain instrument's files is going to be a problem, but we'll try it in the initial
#  implementation, and when I try to actually use INTEGRAL again later I'll see if it works.
# Here I define patterns for the required files, for all cases and for specific instruments (because apparently
#  I'm going to implement this differently for every single mission...)
REQUIRED_FILES = {'all': ['_scw.txt', 'sc_', 'swg.fits'],
                  'jemx1': ['jmx1'],
                  'jemx2': ['jmx2'],
                  'isgri': ['isgri', 'ibis', 'compton'],
                  'picsit': ['picsit', 'ibis', 'compton'],
                  'spi': ['spi']}


class INTEGRALPointed(BaseMission):
    """
    The mission class for pointed observations by the INTErnational Gamma-Ray Astrophysics Laboratory (INTEGRAL); i.e.
    observations taken when the spacecraft isn't slewing, and is not in an engineering mode (and is public).
    The available observation information is fetched from the HEASArc INTSCWPUB table, and data are downloaded from
    the HEASArc https access to their FTP server. Proprietary data are not currently supported by this class.

    NOTE - This class treats Science Window IDs as 'obs ids', though observation IDs are a separate concept in the
    HEASArc table at least. The way DAXA is set up however, science window IDs are a closer analogy to the ObsIDs
    used by the rest of the missions.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include JEMX1, JEMX2, ISGRI, PICsIT, and
        SPI (the default is JEMX1, JEMX2, and ISGRI). OMC is not supported by DAXA.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
        The mission class for pointed observations by the INTErnational Gamma-Ray Astrophysics Laboratory
        (INTEGRAL); i.e. observations taken when the spacecraft isn't slewing, and is not in an engineering
        mode (and is public). The available observation information is fetched from the HEASArc INTSCWPUB table, and
        data are downloaded from the HEASArc https access to their FTP server. Proprietary data are not currently
        supported by this class.

        NOTE - This class treats Science Window IDs as 'obs ids', though observation IDs are a separate concept in the
        HEASArc table at least. The way DAXA is set up however, science window IDs are a closer analogy to the ObsIDs
        used by the rest of the missions.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include JEMX1, JEMX2, ISGRI, PICsIT, and
            SPI (the default is JEMX1, JEMX2, and ISGRI). OMC is not supported by DAXA.
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        super().__init__()

        # Sets the default instruments - the two soft X-ray (JEMX) instruments, and the ISGRI hard X-ray to low-energy
        #  gamma ray detector
        if insts is None:
            insts = ['JEMX1', 'JEMX2', 'ISGRI']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - INTEGRAL has two coded mask soft-band X-ray
        #  instruments (JEMX1&2), the two IBIS instruments (ISGRI and PICsIT), and SPI (8 keV - 8 MeV spectrometer)
        self._miss_poss_insts = ['JEMX1', 'JEMX2', 'ISGRI', 'PICsIT', 'SPI']
        # As far as I know there aren't any other common names for the instruments on INTEGRAL
        self._alt_miss_inst_names = {}

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        self.chosen_instruments = insts

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available INTEGRAL observations and stores that
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
        self._miss_name = "integral_pointed"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "INTEGRAL Pointed"
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
        #  the BaseMission superclass - INTEGRAL splits data into 'science windows', which each have a unique ID, and
        #  this is what we're using as an ObsIDs. They are 12 digits, and have information embedded in them:
        #  https://heasarc.gsfc.nasa.gov/W3Browse/integral/intscwpub.html#ScW_ID
        # ObsIDs are also a concept for INTEGRAL, but don't necessarily seem to be unique identifiers of data chunks
        #  like they are for the rest of the missions in DAXA - SCW IDs are closer to that behaviour
        self._id_format = '^[0-9]{12}$'
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
        # These definitions of field of view are taken from various sources, and don't necessarily represent the
        #  theoretical maximum field of view of the instruments.
        # JEMX info is taken from https://www.cosmos.esa.int/web/integral/instruments-jemx, and we choose to use
        #  the 'fully illuminated' value, where the FoV has a diameter of 4.8deg
        # IBIS info is taken from https://www.cosmos.esa.int/web/integral/instruments-ibis, and we choose to use the
        #  'fully coded' value, where the FoV is 8.3x8.0deg
        # SPI info is taken from https://www.cosmos.esa.int/web/integral/instruments-spi, and we choose to use the
        #  'fully coded flat to flat' value, where the FoV is 14deg in diameter
        self._approx_fov = {'JEMX1': Quantity(2.4, 'deg'), 'JEMX2': Quantity(2.4, 'deg'),
                            'ISGRI': Quantity(4.15, 'deg'), 'PICsIT': Quantity(4.15, 'deg'),
                            'SPI': Quantity(7, 'deg')}
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
        credit) to acquire the 'intscwpub' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available pointed INTEGRAL observations, with important information such as pointing coordinates, ObsIDs,
        and exposure.
        """
        # This is the web interface for querying NASA HEASArc catalogues
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        # This returns the requested information in a FITS format - the idea being I will stream this into memory
        #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited, as we don't know how many pointed INTEGRAL observations there are, and the
        #  number will increase with time
        result_max = "&ResultMax=0"
        # This just tells the interface it's a query (I think?)
        action = "&Action=Query"
        # Tells the interface that I want to retrieve from the intscwpub (INTEGRAL Pointed Public Master) catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20intscwpub"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/integral/intscwpub.html)

        # Could poll the individual instrument observing times for different modes to determine the mode of the
        #  instrument - but that would be a LOT of columns
        which_cols = ['RA', 'DEC', 'ScW_ID', 'Start_Date', 'End_Date', 'SPI_mode', 'Good_JEMX1', 'Good_JEMX2',
                      'Good_ISGRI', 'Good_PICSIT', 'Good_SPI', 'Data_In_HEASARC', 'ScW_Ver']
        # Might add these at some point:
        # Obs_Type, 'JEMX1_mode', 'JEMX2_mode', 'IBIS_Mode'

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
                full_integral = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_integral.select_dtypes(['object']).columns:
                    full_integral[col] = full_integral[col].apply(lambda x: x.strip())

        # Important first step, making any global cuts to the dataframe to remove entries that are not going to be
        #  useful. The table I am pulling from should ONLY include entries with at least one good exposure time, but
        #  that does include OMC, which I am not going to be supporting, so we will check
        rel_integral = full_integral[(full_integral['GOOD_JEMX1'] != 0.0) |
                                     (full_integral['GOOD_JEMX2'] != 0.0) |
                                     (full_integral['GOOD_ISGRI'] != 0.0) |
                                     (full_integral['GOOD_PICSIT'] != 0.0) |
                                     (full_integral['GOOD_SPI'] != 0.0)]
        # We throw a warning that some number of the INTEGRAL observations are dropped because it doesn't seem that
        #  they will be at all useful
        if len(rel_integral) != len(full_integral):
            warn("{ta} of the {tot} observations located for INTEGRAL have been removed due to all instrument "
                 "exposures being zero.".format(ta=len(full_integral)-len(rel_integral), tot=len(full_integral)),
                 stacklevel=2)

        # This removes any ObsIDs that have zero exposure time for the currently selected instruments - if all three
        #  instruments are selected then this won't do anything because it is the same as what we did a couple of
        #  lines up - but if a subset have been selected then it might well do something
        pre_inst_exp_check_num = len(rel_integral)
        rel_integral = rel_integral[np.logical_or.reduce([rel_integral['GOOD_' + inst] != 0
                                                          for inst in self.chosen_instruments])]
        # I warn the user if their chosen instruments have observations that have been removed because the chosen
        #  instruments are all zero exposure
        if len(rel_integral) != pre_inst_exp_check_num:
            warn("{ta} of the {tot} observations located for INTEGRAL have been removed due to all chosen instrument "
                 "({ci}) exposures being zero.".format(ta=pre_inst_exp_check_num-len(rel_integral),
                                                       tot=len(full_integral),
                                                       ci=", ".join(self.chosen_instruments)), stacklevel=2)

        # Lower-casing all the column names (personal preference largely).
        rel_integral = rel_integral.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects
        rel_integral = rel_integral.rename(columns={'scw_id': 'ObsID', 'start_date': 'start', 'end_date': 'end',
                                                    'good_jemx1': 'jemx1_exposure', 'good_jemx2': 'jemx2_exposure',
                                                    'good_isgri': 'isgri_exposure', 'good_picsit': 'picsit_exposure',
                                                    'good_spi': 'spi_exposure'})

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        rel_integral['start'] = pd.to_datetime(Time(rel_integral['start'].values.astype(float), format='mjd',
                                                    scale='utc').to_datetime())
        rel_integral['end'] = pd.to_datetime(Time(rel_integral['end'].values.astype(float), format='mjd',
                                                  scale='utc').to_datetime())
        # Then make a duration column by subtracting the start MJD from the end MJD - not an exposure time but just
        #  how long the observation window was
        rel_integral['duration'] = rel_integral['end'] - rel_integral['start']
        # Cycle through the exposure columns and make sure that they are time objects
        for col in rel_integral.columns[rel_integral.columns.str.contains('exposure')]:
            rel_integral[col] = pd.to_timedelta(rel_integral[col], 's')

        # Not really much information in the table that I can use to decide how to populate this
        rel_integral['science_usable'] = True

        rel_integral['proprietary_usable'] = (rel_integral['data_in_heasarc'] == 'Y')

        # There are no INTEGRAL target categories in the catalogue unfortunately
        # The recognized target category descriptions are converted to DAXA taxonomy
        rel_integral['target_category'] = 'MISC'

        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_integral = rel_integral[['ra', 'dec', 'ObsID', 'science_usable', 'proprietary_usable', 'start', 'end',
                                     'duration', 'target_category', 'jemx1_exposure', 'jemx2_exposure',
                                     'isgri_exposure', 'picsit_exposure', 'spi_exposure', 'scw_ver']]

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_integral = rel_integral.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_integral

    @staticmethod
    def _rev_download_call(rev_id: str, raw_dir: str):
        """
        A download call function particular to the INTEGRAL mission, meant to download the revolution data that
        we seem to be strongly encouraged to download by HEASArc. Specifically the rev.001 directories that accompany
        science windows, and the aux/adp/{rev_id}.001 directories containing auxiliary revolution data.

        :param str rev_id: The ID (i.e. orbit number) for the revolution and auxiliary revolution data that
            we wish to download.
        :param str raw_dir: The raw data directory in which to store the downloaded revolution data.
        """
        rev_dir = "/FTP/integral/data/scw/{rid}/rev.001/".format(rid=rev_id)
        rev_top_url = "https://heasarc.gsfc.nasa.gov" + rev_dir

        aux_dir = "/FTP/integral/data/aux/adp/{rid}.001/".format(rid=rev_id)
        aux_rev_top_url = "https://heasarc.gsfc.nasa.gov" + aux_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the revolution and aux revolution directories -
        #  essentially just to check that they have entries, and then we'll download them. First the rev.001 directory
        #  that is in the SCW revolution ID subdirectories
        rev_dirs = [en['href'] for en in BeautifulSoup(session.get(rev_top_url).text, "html.parser").find_all("a")
                    if '?' not in en['href'] and rev_id not in en['href']]
        aux_rev_to_down = [en['href'] for en in BeautifulSoup(session.get(aux_rev_top_url).text,
                                                              "html.parser").find_all("a")
                           if 'fits' in en['href'] and '?' not in en['href']]

        # This is the local directory where the revolution data are stored - these directories also have the science
        #  window downloaded in them eventually - I'm trying to follow the structure of the HEASArc archive
        local_rev_dir = raw_dir + rev_id + '/rev.001/'
        # Make sure that the local directory is created
        if not os.path.exists(local_rev_dir):
            os.makedirs(local_rev_dir)

        for dat_dir in rev_dirs:
            # The lower level URL of the directory we're currently looking at
            rel_url = rev_top_url + dat_dir

            # The list of files to download from the current data directory in the revolution data directory
            rev_to_down = []

            # Here we cycle through the directories in the revolution data directory
            for en in BeautifulSoup(session.get(rel_url).text, "html.parser").find_all("a"):
                # We have to check that the 'en' isn't some HTML guff that we don't need
                if '?' not in en['href'] and '{}/rev.001/'.format(rev_id) not in en['href']:
                    rev_to_down.append(dat_dir + en['href'])

            if len(rev_to_down) != 0 and not os.path.exists(local_rev_dir + dat_dir):
                os.makedirs(local_rev_dir + dat_dir)

            # Now we cycle through the revolution files and download them
            for down_file in rev_to_down:
                down_url = rev_top_url + down_file
                with session.get(down_url, stream=True) as acquiro:
                    with open(local_rev_dir + down_file, 'wb') as writo:
                        copyfileobj(acquiro.raw, writo)

        # This is where the auxiliary revolution information lives
        local_aux_rev_dir = raw_dir + 'aux/adp/{}.001/'.format(rev_id)
        # Make sure the auxiliary revolution directory is created locally - otherwise we have nowhere to
        #  download stuff too
        if len(aux_rev_to_down) != 0 and not os.path.exists(local_aux_rev_dir):
            os.makedirs(local_aux_rev_dir)

        # Now we cycle through the auxiliary revolution files and download them
        for down_file in aux_rev_to_down:
            down_url = aux_rev_top_url + down_file
            with session.get(down_url, stream=True) as acquiro:
                with open(local_aux_rev_dir + down_file, 'wb') as writo:
                    copyfileobj(acquiro.raw, writo)

        return None

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], scw_ver: str, raw_dir: str):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation. There is no download_processed option
        here because there are no pre-generated products (i.e. images/spectra) to download, though the event lists
        etc. have gone through some form of processing.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param List[str] insts: The instruments which the user wishes to acquire data for.
        :param str scw_ver: The string representing the version of the science window, which needs to be added to
            the observation_id (i.e. scw id in normal INTEGRAL parlance) to construct the HEASArc directory. The
            most common value for this is 001.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        """
        insts = [inst.lower() for inst in insts]
        req_files = list(set(REQUIRED_FILES['all'] + [file for inst in insts for file in REQUIRED_FILES[inst]]))

        # The data on HEASArc are stored in subdirectories named after the orbital revolution that they were taken
        #  in; this can be extracted from the ObsID (what we refer to the SCWID as), as they are the first four digits
        rev_id = observation_id[:4]

        # This is the path to the HEASArc data directory for this ObsID
        obs_dir = "/FTP/integral/data/scw/{rid}/{oid}.{scv}/".format(rid=rev_id, oid=observation_id, scv=scw_ver)
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist - then a lot of the next session is for checking that the expected
        #  directories are present.
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to narrow
        #  down the files to download for the selected instruments
        to_down = [en['href'] for en in BeautifulSoup(session.get(top_url).text, "html.parser").find_all("a")
                   for rf_patt in req_files if rf_patt in en['href']]

        # For some reason raw_dir isn't actually just the base dir it has observation ID already in as well and uuurgh
        #  inconsistent design but rather than sort it I'm bodging it - to try to make this compatible with how the
        #  inflexible OSA software seems to want the data laid out I am storing the SCW in revolution subdirectories
        raw_dir = raw_dir.split(observation_id)[0] + rev_id + '/' + observation_id + '/'
        # Make sure the ObsID directory is created locally - otherwise we have nowhere to download stuff too
        if len(to_down) != 0 and not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        # Now we cycle through the files and download them
        for down_file in to_down:
            down_url = top_url + down_file
            with session.get(down_url, stream=True) as acquiro:
                with open(raw_dir + '/' + down_file, 'wb') as writo:
                    copyfileobj(acquiro.raw, writo)

            # There are a few compressed fits files in each archive, but I think I'm only going to decompress the
            #  event lists, as they're more likely to be used
            if 'evt.gz' in down_file:
                # Open and decompress the events file
                with gzip.open(raw_dir + '/' + down_file, 'rb') as compresso:
                    # Open a new file handler for the decompressed data, then funnel the decompressed events there
                    with open(raw_dir + '/' + down_file.split('.gz')[0], 'wb') as writo:
                        copyfileobj(compresso, writo)
                # Then remove the tarred file to minimise storage usage
                os.remove(raw_dir + '/' + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, download_products: bool = True):
        """
        A method to acquire and download the pointed INTEGRAL data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        There is no download_processed option here because there are no pre-generated products (i.e. images/spectra)
        to download, though the event lists etc. have gone through some form of processing.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: PRESENT FOR COMPATIBILITY WITH OTHER DAXA TASKS - the INTEGRAL archive does
            not provide pre-processed products for download.
        """

        if not self.filtered_obs_info['proprietary_usable'].all():
            warn("Proprietary data have been selected, but cannot be downloaded with DAXA; as such the proprietary "
                 "data have been excluded from download and further processing.", stacklevel=2)
            new_filter = self.filter_array * self.all_obs_info['proprietary_usable'].values
            self.filter_array = new_filter

        # Ensures that a directory to store the 'raw' Swift data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.top_level_path + self.name + '_raw'):
            os.makedirs(self.top_level_path + self.name + '_raw')
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        # This INTEGRAL mission currently only supports the downloading of raw data, they don't store anything
        #  else in their online archive so far as I can tell
        self._download_type = 'raw'

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + '{o}'.format(o=o)) for o in self.filtered_obs_ids]):
            self._download_done = True

        if not self._download_done:
            # INTEGRAL is slightly different to the other missions, in that it seems you are strongly encouraged to
            #  download revolution (i.e. orbit number) level data to accompany the science windows. As such there are
            #  TWO download phases, the first of which downloads the rev data. As such we need to know which
            #  revolutions are relevant, which we can find from the first four digits of the ObsIDs (or rather SCWIDS)
            rev_ids = list(set(self.filtered_obs_info['ObsID'].apply(lambda x: x[:4]).values))

            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                # Revolution data has to be downloaded for INTEGRAL (apparently...), so we're doing that first!
                with tqdm(total=len(rev_ids), desc='Downloading {} revolution data'.format(self._pretty_miss_name)) \
                        as download_prog:
                    for rev_id in rev_ids:
                        self._rev_download_call(rev_id, raw_dir=stor_dir)
                        download_prog.update(1)

                # Then we run the more standard _download_call, to grab the actual data archives
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for row_ind, row in self.filtered_obs_info.iterrows():
                        obs_id = row['ObsID']
                        # Use the internal static method I set up which both downloads and unpacks the Swift data
                        self._download_call(obs_id, insts=self.chosen_instruments, scw_ver=str(row['scw_ver']),
                                            raw_dir=stor_dir + '{o}'.format(o=obs_id))
                        # Update the progress bar
                        download_prog.update(1)

            elif num_cores > 1:
                # List to store any errors raised during download tasks
                raised_errors = []

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

                # Revolution data has to be downloaded for INTEGRAL (apparently...), so we're doing that first!
                with tqdm(total=len(rev_ids), desc='Downloading {} revolution data'.format(self._pretty_miss_name)) \
                        as download_prog, Pool(num_cores) as pool:
                    for rev_id in rev_ids:
                        self._rev_download_call(rev_id, raw_dir=stor_dir)
                        pool.apply_async(self._rev_download_call,
                                         kwds={'rev_id': rev_id,
                                               'raw_dir': stor_dir},
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

                # This time, as we want to use multiple cores, I also set up a Pool to add download tasks too
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) \
                        as download_prog, Pool(num_cores) as pool:

                    # Again nested for loop through ObsIDs and instruments
                    for row_ind, row in self.filtered_obs_info.iterrows():
                        obs_id = row['ObsID']
                        # Add each download task to the pool
                        pool.apply_async(self._download_call,
                                         kwds={'observation_id': obs_id, 'insts': self.chosen_instruments,
                                               "scw_ver": str(row['scw_ver']),
                                               'raw_dir': stor_dir + '{o}'.format(o=obs_id)},
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

    def assess_process_obs(self, obs_info: dict):
        """
        A slightly unusual method which will allow the INTEGRALPointed mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        """
        raise NotImplementedError("The check_process_obs method has not yet been implemented for INTEGRALPointed, as "
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
        # INTEGRAL ObsIDs (or rather science window IDs) are always 12 digits, so we just retrieve the first 12
        return ident[:12]
