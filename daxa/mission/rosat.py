#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 24/04/2024, 10:27. Copyright (c) The Contributors

import io
import os
from multiprocessing import Pool
from pathlib import Path
from shutil import copyfileobj
from typing import Any, Union, List
from warnings import warn

import numpy as np
import pandas as pd
import requests
import unlzw3
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from bs4 import BeautifulSoup
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError, NoObsAfterFilterError
from daxa.mission.base import BaseMission, _lock_check

GOOD_FILE_PATTERNS = {'rass': {'processed': ['{o}_anc.fits.Z', '{o}_bas.fits.Z'],
                               'raw': ['{o}_raw.fits.Z', '{o}_anc.fits.Z']},
                      'pointed': {'processed': ['{o}_anc.fits.Z', '{o}_bas.fits.Z'],
                                  'raw': ['{o}_raw.fits.Z', '{o}_anc.fits.Z']}}

# The energy ranges are defined thus (https://heasarc.gsfc.nasa.gov/docs/rosat/newsletters/rdf_formats10.html):
#   - 1 = "broad band", which I'm taking to mean 0.07-2.4 keV
#   - 2 = "hard band", which is 0.4-2.4 keV
#   - 3 = "soft band", which is 0.07-0.4 keV
#  HRI observations only include im1 images, because the energy resolution of HRI is so poor
# The 'mex' file is the exposure map - for some reason HRI archives directories don't seem to have them??
PROC_PROD_NAMES = {'hri': ['{o}_bk1.fits.Z', '{o}_im1.fits.Z'],
                   'pspc': ['{o}_bk1.fits.Z', '{o}_bk2.fits.Z', '{o}_bk3.fits.Z',
                            '{o}_im1.fits.Z', '{o}_im2.fits.Z', '{o}_im3.fits.Z',
                            '{o}_mex.fits.Z']}


class ROSATPointed(BaseMission):
    """
    The mission class for ROSAT Pointed observations, taken after the initial all-sky survey. This mission includes
    the follow-up observations taken to complete the all-sky survey in pointed mode towards the end of the ROSAT
    lifetime. This mission class pulls observation information from the HEASArc ROSMASTER table, and downloads data
    from the HEASArc website.

    NOTE: Follow-up All-Sky observations are marked as being taken with 'PSPC' (rather than a specific PSPC-C or B)
    in the ROSMASTER table, but they were actually taken with PSPC-B, so DAXA corrects the entries on acquisition of
    the table from HEASArc.

    Another mission class is available for ROSAT All-Sky Survey observations, specifically the ones taken in slewing
    mode with PSPC-C at the beginning of the ROSAT mission.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
        pass either a single string value or a list of strings. They may include PSPCB, PSPCC, and HRI.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None):
        """
        The mission class for ROSAT Pointed observations, taken after the initial all-sky survey. This mission includes
        the follow-up observations taken to complete the all-sky survey in pointed mode towards the end of the ROSAT
        lifetime. This mission class pulls observation information from the HEASArc ROSMASTER table, and downloads data
        from the HEASArc website.

        NOTE: Follow-up All-Sky observations are marked as being taken with 'PSPC' (rather than a specific PSPC-C or B)
        in the ROSMASTER table, but they were actually taken with PSPC-B, so DAXA corrects the entries on acquisition of
        the table from HEASArc.

        Another mission class is available for ROSAT All-Sky Survey observations, specifically the ones taken in slewing
        mode with PSPC-C at the beginning of the ROSAT mission.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. You can
            pass either a single string value or a list of strings. They may include PSPCB, PSPCC, and HRI.
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        super().__init__()

        # Sets the default instruments - ROSAT had the Position Sensitive Proportional Counters (PSPC) and the
        #  High Resolution Imager (HRI). PSPC-C was used for the all-sky survey until it was destroyed by an
        #  accidental pass over the Sun. PSPC-B was used to complete the survey in pointed mode towards the end of
        #  the mission's life.
        if insts is None:
            insts = ['PSPCB', 'PSPCC', 'HRI']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
        # Makes sure everything is uppercase
        insts = [i.upper() for i in insts]

        # These are the allowed instruments for this mission - just the same as the default, as I have
        #  no immediate plans to include the wide field XUV imager
        self._miss_poss_insts = ['PSPCB', 'PSPCC', 'HRI']
        self._alt_miss_inst_names = {'PSPC-B': 'PSPCB', 'PSPC-C': 'PSPCC', 'PSPC B': 'PSPCB', 'PSPC C': 'PSPCC'}

        # Call the name property to set up the name and pretty name attributes
        self.name

        # I don't wish to add any extra columns over the defaults expected by DAXA
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available pointed Chandra observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()
        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

        # Deliberately using the property setter, because it calls the internal _check_chos_insts function
        #  to make sure the input instruments are allowed
        # This instrument stuff is down here because for ROSAT I want it to happen AFTER the Observation info
        #  table has been fetched. As ROSAT uses one instrument per observation, this will effectively be another
        #  filtering operation rather than the download-time operation is has been for NuSTAR for instance
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for ROSAT images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        self._template_en_trans = {'PSPCB': {Quantity(0.07, 'keV'): {Quantity(2.4, 'keV'): "1",
                                                                     Quantity(0.4, 'keV'): "3"},
                                             Quantity(0.4, 'keV'): {Quantity(2.4, 'keV'): "2"}},
                                   'PSPCC': {Quantity(0.07, 'keV'): {Quantity(2.4, 'keV'): "1",
                                                                     Quantity(0.4, 'keV'): "3"},
                                             Quantity(0.4, 'keV'): {Quantity(2.4, 'keV'): "2"}},
                                   'HRI': {Quantity(0.07, 'keV'): {Quantity(2.4, 'keV'): "1"}}
                                   }

        # We set up the ROSAT file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        self._template_evt_name = "{oi}_bas.fits"
        self._template_img_name = "{oi}_im{eb}.fits"
        self._template_exp_name = {"PSPCB": "{oi}_mex.fits",
                                   "PSPCC": "{oi}_mex.fits",
                                   "HRI": None}
        self._template_bck_name = "{oi}_bk{eb}.fits"

        # We use this to specify whether a mission has only one instrument per ObsID (it is quite handy to codify
        #  this for a couple of external processes).
        self._one_inst_per_obs = True

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
        self._miss_name = "rosat_pointed"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "ROSAT Pointed"
        return self._miss_name

    @property
    def chosen_instruments(self) -> List[str]:
        """
        Property getter for the names of the currently selected instruments associated with this mission which
        will be processed into an archive by DAXA functions. Overwritten here because there are custom behaviours
        for ROSATPointed, as it has one instrument per ObsID.

        :return: A list of instrument names.
        :rtype: List[str]
        """
        return self._chos_insts

    @chosen_instruments.setter
    @_lock_check
    def chosen_instruments(self, new_insts: List[str]):
        """
        Property setter for the instruments associated with this mission that should be processed. This property
        may only be set to a list that is a subset of the existing property value. Overwritten here because there
        are custom behaviours for ROSATPointed, as it has one instrument per ObsID.

        :param List[str] new_insts: The new list of instruments associated with this mission which should
            be processed into the archive.
        """
        # First of all, check whether the new instruments are valid for this mission
        new_insts = super().check_inst_names(new_insts, True)

        # If we've gotten through the super call then the instruments are acceptable, so now we filter the
        #  observation info table using them.
        sel_inst_mask = self._obs_info['instrument'].isin(new_insts)

        # I can't think of a way this would happen, but I will just quickly ensure that this filtering didn't
        #  return zero results
        if sel_inst_mask.sum() == 0:
            raise NoObsAfterFilterError("No ROSAT observations are left after instrument filtering.")

        # The boolean mask can be multiplied with the existing filter array (by default all ones, which means
        #  all observations are let through) to produce an updated filter.
        new_filter = self.filter_array * sel_inst_mask
        # Then we set the filter array property with that updated mask
        self.filter_array = new_filter

        self._chos_insts = new_insts

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission. Not completely
        certain that FK5 is the correct frame for RASS, but a processed image downloaded from HEASArc used FK5 as
        the reference frame for its WCS.

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
        #  the BaseMission superclass - RASS (and possibly all ROSAT?) observations have an ObsID of length
        #  11 (e.g. RS123456N00) APART FROM GERMAN PROCESSED OBSERVATIONS, they are 8 long??.
        #
        # rp = US PSPC
        # rf = US PSPC + BORON FILTER
        # rh = US HRI
        # wp = MPE PSPC
        # wf = MPE PSPC + BORON FILTER
        # wh = MPE HRI
        #
        #  The first two digits of RASS ObsIDs are always RS (which indicates scanning
        #  mode), the next 6 characters are the ROSAT observation request sequence number or ROR, while the
        #  following 3 characters after the ROR number are the follow-on suffix.
        self._id_format = r'^(RH|rh|RP|rp|RF|rf|WH|wh|WP|wp|WF|wf)\d{6}([A-Z]\d{2}|)$'
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
        #  new mission class. Values taken from https://heasarc.gsfc.nasa.gov/docs/rosat/rosat.html
        self._approx_fov = {'PSPCB': Quantity(60, 'arcmin'), 'PSPCC': Quantity(60, 'arcmin'),
                            'HRI': Quantity(19, 'arcmin')}
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
        credit) to acquire the 'ROSMASTER' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available ROSAT pointed observations, with important information such as pointing coordinates,
        ObsIDs, and exposure.
        """
        # This is the web interface for querying NASA HEASArc catalogues
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        # This returns the requested information in a FITS format - the idea being I will stream this into memory
        #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited, as though we could hard code how many ROSAT observations there are (there aren't
        #  going to be any more...) we should still try to avoid that
        result_max = "&ResultMax=0"
        # This just tells the interface it's a query (I think?)
        action = "&Action=Query"
        # Tells the interface that I want to retrieve from the ROSMASTER catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20rosmaster"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/rosat/rosmaster.html)
        which_cols = ['RA', 'DEC', 'Seq_ID', 'Start_Time', 'End_Time', 'Exposure', 'FITS_Type', 'Instrument',
                      'Filter', 'Proc_Rev', 'Subj_Cat', 'Name', 'Site']
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
                full_ros = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_ros.select_dtypes(['object']).columns:
                    full_ros[col] = full_ros[col].apply(lambda x: x.strip())

        # Lower-casing all the column names (personal preference largely).
        full_ros = full_ros.rename(columns=str.lower)

        # Changing a few column names to match what BaseMission expects - changing 'exposure' to duration might not
        #  be entirely valid as I'm not sure that they have consistent meanings throughout DAXA.
        #  TODO CHECK DURATION MEANING
        full_ros = full_ros.rename(columns={'seq_id': 'ObsID', 'start_time': 'start', 'end_time': 'end',
                                            'exposure': 'duration', 'filter': 'with_filter', 'name': 'target_name',
                                            'subj_cat': 'target_category'})

        # We make sure that the start and end columns are floats, otherwise the conversion to datetime doesn't work
        full_ros['start'] = full_ros['start'].astype(float)
        full_ros['end'] = full_ros['end'].astype(float)

        # The target category column is turned to strings, because that's how I wrote the conversion dictionary and
        #  I can't be bothered to change it
        full_ros['target_category'] = full_ros['target_category'].astype(str)

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        full_ros['start'] = pd.to_datetime(Time(full_ros['start'].values, format='mjd', scale='utc').to_datetime())
        full_ros['end'] = pd.to_datetime(Time(full_ros['end'].values, format='mjd', scale='utc').to_datetime())
        # Convert the exposure time into a Pandas datetime delta
        full_ros['duration'] = pd.to_timedelta(full_ros['duration'], 's')

        # At this point in other missions I have dealt with the proprietary release data, and whether data are
        #  currently in a proprietary period, but that isn't really a consideration for this mission as ROSAT died
        #  many years ago

        # There isn't really a flag that translates to this in the online table - all I have to go on is that there
        #  are some observations with an exposure time (in the ROSMASTER table at least) of 0, so we'll mark them
        #  as not usable until I know better (see issue #185)
        full_ros['science_usable'] = full_ros['duration'].apply(lambda x: False if x <= np.timedelta64(0) else True)

        # This step is necessary because some of the ROSAT Pointed observations are labelled as having instrument
        #  'PSPC', rather than a specific 'PSPCB' or 'PSPCC'. These are, apparently, the finishing-up observations
        #  for RASS, and even though they were all taken with PSPCB, they've just been generically labelled. This
        #  is according to communications with HEASArc - as such I change them here
        full_ros['instrument'] = full_ros['instrument'].apply(lambda x: x if x != 'PSPC' else 'PSPCB')

        # Here we translate the target categories to the DAXA taxonomy, which is shared between all DAXA missions
        # From the ROSMASTER catalogue page
        # 1        Normal (non-degenerate) stars
        # 2        White dwarf stars
        # 3        Cataclysmic variables
        # 4        Neutron stars and black holes
        # 5        Supernova remnants
        # 6        Normal galaxies
        # 7        Active Galactic Nuclei (AGN)
        # 8        Clusters of galaxies
        # 9        Extended and diffuse X-ray emission
        # 10        Other types of objects
        # I think 0 means mispoints (if we go by the table on this page
        #  https://heasarc.gsfc.nasa.gov/docs/rosat/archive_access.html)

        conv_dict = {'1': 'STR', '2': 'STR', '3': 'CV',
                     '4': 'NS', '5': 'SNR', '6': 'NGS', '7': 'AGN',
                     '8': 'GCL', '9': 'EGE', '10': 'MISC', '0': 'MISC'}

        # I don't want to assume that the types I've seen in the catalogue will stay forever (though they probably
        #  will), as such I construct a mask that tells me which entries have a recognised description - any that
        #  don't will be set to the 'MISC' code
        type_recog = full_ros['target_category'].isin(list(conv_dict.keys()))
        # The recognized target category descriptions are converted to DAXA taxonomy
        full_ros.loc[type_recog, 'target_category'] = full_ros.loc[type_recog, 'target_category'].apply(
            lambda x: conv_dict[x])

        # Re-ordering the table, and not including certain columns which have served their purpose
        full_ros = full_ros[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration', 'instrument',
                             'with_filter', 'target_category', 'target_name', 'proc_rev', 'fits_type']]

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = full_ros

    @staticmethod
    def _download_call(observation_id: str, raw_dir: str, download_products: bool):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        :param bool download_products: This controls whether the HEASArc-published images and exposure maps are
            downloaded alongside the event lists and attitude files. Setting this to True will download the
            images/exposure maps. The default is False.
        """

        # Make sure raw_dir has a slash at the end
        if raw_dir[-1] != '/':
            raw_dir += '/'

        # This grabs the first digit of the ROSAT observation request sequence number (ROR), which we need to know
        #  to construct a part of the URL for downloading the data
        obj_type = observation_id[2] + '00000'
        # Also need to determine the instrument, as that also factors into the URL
        inst = 'hri' if observation_id[1] == 'H' else 'pspc'

        # Setting up the FTP paths for ROSAT pointed data is slightly more complicated than for the All-Sky Survey, as
        #  pointed data can be with HRI or PSPC instruments, and the first digit of the six-digit chunk of the ObsID
        #  can be something other than 9, as that indicates what type of object was being observed
        obs_dir = "/FTP/rosat/data/{inst}/processed_data/{ot}/{oid}/".format(oid=observation_id.lower(),
                                                                             inst=inst, ot=obj_type)
        # This defines the files we're looking to download, based on the fact this is the pointed ROSAT
        #  mission, and we want the pre-processed data
        sel_files = [fp.format(o=observation_id.lower()) for fp in GOOD_FILE_PATTERNS['pointed']['processed']]

        if download_products:
            oth_sel_files = [fp.format(o=observation_id.lower()) for fp in PROC_PROD_NAMES[inst]]
            sel_files += oth_sel_files

        # TODO Probably remove this honestly
        # This URL is for downloading RAW data, not the pre-processed stuff
        # else:
        #     obs_dir = "/FTP/rosat/data/{inst}/RDA/{ot}/{oid}/".format(oid=observation_id.lower(), inst=inst,
        #                                                               ot=obj_type)
        #     # This defines the files we're looking to download, based on the fact this is the pointed ROSAT
        #     #  mission, and we want the raw data
        #     sel_files = [fp.format(o=observation_id.lower()) for fp in GOOD_FILE_PATTERNS['pointed']['raw']]

        # Assembles the full URL to the archive directory
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the files that I need to download RASS data are present
        top_data = [en['href'] for en in BeautifulSoup(session.get(top_url).text, "html.parser").find_all("a")
                    if en['href'] in sel_files]

        # If the lengths of top_data and the file list are different, then one or more of the
        #  expected dirs is not present
        if len(top_data) != len(sel_files):
            # This list comprehension figures out what file is missing and reports it
            missing = [fp for fp in sel_files if fp not in top_data]
            raise FileNotFoundError("The archive data directory for {o} does not contain the following required "
                                    "files; {rq}".format(o=observation_id, rq=", ".join(missing)))

        # This is where the data for this observation are to be downloaded, need to make sure said directory exists
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        for down_file in sel_files:
            stor_name = down_file.replace('.Z', '')
            down_url = top_url + down_file
            with session.get(down_url, stream=True) as acquiro:
                with open(raw_dir + down_file, 'wb') as writo:
                    copyfileobj(acquiro.raw, writo)

            # The files we're downloading are compressed
            if '.Z' in down_file:
                # Open and decompress the events file - as the storage setup for ROSAT Pointed uses an old compression
                #  algorithm we have to use this specialised module to decompress
                decomp = unlzw3.unlzw(Path(raw_dir + down_file))

                # Open a new file handler for the decompressed data, and store the decompressed bytes there
                with open(raw_dir + stor_name, 'wb') as writo:
                    writo.write(decomp)
                # Then remove the tarred file to minimise storage usage
                os.remove(raw_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, download_products: bool = True):
        """
        A method to acquire and download the ROSAT pointed data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded).

        Proprietary data is not a relevant concept for ROSAT at this point, so no option to provide
        credentials is provided here as it is in some other mission classes.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: This controls whether the HEASArc-published images and exposure maps are
            downloaded alongside the event lists and attitude files. Setting this to True will download the
            images/exposure maps. The default is True.
        """

        # Ensures that a directory to store the 'raw' RASS data in exists - once downloaded and unpacked
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

        # We store the type of data that was downloaded
        if download_products:
            self._download_type = "raw+preprocessed"
        else:
            self._download_type = "raw"

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in self.filtered_obs_ids:
                        # Use the internal static method I set up which both downloads and unpacks the RASS data
                        self._download_call(obs_id, raw_dir=stor_dir + '{o}'.format(o=obs_id),
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
                                         kwds={'observation_id': obs_id, 'raw_dir': stor_dir + '{o}'.format(o=obs_id),
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

    def assess_process_obs(self, obs_info: dict):
        raise NotImplementedError("The observation assessment process has not been implemented for ROSATPointed.")

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
        # Will just replace any of the instrument names with nothing, if they are present
        return ident.replace('HRI', '').replace('PSPCC', '').replace('PSPCB', '')


class ROSATAllSky(BaseMission):
    """
    The mission class for ROSAT All-Sky Survey (RASS) observations. The available observation information is
    fetched from the HEASArc  RASSMASTER table, and data are downloaded from the HEASArc https access to their FTP
    server. Only data from the initial scanning phase of RASS will be fetched by this class, not the follow-up pointed
    mode observations used to complete the survey towards the end of the ROSAT mission.

    Another mission class is available for pointed ROSAT observations.

    No instrument choice is offered for this mission class because all RASS observations in the scanning portion
    of the survey were taken with PSPC-C.

    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    """

    def __init__(self, save_file_path: str = None):
        """
        The mission class for ROSAT All-Sky Survey (RASS) observations. The available observation information is
        fetched from the HEASArc  RASSMASTER table, and data are downloaded from the HEASArc https access to their FTP
        server. Only data from the initial scanning phase of RASS will be fetched by this class, not the follow-up
        pointed mode observations used to complete the survey towards the end of the ROSAT mission.

        No instrument choice is offered for this mission class because all RASS observations in the scanning portion
        of the survey were taken with PSPC-C.

        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        """
        super().__init__()

        # Sets the default instrument - I have this in the same format (i.e. a list) as every other mission class, but
        #  given that the RASS data were all taken with PSPC I don't give the user a choice of instruments.
        insts = ['PSPC']

        # These are the allowed instruments for this mission - again it is just PSPC, but the mission class expects
        #  this attribute to be set
        self._miss_poss_insts = ['PSPC']
        # There are no alternative instrument names, especially because the user can't set the instruments.
        self._alt_miss_inst_names = {}

        # Setting the chosen instruments property, still using the BaseMission infrastructure even though we know
        #  there will only ever be the PSPC instrument for this mission
        self.chosen_instruments = insts

        # These are the 'translations' required between energy band and filename identifier for ROSAT images/expmaps -
        #  it is organised so that top level keys are instruments, middle keys are lower energy bounds, and the lower
        #  level keys are upper energy bounds, then the value is the filename identifier
        self._template_en_trans = {Quantity(0.07, 'keV'): {Quantity(2.4, 'keV'): "1",
                                                           Quantity(0.4, 'keV'): "3"},
                                   Quantity(0.4, 'keV'): {Quantity(2.4, 'keV'): "2"}}

        # We set up the ROSAT file name templates, so that the user (or other parts of DAXA) can retrieve paths
        #  to the event lists, images, exposure maps, and background maps that can be downloaded
        self._template_evt_name = "{oi}_bas.fits"
        self._template_img_name = "{oi}_im{eb}.fits"
        self._template_exp_name = "{oi}_mex.fits"
        self._template_bck_name = "{oi}_bk{eb}.fits"

        # Call the name property to set up the name and pretty name attributes
        self.name

        # I don't wish to add any extra columns over the defaults expected by DAXA
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available RASS observations and stores that
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
        self._miss_name = "rosat_all_sky"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "RASS"
        return self._miss_name

    @property
    def coord_frame(self) -> BaseRADecFrame:
        """
        Property getter for the coordinate frame of the RA-Decs of the observations of this mission. Not completely
        certain that FK5 is the correct frame for RASS, but a processed image downloaded from HEASArc used FK5 as
        the reference frame for its WCS.

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
        #  the BaseMission superclass - RASS (and possibly all ROSAT?) observations have an ObsID of length
        #  11 (e.g. RS123456N00). The first two digits of RASS ObsIDs are always RS (which indicates scanning
        #  mode), the next 6 characters are the ROSAT observation request sequence number or ROR, while the
        #  following 3 characters after the ROR number are the follow-on suffix. A complete pointing at a given
        #  ROSAT target comprises all the datasets having the same prefix and ROR numbers.
        self._id_format = r'^(RS|rs)\d{6}[A-Z]\d{2}$'
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
        # This isn't really the typical case as the field of view is artificial, based on the chunking of the data,
        #  but as RASS is in 6x6 degree chunks I think this is what makes the most sense.
        self._approx_fov = Quantity(180, 'arcmin')
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
        credit) to acquire the 'RASSMASTER' table from HEASArc - this method is much simpler, as it doesn't need to be
        dynamic and accept different arguments, and we will filter observations locally. This table describes the
        available ROSAT All-Sky Survey observations, with important information such as pointing coordinates,
        ObsIDs, and exposure.
        """
        # This is the web interface for querying NASA HEASArc catalogues
        host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

        # This returns the requested information in a FITS format - the idea being I will stream this into memory
        #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
        down_form = "&displaymode=FitsDisplay"
        # This should mean unlimited, as though we could hard code how many RASS observations there are (there aren't
        #  going to be any more...) we should still try to avoid that
        result_max = "&ResultMax=0"
        # This just tells the interface it's a query (I think?)
        action = "&Action=Query"
        # Tells the interface that I want to retrieve from the RASSMASTER catalogue
        table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20rassmaster"

        # The definition of all of these fields can be found here:
        #  (https://heasarc.gsfc.nasa.gov/W3Browse/rosat/rassmaster.html)
        # The INSTRUMENT_MODE is acquired here even though they say that it is unlikely any observations will be made
        #  in 'normal' mode, just so I can exclude those observations because frankly I don't know the difference
        # SPACECRAFT_MODE is acquired because the 'STELLAR' mode might not be suitable for science so may be excluded
        which_cols = ['RA', 'DEC', 'Seq_ID', 'Start_Date', 'End_Date', 'Exposure']
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
                full_rass = Table(full_fits[1].data).to_pandas()
                # This cycles through any column with the 'object' data type (string in this instance), and
                #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                #  string data).
                for col in full_rass.select_dtypes(['object']).columns:
                    full_rass[col] = full_rass[col].apply(lambda x: x.strip())

        # Lower-casing all the column names (personal preference largely).
        full_rass = full_rass.rename(columns=str.lower)
        # Changing a few column names to match what BaseMission expects - changing 'exposure' to duration might not
        #  be entirely valid as I'm not sure that they have consistent meanings throughout DAXA.
        #  TODO CHECK DURATION MEANING
        full_rass = full_rass.rename(columns={'seq_id': 'ObsID', 'start_date': 'start', 'end_date': 'end',
                                              'exposure': 'duration'})

        # We convert the Modified Julian Date (MJD) dates into Pandas datetime objects, which is what the
        #  BaseMission time selection methods expect
        full_rass['start'] = pd.to_datetime(Time(full_rass['start'].values, format='mjd', scale='utc').to_datetime())
        full_rass['end'] = pd.to_datetime(Time(full_rass['end'].values, format='mjd', scale='utc').to_datetime())
        # Convert the exposure time into a Pandas datetime delta
        full_rass['duration'] = pd.to_timedelta(full_rass['duration'], 's')

        # At this point in other missions I have dealt with the proprietary release data, and whether data are
        #  currently in a proprietary period, but that isn't really a consideration for this mission as RASS finished
        #  decades ago

        # There isn't really a flag that translates to this in the online table, and I hope that if the data are
        #  being served on HEASArc after this long then they are scientifically usable
        full_rass['science_usable'] = True

        # There isn't target information because this is an all sky survey, but I have actually added an 'all sky
        #  survey' target type to the DAXA taxonomy. So we'll set all the observations to that
        full_rass['target_category'] = 'ASK'

        # Re-ordering the table, and not including certain columns which have served their purpose
        full_rass = full_rass[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration', 'target_category']]

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = full_rass

    @staticmethod
    def _download_call(observation_id: str, raw_dir: str, download_products: bool):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        :param bool download_products: This controls whether the HEASArc-published images and exposure maps are
            downloaded alongside the event lists and attitude files. Setting this to True will download the
            images/exposure maps. The default is False.
        """

        # Make sure raw_dir has a slash at the end
        if raw_dir[-1] != '/':
            raw_dir += '/'

        # This is the path to the HEASArc data directory for this ObsID - all PSPC data are stored in parent
        #  directories that have names/IDs corresponding to the targeted object type. In the case of RASS that
        #  will always be 900000, as it corresponds to Solar Systems, SURVEYS, and Miscellaneous. Specifically this
        #  is the URL for downloading the pre-processed data
        obs_dir = "/FTP/rosat/data/pspc/processed_data/900000/{oid}/".format(oid=observation_id.lower())
        # This defines the files we're looking to download, based on the fact this is a RASS mission, and we want
        #  the pre-processed data
        sel_files = [fp.format(o=observation_id.lower()) for fp in GOOD_FILE_PATTERNS['rass']['processed']]

        if download_products:
            oth_sel_files = [fp.format(o=observation_id.lower()) for fp in PROC_PROD_NAMES['pspc']]
            sel_files += oth_sel_files

        # TODO Probably remove this entirely
        # This URL is for downloading RAW data, not the pre-processed stuff
        # else:
        #     obs_dir = "/FTP/rosat/data/pspc/RDA/900000/{oid}/".format(oid=observation_id)
        #     # This defines the files we're looking to download, based on the fact this is a RASS mission, and we want
        #     #  the raw data
        #     sel_files = [fp.format(o=observation_id.lower()) for fp in GOOD_FILE_PATTERNS['rass']['raw']]

        # Assembles the full URL to the archive directory
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist
        session = requests.Session()

        # This uses the beautiful soup module to parse the HTML of the top level archive directory - I want to check
        #  that the files that I need to download RASS data are present
        if not download_products:
            top_data = [en['href'] for en in BeautifulSoup(session.get(top_url).text, "html.parser").find_all("a")
                        if en['href'] in sel_files]
        # Note that in the case I am downloading processed data, I add an extra file to the check, a variation of the
        #  exposure map that hasn't been compressed (I've noticed this happens sometimes). The idea being if it is
        #  there then the fits.Z version WON'T be, and the next check that top_data has the same length as sel_files
        #  won't fail.
        else:
            top_data = [en['href'] for en in BeautifulSoup(session.get(top_url).text, "html.parser").find_all("a")
                        if en['href'] in (sel_files + ['{o}_mex.fits'.format(o=observation_id).lower()])]

        # If the lengths of top_data and the file list are different, then one or more of the
        #  expected dirs is not present
        if len(top_data) != len(sel_files):
            # This list comprehension figures out what file is missing and reports it
            missing = [fp for fp in sel_files if fp not in top_data]
            raise FileNotFoundError("The archive data directory for {o} does not contain the following required "
                                    "files; {rq}".format(o=observation_id, rq=", ".join(missing)))

        # This is where the data for this observation are to be downloaded, need to make sure said directory exists
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        # I use the top_data list here because it contains the filenames, but might be different from sel_files in the
        #  case where an exposure map is stored as a fits and not a fits.Z
        for down_file in top_data:
            stor_name = down_file.replace('.Z', '')
            down_url = top_url + down_file
            with session.get(down_url, stream=True) as acquiro:
                with open(raw_dir + down_file, 'wb') as writo:
                    copyfileobj(acquiro.raw, writo)

            # The files we're downloading are compressed
            if '.Z' in down_file:
                # Open and decompress the events file - as the storage setup for RASS uses an old compression
                #  algorithm we have to use this specialised module to decompress
                decomp = unlzw3.unlzw(Path(raw_dir + down_file))

                # Open a new file handler for the decompressed data, and store the decompressed bytes there
                with open(raw_dir + stor_name, 'wb') as writo:
                    writo.write(decomp)
                # Then remove the tarred file to minimise storage usage
                os.remove(raw_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, download_products: bool = True):
        """
        A method to acquire and download the ROSAT All-Sky Survey data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded).

        Proprietary data is not a relevant concept for RASS, so no option to provide credentials is provided here
        as it is in some other mission classes.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_products: This controls whether the HEASArc-published images and exposure maps are
            downloaded alongside the event lists and attitude files. Setting this to True will download the
            images/exposure maps. The default is True.
        """

        # Ensures that a directory to store the 'raw' RASS data in exists - once downloaded and unpacked
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

        # We store the type of data that was downloaded
        if download_products:
            self._download_type = "raw+preprocessed"
        else:
            self._download_type = "raw"

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in self.filtered_obs_ids:
                        # Use the internal static method I set up which both downloads and unpacks the RASS data
                        self._download_call(obs_id, raw_dir=stor_dir + '{o}'.format(o=obs_id),
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
                                         kwds={'observation_id': obs_id, 'raw_dir': stor_dir + '{o}'.format(o=obs_id),
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
        raise NotImplementedError("The observation assessment process has not been implemented for ROSATAllSky.")

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
        # Will just replace the one instrument identifier possible for this mission with nothing, if it is present
        return ident.replace('PSPC', '')
