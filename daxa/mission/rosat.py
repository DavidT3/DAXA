#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 27/07/2023, 07:45. Copyright (c) The Contributors

import io
import os
from multiprocessing import Pool
from pathlib import Path
from shutil import copyfileobj
from typing import Any
from warnings import warn

import pandas as pd
import requests
import unlzw3
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from bs4 import BeautifulSoup
from tqdm import tqdm

from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission

GOOD_FILE_PATTERNS = {'rass': {'processed': ['{o}_anc.fits.Z', '{o}_bas.fits.Z'],
                               'raw': ['{o}_raw.fits.Z', '{o}_anc.fits.Z']}}


class ROSATAllSky(BaseMission):
    """

    No instrument choice is offered for this mission class because all RASS observations were taken with PSPC-C.
    """

    def __init__(self):
        """

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

        # Call the name property to set up the name and pretty name attributes
        self.name

        # TODO Revisit this when I've explored what is actually in the table for RASS
        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = []

        # Runs the method which fetches information on all available RASS observations and stores that
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
        self._id_format = '^[A-Z]{2}+[0-9]{6}+[A-Z]{1}+[0-9]{2}$'
        return self._id_format

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
        # Tells the interface that I want to retrieve from the numaster (RASSMASTER) catalogue
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
    def _download_call(observation_id: str, raw_dir: str, download_processed: bool):
        """
        The internal method called (in a couple of different possible ways) by the download method. This will check
        the availability of, acquire, and decompress the specified observation.

        :param str observation_id: The ObsID of the observation to be downloaded.
        :param str raw_dir: The raw data directory in which to create an ObsID directory and store the downloaded data.
        :param bool download_processed: This controls whether the data downloaded are the pre-processed event lists
            stored by HEASArc, or whether they are the original raw event lists. Default is to download pre-processed
            data.
        """

        # Make sure raw_dir has a slash at the end
        if raw_dir[-1] != '/':
            raw_dir += '/'

        # This is the path to the HEASArc data directory for this ObsID - all PSPC data are stored in parent
        #  directories that have names/IDs corresponding to the targeted object type. In the case of RASS that
        #  will always be 900000, as it corresponds to Solar Systems, SURVEYS, and Miscellaneous. Specifically this
        #  is the URL for downloading the pre-processed data
        if download_processed:
            obs_dir = "/FTP/rosat/data/pspc/processed_data/900000/{oid}/".format(oid=observation_id.lower())
        # This URL is for downloading RAW data, not the pre-processed stuff
        else:
            obs_dir = "/FTP/rosat/data/pspc/RDA/900000/{oid}/".format(oid=observation_id)
        # Assembles the full URL to the archive directory
        top_url = "https://heasarc.gsfc.nasa.gov" + obs_dir

        # This opens a session that will persist
        session = requests.Session()

        # This defines the files we're looking to download, based on the fact this is a RASS mission, and we want
        #  the pre-processed data
        sel_files = [fp.format(o=observation_id.lower()) for fp in GOOD_FILE_PATTERNS['rass']['processed']]

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
                # Open and decompress the events file - as the storage setup for RASS uses an old compression
                #  algorithm we have to use this specialised module to decompress
                decomp = unlzw3.unlzw(Path(raw_dir + down_file))

                # Open a new file handler for the decompressed data, and store the decompressed bytes there
                with open(raw_dir + stor_name, 'wb') as writo:
                    writo.write(decomp)
                # Then remove the tarred file to minimise storage usage
                os.remove(raw_dir + down_file)

        return None

    def download(self, num_cores: int = NUM_CORES, download_processed: bool = True):
        """
        A method to acquire and download the ROSAT All-Sky Survey data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded).

        Proprietary data is not a relevant concept for RASS, so no option to provide credentials is provided here
        as it is in some other mission classes.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine.
        :param bool download_processed: This controls whether the data downloaded are the pre-processed event lists
            stored by HEASArc, or whether they are the original raw event lists. Default is to download pre-processed
            data.
        """

        if not download_processed:
            raise NotImplementedError("The ability to download completely unprocessed RASS data has not been added "
                                      "yet, mainly due to confusion about the location of the data and whether the "
                                      "software to process it still exists.")

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

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in self.filtered_obs_ids:
                        # Use the internal static method I set up which both downloads and unpacks the RASS data
                        self._download_call(obs_id, raw_dir=stor_dir + '{o}'.format(o=obs_id),
                                            download_processed=download_processed)
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
                                               'download_processed': download_processed},
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
        raise NotImplementedError("The observation assessment process has not been implemented for ROSATAllSky.")
