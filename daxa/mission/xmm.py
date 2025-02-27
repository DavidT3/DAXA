#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 27/02/2025, 10:53. Copyright (c) The Contributors
import gzip
import io
import os.path
import tarfile
from datetime import datetime
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
from astroquery import log
from astroquery.esa.xmm_newton import XMMNewton as AQXMMNewton
from bs4 import BeautifulSoup
from daxa import NUM_CORES
from daxa.exceptions import DAXADownloadError
from daxa.mission.base import BaseMission
from tqdm import tqdm

log.setLevel(0)


class XMMPointed(BaseMission):
    """
    The mission class for pointed XMM observations (i.e. slewing observations are NOT included in the data accessed
    and collected by instances of this class). The available observation information is fetched from the XMM Science
    Archive using AstroQuery, and data are downloaded with the same module.

    :param List[str]/str insts: The instruments that the user is choosing to download/process data from. The EPIC PN,
        MOS1, and MOS2 instruments are selected by default. You may also select RGS1 (R1) and RGS2 (R2), though
        as they less widely used they are not selected by default. It is also possible to select the
        Optical Monitor (OM), though it is an optical/UV telescope, and as such it is not selected by default.
    :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
        state of a previously defined mission (the same filters having been applied etc.)
    :param bool use_heasarc: A boolean argument that controls how XMM data are acquired. The default
        value (False) means that the AstroQuery module will be used to fetch information/data from XSA, and
        setting it to True will tell this mission to acquire information/data from HEASARC instead. We find
        that some HPC configurations don't allow AstroQuery to work properly, in which case this argument will
        be automatically set to True, and data will be acquired from HEASARC instead.
    """
    def __init__(self, insts: Union[List[str], str] = None, save_file_path: str = None, use_heasarc: bool = False):
        """
        The mission class init for pointed XMM observations (i.e. slewing observations are NOT included in the data
        accessed and collected by instances of this class). The available observation information is fetched from
        the XMM Science Archive using AstroQuery, and data are downloaded with the same module.

        :param List[str]/str insts: The instruments that the user is choosing to download/process data from. The EPIC
            PN, MOS1, and MOS2 instruments are selected by default. You may also select RGS1 (R1) and RGS2
            (R2), though as they less widely used they are not selected by default. It is also possible to select the
            Optical Monitor (OM), though it is an optical/UV telescope, and as such it is not selected by default.
        :param str save_file_path: An optional argument that can use a DAXA mission class save file to recreate the
            state of a previously defined mission (the same filters having been applied etc.)
        :param bool use_heasarc: A boolean argument that controls how XMM data are acquired. The default
            value (False) means that the AstroQuery module will be used to fetch information/data from XSA, and
            setting it to True will tell this mission to acquire information/data from HEASARC instead. We find
            that some HPC configurations don't allow AstroQuery to work properly, in which case this argument will
            be automatically set to True, and data will be acquired from HEASARC instead.
        """
        # Call the init of parent class with the required information
        super().__init__()

        # Sets the default instruments
        if insts is None:
            insts = ['M1', 'M2', 'PN']
        elif isinstance(insts, str):
            # Makes sure that, if a single instrument is passed as a string, the insts variable is a list for the
            #  rest of the work done using it
            insts = [insts]
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

        # Call the name property to set up the name and pretty name attributes
        self.name

        # This sets up extra columns which are expected to be present in the all_obs_info pandas dataframe
        self._required_mission_specific_cols = ['proprietary_end_date', 'proprietary_usable', 'science_usable',
                                                'revolution']

        # This attribute is only present in XMM mission classes, as sometimes AstroQuery can have a hard time with
        #  some HPC setups, and we wish to fail over to HEASARC data acquisition. If that happens we wish to keep
        #  a record of it in this attribute, AND to allow the user to manually control whether they want to use
        #  HEASARC instead of AstroQuery
        self._use_heasarc = use_heasarc

        # Runs the method which fetches information on all available pointed XMM observations and stores that
        #  information in the all_obs_info property
        self._fetch_obs_info()

        # Slightly cheesy way of setting the _filter_allowed attribute to be an array identical to the usable
        #  column of all_obs_info, rather than the initial None value
        self.reset_filter()

        # We now will read in the previous state, if there is one to be read in.
        if save_file_path is not None:
            self._load_state(save_file_path)

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
        #  new mission class
        self._approx_fov = Quantity(15, 'arcmin')
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
        self._obs_info_checks(new_info)
        self._obs_info = new_info
        self.reset_filter()

    # Then define user-facing methods
    def _fetch_obs_info(self):
        """
        This method uses the AstroQuery table access protocol implemented for the XMM Science Archive to pull
        down information on all of the pointed XMM observations which are stored in XSA. The data are processed
        into a Pandas dataframe and stored.
        """
        def aq_acquisition() -> pd.DataFrame:
            # First of all I want to know how many entries there are in the 'all observations' table, because
            #  I need to specify the number of rows to select in my ADQL (Astronomical Data Query Language) command
            #  for reasons I'll explain in a second
            count_tab = AQXMMNewton.query_xsa_tap('select count(observation_id) from xsa.v_all_observations')

            # Then I round up to the nearest 1000, probably unnecessary but oh well
            num_obs = np.ceil(count_tab['COUNT'].tolist()[0] / 1000).astype(int) * 1000
            # Now I have to be a bit cheesy - If I used select * (which is what I would normally do in an SQL-derived
            #  language to grab every row) it actually only returns the top 2000. I think that * is replaced with
            #  TOP 2000 before the query is sent to the server. However if I specify a TOP N, where N is greater
            #  than 2000, then it works as intended. I hope this is a stable behaviour!
            obs_info = AQXMMNewton.query_xsa_tap("select TOP {} ra, dec, observation_id, start_utc, with_science, "
                                                 "duration, proprietary_end_date, revolution "
                                                 "from v_all_observations".format(num_obs))
            # The above command has gotten some basic information; central coordinates, observation ID, start time
            #  and duration, whether the data are proprietary etc. Now this Astropy table object is turned into a
            #  Pandas dataframe (which I much prefer working with).
            rel_df: pd.DataFrame = obs_info.to_pandas()

            # Convert the string representation of proprietary period ending into a datetime object. I have to use
            #  errors='coerce' here because for some reason some proprietary end times are set ~1000 years in
            #  the future, which Pandas implementation of datetime does not like. Errors coerce means that such
            #  datetimes are just set to NaT (not a time) rather than erroring everything out.
            rel_df['proprietary_end_date'] = pd.to_datetime(rel_df['proprietary_end_date'], utc=False, errors='coerce')
            # Convert the start time to a datetime
            rel_df['start_utc'] = pd.to_datetime(rel_df['start_utc'], utc=False, errors='coerce')

            return rel_df

        def heasarc_acquisition() -> pd.DataFrame:

            # This is the web interface for querying NASA HEASArc catalogues
            host_url = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?"

            # This returns the requested information in a FITS format - the idea being I will stream this into memory
            #  and then have a fits table that I can convert into a Pandas dataframe (which I much prefer working with).
            down_form = "&displaymode=FitsDisplay"
            # This should mean unlimited, as we don't know how many XMM observations there are, and the number will
            #  increase with time (so long as the telescope doesn't break...)
            result_max = "&ResultMax=0"
            # This just tells the interface it's a query (I think?)
            action = "&Action=Query"
            # Tells the interface that I want to retrieve from the xmmmaster (XMM Master) catalogue
            table_head = "tablehead=name=BATCHRETRIEVALCATALOG_2.0%20xmmmaster"

            # The definition of all of these fields can be found here:
            #  (https://heasarc.gsfc.nasa.gov/W3Browse/xmm-newton/xmmmaster.html)
            which_cols = ['RA', 'DEC', 'TIME', 'END_TIME', 'OBSID', 'STATUS', 'DURATION', 'PUBLIC_DATE',
                          'DATA_IN_HEASARC', 'XMM_REVOLUTION']

            # This is what will be put into the URL to retrieve just those data fields - there are quite a few more
            #  but I curated it to only those I think might be useful for DAXA
            fields = '&Fields=' + '&varon=' + '&varon='.join(which_cols)

            # A fix to find only those entries with valid coordinates - we found that the HEASARC table has two
            # entries with invalid RA and DEC values (just error) which really throw off the fits reader. We'll
            # contact them to fix it, but this just performs a search across all the sky, which in turn picks up
            # all entries with valid positions
            pos = "&Coordinates=%27Equatorial%3a+R%2eA%2e+Dec%27&Equinox=2000&Radius=21600arcmin&Entry=0%2e0%2c0%2e0&"

            # The full URL that we will pull the data from, with all the components we have previously defined
            fetch_url = host_url + table_head + action + result_max + down_form + fields + pos

            # Opening that URL, we can access the results of our request!
            with requests.get(fetch_url, stream=True) as urlo:
                # This opens the data as using the astropy fits interface (using io.BytesIO() to stream it into memory
                #  first so that fits.open can access it as an already opened file handler).
                with fits.open(io.BytesIO(urlo.content)) as full_fits:
                    # Then convert the data in that fits file just into an astropy table object, and from there to a DF
                    full_xmm = Table(full_fits[1].data).to_pandas()
                    # This cycles through any column with the 'object' data type (string in this instance), and
                    #  strips it of white space (I noticed there was extra whitespace on the end of a lot of the
                    #  string data).
                    for col in full_xmm.select_dtypes(['object']).columns:
                        full_xmm[col] = full_xmm[col].apply(lambda x: x.strip())

            # This removes entries for XMM observations that aren't relevant to this class - most importantly the
            #  XMM slew observations (ObsIDs beginning with '9'), and any data that haven't been taken yet
            rel_xmm = full_xmm[(~full_xmm['OBSID'].str.startswith('9')) &
                               (full_xmm['STATUS'].isin(['processed', 'archived']))]

            # Lower-casing all the column names (personal preference largely).
            rel_xmm = rel_xmm.rename(columns=str.lower)
            # Changing a few column names to match what BaseMission expects
            rel_xmm = rel_xmm.rename(columns={'obsid': 'ObsID', 'time': 'start_utc',
                                              'public_date': 'proprietary_end_date',
                                              'xmm_revolution': 'revolution',
                                              'end_time': 'end'})

            # Remove the status column, we're done with it!
            del rel_xmm['status']
            # Also remove the search_offset_ column, a by-product of the bodge to get rid of the two messed up
            #  entries in the HEASARC table
            del rel_xmm['search_offset_']

            # The HEASARC table doesn't have a 'science usable' like XSA does, so unfortunately we just set the
            #  column to be True for every observation
            rel_xmm['science_usable'] = True

            # Also convert the 'data in HEASARC' column to a boolean value, rather than 'Y' or 'N'
            rel_xmm['data_in_heasarc'] = rel_xmm['data_in_heasarc'].apply(lambda x: True if x == 'Y' else False)

            # These times are in a different format to those acquired through HEASARC - so we convert them here
            rel_xmm['start_utc'] = pd.to_datetime(Time(rel_xmm['start_utc'].values.astype(float).round(4), format='mjd',
                                                       scale='utc').to_datetime())
            rel_xmm['end'] = pd.to_datetime(Time(rel_xmm['end'].values.astype(float).round(4), format='mjd',
                                                 scale='utc').to_datetime())

            # Slightly more complicated with the public release dates, as some of them are set to 0 MJD, which makes the
            #  conversion routine quite upset (0 is not valid) - as such I convert only those which aren't 0, then
            #  replace the 0 valued ones with Pandas' Not a Time (NaT) value
            val_end_dates = (rel_xmm['proprietary_end_date'] != 0) & (rel_xmm['proprietary_end_date'] != 416422)

            # We make a copy of the proprietary end dates to work on because pandas will soon not allow us to set what
            #  was previously an int column with a datetime - so we need to convert it but retain the original
            #  data as well
            prop_end_dates = rel_xmm['proprietary_end_date'].copy()
            # Convert the original column to datetime
            rel_xmm['proprietary_end_date'] = rel_xmm['proprietary_end_date'].astype('datetime64[ns]')
            rel_xmm.loc[val_end_dates, 'proprietary_end_date'] = pd.to_datetime(
                Time(prop_end_dates[val_end_dates.values], format='mjd', scale='utc').to_datetime(), errors='coerce')

            rel_xmm.loc[~val_end_dates, 'proprietary_end_date'] = pd.NaT

            return rel_xmm

        # We have found that some HPC compute nodes don't allow AstroQuery to download anything, because of some
        #  proxy configuration(?) - ultimately it is more of an issue with how the HPC is set up than with
        #  AstroQuery, but we still want to be able to deal with it. As such, we check to see if AstroQuery can
        #  get the observation info table, and if not we switch over to a HEASARC-based method. We also allow the user
        #  to select which data source to use when defining the mission instance.
        if not self._use_heasarc:
            try:
                obs_info_pd = aq_acquisition()
            except OSError:
                warn("Astroquery is not able to connect to the XMM Science Archive, switching to using "
                     "the HEASARC XMM-Newton archive.", stacklevel=2)
                self._use_heasarc = True
                obs_info_pd = heasarc_acquisition()
        else:
            obs_info_pd = heasarc_acquisition()

        # Grab the current date and time
        today = datetime.today()

        # This adds a column that describes whether the data are out of their proprietary period, and thus
        #  usable by the general community. Can just use less than or equal to operator because everything involved
        #  is now a datetime object.
        obs_info_pd['proprietary_usable'] = obs_info_pd['proprietary_end_date'].apply(
            lambda x: ((x <= today) & (pd.notnull(x)))).astype(bool)
        # We have to do one extra check for the observation info table assembled from HEASARC (if AstroQuery has had
        #  issues or if the user specified that they wished to use HEASARC) - there is a column that specifies
        #  whether the data are actually in HEASARC yet, and we'll set them to proprietary usable False if the
        #  data aren't there.
        if self._use_heasarc:
            obs_info_pd['proprietary_usable'] *= obs_info_pd['data_in_heasarc']
            # Don't need this anymore!
            del obs_info_pd['data_in_heasarc']

        # Just renaming some of the columns
        obs_info_pd = obs_info_pd.rename(columns={'observation_id': 'ObsID', 'with_science': 'science_usable',
                                                  'start_utc': 'start'})

        # Converting the duration column to a timedelta object, which can then be directly added to the start column
        #  which should be a datetime object itself
        obs_info_pd['duration'] = pd.to_timedelta(obs_info_pd['duration'], 's')
        # Now creating an end column by adding duration to start
        obs_info_pd['end'] = obs_info_pd.apply(lambda x: x.start + x.duration, axis=1)

        # This checks for NaN values of RA or Dec, which for some reason do appear sometimes??
        obs_info_pd['radec_good'] = obs_info_pd.apply(lambda x: bool((np.isfinite(x['ra']) & np.isfinite(x['dec'])) &
                                                                     (~((x['ra'] == 0) & (x['dec'] == 0)))), axis=1)

        # Throws a warning if there are some.
        if len(obs_info_pd) != obs_info_pd['radec_good'].sum():
            warn("{ta} of the {tot} observations located for this mission have been removed due to NaN "
                 "RA or Dec values".format(ta=len(obs_info_pd)-obs_info_pd['radec_good'].sum(), tot=len(obs_info_pd)),
                 stacklevel=2)
        # Cut the total information down to just those that don't have NaN positions. I've done it this way rather
        #  than adding the radec_good column as another input to the usable column (see below) because having NaN
        #  positions really screws up the filter_on_positions method in BaseMission
        obs_info_pd = obs_info_pd[obs_info_pd['radec_good']]
        # Don't really care about this column now so remove
        del obs_info_pd['radec_good']

        # This just resets the index, as some of the rows may have been removed
        obs_info_pd = obs_info_pd.reset_index(drop=True)

        # Putting the columns in the order we want
        obs_info_pd = obs_info_pd[['ra', 'dec', 'ObsID', 'start', 'science_usable', 'duration',
                                   'proprietary_end_date', 'revolution', 'proprietary_usable', 'end']]

        self.all_obs_info = obs_info_pd

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], level: str, filename: str, use_heasarc: bool):
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
        # The astroquery module will only download to the cwd, so need to change to where we want the data downloaded
        og_dir = os.getcwd()  # storing this info so we can return here after the download is done
        dest_dir_path = filename.replace(observation_id, '')  # this is a static method, so I cant use self.raw_data_path
        os.chdir(dest_dir_path)  # changing to where we want the file downloaded
        
        # Another part of the very unsophisticated method I currently have for checking whether a raw XMM data
        #  download has already been performed (see issue #30). If the ObsID directory doesn't exist then
        #  an attempt will be made.
        if not os.path.exists(filename):
            # Set this again here because otherwise its annoyingly verbose
            log.setLevel(0)

            # It is possible for a download to be interrupted and the incomplete tar.gz to hand around and cause
            #  us problems, so we check and delete the offending tar.gz if it is present
            if os.path.exists(filename+'.tar.gz'):
                os.remove(filename+'.tar.gz')

            # There are two different ways this class can get its data - from XSA via AstroQuery, or from HEASARC. Here
            #  we have to add downloading methods for both possibilities. Firstly, if we're just using AstroQuery
            if not use_heasarc:
                try:
                    # Download the requested data
                    AQXMMNewton.download_data(observation_id=observation_id, level=level, filename=filename)
                except Exception as err:
                    os.chdir(og_dir)  # if an error is raised we still need to return to the original dir
                    raise Exception("{oi} data failed to "
                                    "download.".format(oi=observation_id)).with_traceback(err.__traceback__)
                # As the above function downloads the data as compressed tars, we need to decompress them
                with tarfile.open(filename+'.tar.gz') as zippo:
                    zippo.extractall(filename)

                # Then remove the original compressed tar to save space
                os.remove(filename + '.tar.gz')

                # Finally, the actual telescope data is in another .tar, so we expand that as well, first making
                #  sure that there is only one tar in the observations folder
                rel_tars = [f for f in os.listdir(filename) if "{o}.tar".format(o=observation_id) in f.lower()]

                # Checks to make sure there is only one tarred file (otherwise I don't know what this will be
                #  unzipping)
                if len(rel_tars) == 0 or len(rel_tars) > 1:
                    os.chdir(og_dir)  # if an error is raised we still need to return to the original dir
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
                              and f.split(observation_id + '_')[1][:2] not in to_keep]
                for for_removal in throw_away:
                    os.remove(untar_path + for_removal)

            # Now write the OTHER way of downloading XMM data, which follows the pattern established for all the DAXA
            #  missions that pull from HEASARC (the vast majority of them).
            else:
                # This opens a session that will persist
                session = requests.Session()
                h_url = "https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/{oi}/ODF/".format(oi=observation_id)

                # Unlike with the astroquery method, we can exclude files for instruments that aren't selected
                #  before we download them (with astroquery we deleted them after the fact)
                to_keep = insts + ['SC']
                to_down = [en['href'] for en in BeautifulSoup(session.get(h_url).text, "html.parser").find_all("a")
                           if any([observation_id + "_" + tk in en['href'] for tk in to_keep])
                           or 'MANIFEST' in en['href']]

                # Make sure that the local directory is created
                if not os.path.exists(filename):
                    os.makedirs(filename)

                # And now we can loop through our files of interest, and download them into a raw XMM data directory
                for down_file in to_down:
                    down_url = h_url + down_file
                    with session.get(down_url, stream=True) as acquiro:
                        with open(os.path.join(filename, down_file), 'wb') as writo:
                            copyfileobj(acquiro.raw, writo)

                    # Open and decompress the files that need it
                    if 'gz' in down_file:
                        with gzip.open(os.path.join(filename, down_file), 'rb') as compresso:
                            # Open a new file handler for the decompressed data, then funnel the decompressed contents there
                            with open(os.path.join(filename, down_file).split('.gz')[0], 'wb') as writo:
                                copyfileobj(compresso, writo)
                        # Then remove the tarred file to minimise storage usage
                        os.remove(os.path.join(filename, down_file))

        os.chdir(og_dir)  # returning to the original working dir

        return None

    def download(self, num_cores: int = NUM_CORES, credentials: Union[dict, str] = None,
                 download_products: bool = False):
        """
        A method to acquire and download the pointed XMM data that have not been filtered out (if a filter
        has been applied, otherwise all data will be downloaded). Instruments specified by the chosen_instruments
        property will be downloaded, which is set either on declaration of the class instance or by passing
        a new value to the chosen_instruments property.

        :param int num_cores: The number of cores that can be used to parallelise downloading the data. Default is
            the value of NUM_CORES, specified in the configuration file, or if that hasn't been set then 90%
            of the cores available on the current machine. The number of cores will be CAPPED AT 10 FOR XMM - we
            have experienced reliably dropped connections when more than 10 download processes are created.
        :param dict/str credentials: The path to an ini file containing credentials, a dictionary containing 'user'
            and 'password' entries, or a dictionary of ObsID top level keys, with 'user' and 'password' entries
            for providing different credentials for different observations.
        :param bool download_products: CURRENTLY NON-FUNCTIONAL.
        """

        if credentials is not None and not self.filtered_obs_info['proprietary_usable'].all():
            raise NotImplementedError("Support for credentials for proprietary data is not yet implemented.")
        elif not self.filtered_obs_info['proprietary_usable'].all() and credentials is None:
            warn("Proprietary data have been selected, but no credentials provided; as such the proprietary data have "
                 "been excluded from download and further processing.", stacklevel=2)
            new_filter = self.filter_array * self.all_obs_info['proprietary_usable'].values
            self.filter_array = new_filter

        # Ensures that a directory to store the 'raw' pointed XMM data in exists - once downloaded and unpacked
        #  this data will be processed into a DAXA 'archive' and stored elsewhere.
        if not os.path.exists(self.top_level_path + self.name + '_raw'):
            os.makedirs(self.top_level_path + self.name + '_raw')
        # Grabs the raw data storage path
        stor_dir = self.raw_data_path

        # This XMM mission currently only supports the downloading of raw data - I should try to address that
        self._download_type = 'raw'

        # A very unsophisticated way of checking whether raw data have been downloaded before (see issue #30)
        #  If not all data have been downloaded there are also secondary checks on an ObsID by ObsID basis in
        #  the _download_call method
        if all([os.path.exists(stor_dir + '{o}'.format(o=o)) for o in self.filtered_obs_ids]):
            self._download_done = True

        # I have found that having more than 10 simultaneous XMM download processes tends to result in dropped
        #  connections and having to re-run mission downloads, so I am going to enforce that on the num_cores
        #  argument here for a better user experience
        new_num_cores = min([num_cores, 10])
        if new_num_cores != num_cores:
            warn("The number of cores assigned to XMMPointed downloads has been capped at 10, this will minimise "
                 "dropped connections.", stacklevel=2)
            num_cores = new_num_cores

        if not self._download_done:
            # If only one core is to be used, then it's simply a case of a nested loop through ObsIDs and instruments
            if num_cores == 1:
                with tqdm(total=len(self), desc="Downloading {} data".format(self._pretty_miss_name)) as download_prog:
                    for obs_id in self.filtered_obs_ids:
                        # Use the internal static method I set up which both downloads and unpacks the XMM data
                        self._download_call(obs_id, insts=self.chosen_instruments, level='ODF',
                                            filename=stor_dir + '{o}'.format(o=obs_id), use_heasarc=self._use_heasarc)
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
                                               'level': 'ODF', 'filename': stor_dir + '{o}'.format(o=obs_id),
                                               'use_heasarc': self._use_heasarc},
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

    def assess_process_obs(self, obs_info: dict) -> dict:
        """
        A slightly unusual method which will allow the XMMPointed mission to assess the information on a particular
        observation that has been put together by an Archive (the archive assembles it because sometimes this
        detailed information only becomes available at the first stages of processing), and make a decision on whether
        that particular observation-instrument-subexposure should be processed further for scientific use.

        This method should never need to be triggered by the user, as it will be called automatically when detailed
        observation information becomes available to the Archive.

        :param dict obs_info: The multi-level dictionary containing available observation information for an
            observation.
        :return: A two-level dictionary with instruments as the top level keys, and sub-exposure IDs as the low
            level keys. The values associated with the sub-exposure keys are boolean, True for usable, False for not.
        :rtype: dict
        """
        insts = list(obs_info.keys())

        # This is the dictionary which we'll be sending back, with top level instrument keys and lower level sub
        #  exposure keys. We start off by assuming all of the data should be used, leaving it to the rest of this
        #  method to disable sub-exposures and set these booleans to False. This will only trigger for those
        #  instruments which have an exposures entry, and thus are implicitly active
        to_return = {inst: {e_id: True for e_id in list(obs_info[inst]['exposures'].keys())} for inst in insts
                     if 'exposures' in obs_info[inst]}

        for inst in to_return:
            # One check later on needs to know whether the current instrument is an RGS
            rgs = inst == 'R1' or inst == 'R2'

            for e_id in to_return[inst]:
                rel_info = obs_info[inst]['exposures'][e_id]

                # Observation type check, we want to ensure that the type for the observation is 'SCIENCE'
                if rel_info['type'] != 'SCIENCE':
                    to_return[inst][e_id] = False

                # Filter check - we want to exclude observations where the filter is in CalClosed (or Closed, as I
                #  saw it called in one observation), or where the filter is None (apart from RGS, no filters there).
                #  The Blocked filter that I'm excluding should only ever belong to the optical monitor, and
                #  though I haven't decided if DAXA will ever reduce OM data, I'm including it here for future proofing.
                if not rgs and (rel_info['filter'] is None or rel_info['filter']
                                in ['CalClosed', 'Closed', 'Blocked', 'Unknown']):
                    to_return[inst][e_id] = False

                # Now we check the observing mode, there are quite a few that are non-science modes which don't
                #  need to be reduced for scientific purposes. The list I'm using for this is on this website:
                #  https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/dfhb/node72.html
                # CentroidingConfirmation and subsequent modes are all for OM, again not sure if I'll ever implement
                #  OM but trying to be thorough
                bad_modes = ['Diagnostic3x3', 'Diagnostic1x1', 'CcdDiagnostic', 'Diagnostic1x1ResetPerPixel',
                             'Diagnostic', 'Noise', 'Offset', 'HighTimeResolutionSingleCcd', 'OffsetVariance',
                             'CentroidingConfirmation', 'CentroidingData', 'DarkHigh', 'DarkLow', 'FlatFieldHigh',
                             'FlatFieldLow']
                # Performs the actual mode check
                if rel_info['mode'] in bad_modes:
                    to_return[inst][e_id] = False

        # I do want entries for the instruments which aren't active, even if they are just empty
        to_return.update({inst: {} for inst in insts if 'exposures' not in obs_info[inst]})

        return to_return

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
        # The XMM unique processing identifiers can take a few forms - the simplest is just the ObsID, the second
        #  simplest is ObsID + two character instrument identifier (e.g. 0201903501PN), and the final layer of
        #  complexity is where there is a sub-exposure identifier as well (e.g. 0201903501PNS003).
        # Thankfully, all XMM ObsIDs are 10 digits long, so we're just going to take the first 10 digits
        return ident[:10]
