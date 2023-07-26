#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 26/07/2023, 04:41. Copyright (c) The Contributors

import io

import pandas as pd
import requests
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from daxa.mission.base import BaseMission


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

        # # Important first step, select only 'science mode' observations, slew observations will be dealt with in
        # #  another class - this includes excluding observations with 'STELLAR' spacecraft mode, as they are likely
        # #  rotating. Also select only those observations which have actually been taken (this table contains
        # #  planned observations as well).
        # rel_rass = full_nustar[(full_nustar['OBSERVATION_MODE'] == 'SCIENCE') &
        #                          (full_nustar['SPACECRAFT_MODE'] == 'INERTIAL') &
        #                          (full_nustar['STATUS'].isin(['processed', 'archived']))]

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

        # There is not target information because this is an all sky survey, but I have actually added an 'all sky
        #  survey' target type to the DAXA taxonomy. So we'll set all the observations to that
        full_rass['target_category'] = 'ASK'

        # Re-ordering the table, and not including certain columns which have served their purpose
        full_rass = full_rass[['ra', 'dec', 'ObsID', 'science_usable', 'start', 'end', 'duration', 'target_category',
                               'exposure_a', 'exposure_b']]

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = full_rass

    def download(self):
        pass

    def assess_process_obs(self, obs_info: dict):
        pass