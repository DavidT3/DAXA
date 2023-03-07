#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/03/2023, 18:06. Copyright (c) The Contributors
import io
from datetime import datetime
from typing import List, Union
from urllib.request import urlopen

import pandas as pd
from astropy.coordinates import BaseRADecFrame, FK5
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

from daxa.mission.base import BaseMission


class NuSTARPointed(BaseMission):
    """
    spacecraft mode inertial only
    """

    def __init__(self, insts: Union[List[str], str] = None):
        super().__init__()

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
        self._miss_name = "nustar"
        # This won't be used to name directories, but will be used for things like progress bar descriptions
        self._pretty_miss_name = "NuSTAR"
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

    def fetch_obs_info(self):
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
                      'SPACECRAFT_MODE']
        # This is what will be put into the URL to retrieve just those data fields - there are quite a few more
        #  but I curated it to only those I think might be useful for DAXA
        fields = '&Fields=' + '&varon=' + '&varon='.join(which_cols)

        # The full URL that we will pull the data from, with all the components we have previously defined
        fetch_url = host_url + table_head + action + result_max + down_form + fields

        # Opening that URL, we can access the results of our request!
        with urlopen(fetch_url) as urlo:
            # This opens the data as using the astropy fits interface (using io.BytesIO() to stream it into memory
            #  first so that fits.open can access it as an already opened file handler).
            with fits.open(io.BytesIO(urlo.read())) as full_fits:
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
                                                'public_date': 'proprietary_end_date'})

        # print(rel_nustar[rel_nustar['proprietary_end_date'] == 0])
        # stop

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
        rel_nustar.loc[val_end_dates, 'proprietary_end_date'] = pd.to_datetime(
            Time(rel_nustar.loc[val_end_dates, 'proprietary_end_date'].values, format='mjd', scale='utc').to_datetime())
        rel_nustar.loc[~val_end_dates, 'proprietary_end_date'] = pd.NaT
        # Grab the current date and time
        today = datetime.today()
        # Create a boolean column that describes whether the data are out of their proprietary period yet
        rel_nustar['usable_proprietary'] = rel_nustar['proprietary_end_date'].apply(lambda x:
                                                                                    ((x <= today) &
                                                                                     (pd.notnull(x)))).astype(bool)

        # I was going to use the 'issue_flag' column as a way of deciding scientific viability, but over 1500
        #  observations are marked '1' (for an issue) and I don't really want to exclude that many out of hand so
        #  I will just make anything public usable for now.
        rel_nustar['usable'] = rel_nustar['usable_proprietary']

        # Re-ordering the table, and not including certain columns which have served their purpose
        rel_nustar = rel_nustar[['ra', 'dec', 'ObsID', 'usable', 'start', 'end', 'duration',
                                 'proprietary_end_date', 'exposure_a', 'exposure_b', 'ontime_a', 'ontime_b',
                                 'nupsdout', 'issue_flag']]

        # Reset the dataframe index, as some rows will have been removed and the index should be consistent with how
        #  the user would expect from  a fresh dataframe
        rel_nustar = rel_nustar.reset_index(drop=True)

        # Use the setter for all_obs_info to actually add this information to the instance
        self.all_obs_info = rel_nustar

    @staticmethod
    def _download_call(observation_id: str, insts: List[str], level: str, filename: str):
        pass

    def download(self):
        pass
