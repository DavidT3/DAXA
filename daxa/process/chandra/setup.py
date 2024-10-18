#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 17/10/2024, 22:13. Copyright (c) The Contributors

import os

from astropy.io import fits
from astropy.table import Table

from daxa.archive import Archive
from daxa.process.chandra._common import _ciao_process_setup


def parse_oif(oif_path: str):
    """
    A function that takes a path to a Chandra 'oif.fits' file included in each ObsID directory on the remote
    archive. The file will be filtered and parsed so that data relevant to DAXA processing valid scientific
    observations can be extracted. This includes things like which mode the detector was in, whether a grating
    was deployed, etc.

    :param str oif_path: The path to the Chandra 'oif.fits' file that is to be parsed into a dictionary
        of relevant information.
    :return: Multi-level dictionary of information.
    :rtype: dict
    """

    # Firstly, lets get this observation index file loaded in, so we can start reading out the important information
    oif_file = fits.open(oif_path)

    # We want both the header and the table - the header will give us some instrument info
    oif_hdr = oif_file[1].header
    # Convert to pandas because I prefer working with dataframes
    oif_tbl = Table(oif_file[1].data).to_pandas()

    # Now we can pull out the useful information and start populating out observation info dictionary to return - set
    #  it up empty at first
    obs_info = {}
    # This feels somehow wrong, but we're just going to pull out the header keywords that I know are relevant D: Won't
    #  do a huge load of individual commands though, we'll set up a one-liner. We'll also set it up so the original
    #  header names can be converted to another name if we want to (None means the original name will be kept).
    hdr_to_store = {'SEQ_NUM': 'SEQUENCE', 'INSTRUME': 'INSTRUMENT', 'DETNAM': 'DETECTOR', 'GRATING': None,
                    'OBS_MODE': None, 'DATAMODE': 'MODE', 'RA_NOM': None, 'DEC_NOM': None, 'ROLL_NOM': None}
    obs_info.update({hdr_key if new_key is None else new_key: oif_hdr[hdr_key]
                     for hdr_key, new_key in hdr_to_store.items()})

    # ------------------- ANY MODIFICATION OF HEADER DATA HAPPENS HERE ------------------
    obs_info['DETECTOR'] = obs_info['DETECTOR'].split('-')[-1]
    # -----------------------------------------------------------------------------------

    # Now we move to examining the data file table - this should tell us what we really have to work with


    return obs_info


def prepare_chandra_info(archive: Archive):
    """
    A simple function that runs through all Chandra observations, reads through their file inventory, and includes
    that information in the Archive in a parsed, standardized form.

    :param Archive archive: A DAXA archive containing a Chandra mission.
    """

    # Check that the archive in question has got Chandra data etc.
    ciao_vers, caldb_vers, chandra_miss = _ciao_process_setup(archive, make_dirs=False)

    # This very simply iterates through the Chandra missions, and through all their ObsIDs, and parses the
    #  observation index files (assuming that is what 'oif' stands for?)
    obs_sums = {}
    for miss in chandra_miss:
        # Add an entry for the current mission
        obs_sums.setdefault(miss.name, {})
        for oi in miss.filtered_obs_ids:
            # This sets up the absolute path to the 'oif.fits' file for the current Chandra mission and ObsID
            cur_path = os.path.join(archive.top_level_path, oi, 'oif.fits')
            # The parsing function reads through that file and spits out the important information
            parsed_info = parse_oif(cur_path, oi)
            # Then we add it to the dictionary
            obs_sums[miss.name][oi] = parsed_info

    # Finally the fully populated dictionary is added to the archive - this will be what informs DAXA about
    #  which Chandra observations it can actually process into something useable
    archive.observation_summaries = obs_sums
