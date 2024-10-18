#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 17/10/2024, 19:38. Copyright (c) The Contributors
import os

from daxa.archive import Archive
from daxa.process.chandra._common import _ciao_process_setup


def parse_oif(sum_path: str, obs_id: str = None):
    """
    A function that takes a path to a Chandra 'oif.fits' file included in each ObsID directory on the remote
    archive. The file will be filtered and parsed so that data relevant to DAXA processing valid scientific
    observations can be extracted. This includes things like which mode the detector was in, whether a grating
    was deployed, etc.

    :param str sum_path: The path to the Chandra 'oif.fits' file that is to be parsed into a dictionary
        of relevant information.
    :param str obs_id: Optionally, the observation ID that goes with this summary file can be passed, purely to
        make a possible error message more useful.
    :return: Multi-level dictionary of information.
    :rtype: dict
    """
    # TODO ADD MORE INFORMATION TO THE 'return' PARAM IN THE DOCSTRING
    pass


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
