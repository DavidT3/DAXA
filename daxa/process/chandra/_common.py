#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 17/10/2024, 14:22. Copyright (c) The Contributors

from typing import Tuple
from warnings import warn

from packaging.version import Version

from daxa.archive.base import Archive
from daxa.exceptions import NoValidMissionsError
from daxa.process._backend_check import find_ciao
from daxa.process._common import create_dirs

ALLOWED_CHANDRA_MISSIONS = ['chandra']


def _ciao_process_setup(obs_archive: Archive) -> Tuple[Version, Version]:
    """
    This function is to be called at the beginning of CIAO specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :return: The version numbers of the CIAO and CALDB installs located on the system.
    :rtype: Tuple[Version, Version]
    """
    # This makes sure that CIAO is installed on the host system, and also identifies the version
    ciao_vers, caldb_vers = find_ciao()

    if not isinstance(obs_archive, Archive):
        raise TypeError('The passed obs_archive must be an instance of the Archive class, which is made up of one '
                        'or more mission class instances.')

    # Now we ensure that the passed observation archive actually contains a Chandra mission (only one Chandra
    #  mission currently and I can't see there being any more added, but still good to be general)
    chandra_miss = [mission for mission in obs_archive if mission.name in ALLOWED_CHANDRA_MISSIONS]
    if len(chandra_miss) == 0:
        raise NoValidMissionsError("None of the missions that make up the passed observation archive are "
                                   "Chandra missions, and thus this Chandra-specific function cannot continue.")
    else:
        processed = [ch.processed for ch in chandra_miss]
        if any(processed):
            warn("One or more Chandra missions have already been fully processed", stacklevel=2)

    # This bit creates the storage directories for Chandra missions
    for miss in chandra_miss:
        create_dirs(obs_archive, miss.name)

    return ciao_vers, caldb_vers
