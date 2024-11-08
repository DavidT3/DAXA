#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/11/2024, 22:43. Copyright (c) The Contributors

from typing import Tuple, List
from warnings import warn

from packaging.version import Version

from daxa import BaseMission
from daxa.archive.base import Archive
from daxa.exceptions import NoValidMissionsError
from daxa.process._backend_check import find_nustardas
from daxa.process._common import create_dirs

ALLOWED_NUSTAR_MISSIONS = ['nustar_pointed']


def _nustardas_process_setup(obs_archive: Archive,
                             make_dirs: bool = True) -> Tuple[Version, Version, List[BaseMission]]:
    """
    This function is to be called at the beginning of NuSTARDAS specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :param bool make_dirs: A boolean variable that controls whether the setup process should ensure that the
        storage directories for the future processed NuSTAR data are made or not. Default is True.
    :return: The version numbers of the NuSTARDAS and NuSTAR CALDB installs located on the system, as well as the
        relevant missions.
    :rtype: Tuple[Version, Version, List[BaseMission]]
    """
    # This makes sure that NuSTARDAS is installed on the host system, and also identifies the version
    nudas_vers, caldb_vers = find_nustardas()

    if not isinstance(obs_archive, Archive):
        raise TypeError('The passed obs_archive must be an instance of the Archive class, which is made up of one '
                        'or more mission class instances.')

    # Now we ensure that the passed observation archive actually contains a NuSTAR mission - slew data are collected
    #  by NuSTAR in addition to pointed data, so it is possible there could be multiple missions
    nustar_miss = [mission for mission in obs_archive if mission.name in ALLOWED_NUSTAR_MISSIONS]
    if len(nustar_miss) == 0:
        raise NoValidMissionsError("None of the missions that make up the passed observation archive are "
                                   "NuSTAR missions, and thus this NuSTAR-specific function cannot continue.")
    else:
        processed = [nu.processed for nu in nustar_miss]
        if any(processed):
            warn("One or more NuSTAR missions have already been fully processed", stacklevel=2)

    # We allow the option of not making directories, but the default is that we will
    if make_dirs:
        # This bit creates the storage directories for NuSTAR missions
        for miss in nustar_miss:
            create_dirs(obs_archive, miss.name)

    return nudas_vers, caldb_vers, nustar_miss