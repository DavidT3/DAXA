from typing import Bool
from warnings import warn
import os.path

from daxa.archive.base import Archive
from daxa.exceptions import NoEROSITAMissionsError
from daxa.process._backend_check import find_esass

ALLOWED_EROSITA_MISSIONS = ['erosita_calpv']

def _esass_process_setup(obs_archive: Archive) -> Bool:
    """
    This function is to be called at the beginning of eROSITA specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :return: A bool indicating whether or not eSASS is being used via Docker or not, set to True if Docker is being used. 
    :rtype: Bool
    """

    # This makes sure that SAS is installed on the host system, and also identifies the version
    esass_in_docker = find_esass()

    if not isinstance(obs_archive, Archive):
        raise TypeError('The passed obs_archive must be an instance of the Archive class, which is made up of one '
                        'or more mission class instances.')
    
    # Now we ensure that the passed observation archive actually contains XMM mission(s)
    erosita_miss = [mission for mission in obs_archive if mission.name in ALLOWED_EROSITA_MISSIONS]
    if len(erosita_miss) == 0:
        raise NoEROSITAMissionsError("None of the missions that make up the passed observation archive are "
                                 "eROSITA missions, and thus this eROSITA-specific function cannot continue.")
    else:
        processed = [em.processed for em in erosita_miss]
        if any(processed):
            warn("One or more eROSITA missions have already been fully processed")

    for miss in erosita_miss:
        # We make sure that the archive directory has folders to store the processed eROSITA data that will eventually
        #  be created by most functions that call this _esass_process_setup function
        for obs_id in miss.filtered_obs_ids:
            stor_dir = obs_archive.get_processed_data_path(miss, obs_id)
            if not os.path.exists(stor_dir):
                os.makedirs(stor_dir)

    return esass_in_docker

