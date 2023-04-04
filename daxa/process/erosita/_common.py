import glob
from typing import Bool, Tuple, List
from warnings import warn
import os.path
from subprocess import Popen, PIPE

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

def execute_cmd(cmd: str, docker: Bool, rel_id: str, miss_name: str, check_path: str,
                extra_info: dict) -> Tuple[str, str, List[bool], str, str, dict]:
    """
    This is a simple function designed to execute eSASS commands either through Docker or the command line
    for the processing and reduction of eROSITA mission data. It will collect the stdout and stderr values 
    for each command and return them too for the process of logging. Finally, it checks that a specified 'final file' 
    (or a set of 'final files') actually exists at the expected path, as a final check of the success of whatever 
    process has been run.

    :param str cmd: The command that should be executed in a bash shell.
    :param Bool docker: Set to True if eSASS is being used via Docker.
    :param str rel_id: Whatever ID has been attached to the particular command (it could be an ObsID, or an ObsID
        + instrument combination depending on the task).
    :param str miss_name: The specific eROSITA mission name that this task belongs to.
    :param str/list check_path: The path (or a list of paths) where a 'final file' (or final files) should exist, used
        for the purposes of checking that it (they) exists.
    :param dict extra_info: A dictionary which can contain extra information about the process or output that will
        eventually be stored in the Archive.
    :return: The rel_id, a list of boolean flags indicating whether the final files exist, the std_out, and the
        std_err. The final dictionary can contain extra information recorded by the processing function.
    :rtype: Tuple[str, str, List[bool], str, str, dict]
    """
    # Either a single path or a list of paths can be passed to check - I make sure that the checking process only
    #  ever has to deal with a list
    if isinstance(check_path, str):
        check_path = [check_path]

    if docker:
        raise NotImplementedError("The use of eSASS through Docker has not been implemented.")

    # Starts the process running on a shell, connects to the process and waits for it to terminate, and collects
    #  the stdout and stderr
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    # Decodes the stdout and stderr from the binary encoding it currently exists in. The errors='ignore' flag
    #  means that it doesn't throw errors if there is a character it doesn't recognize
    out = out.decode("UTF-8", errors='ignore')
    err = err.decode("UTF-8", errors='ignore')

    # Simple check on whether the 'final file' passed into this function actually exists or not - even if there is only
    #  one path to check we made sure that its in a list so the check can be done easily for multiple paths
    files_exist = []
    for path in check_path:
        if '*' not in path and os.path.exists(path):
            files_exist.append(True)
        # In the case where a wildcard is in the final file path (I will try to make sure that this is avoided, but it
        #  is necessary right now) we use glob to find a match list and check to make sure there is at least one entry
        elif '*' in path and len(glob.glob(path)) > 0:
            files_exist.append(True)
        else:
            files_exist.append(False)

    return rel_id, miss_name, files_exist, out, err, extra_info