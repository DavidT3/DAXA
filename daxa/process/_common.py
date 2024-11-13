#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 13/11/2024, 12:57. Copyright (c) The Contributors

import glob
import os
import sys
from subprocess import Popen, PIPE, TimeoutExpired
from typing import Tuple, List
from warnings import warn

from daxa.archive import Archive


def execute_cmd(cmd: str, rel_id: str, miss_name: str, check_path: str, extra_info: dict,
                timeout: float = None) -> Tuple[str, str, List[bool], str, str, dict]:
    """
    This is a simple function designed to execute mission specific processing commands either through Docker or
    the command line for the processing and reduction of mission data. It will collect the stdout and stderr values
    for each command and return them too for the process of logging. Finally, it checks that a specified 'final file'
    (or a set of 'final files') actually exists at the expected path, as a final check of the success of whatever
    process has been run.

    :param str cmd: The command that should be executed in a bash shell.
    :param str rel_id: Whatever ID has been attached to the particular command (it could be an ObsID, or an ObsID
        + instrument combination depending on the task).
    :param str miss_name: The specific mission name that this task belongs to.
    :param str/list check_path: The path (or a list of paths) where a 'final file' (or final files) should exist, used
        for the purposes of checking that it (they) exists.
    :param dict extra_info: A dictionary which can contain extra information about the process or output that will
        eventually be stored in the Archive.
    :param float timeout: The length of time (in seconds) which the process is allowed to run for before being
        killed. Default is None, which is supported as an input by communicate().
    :return: The rel_id, a list of boolean flags indicating whether the final files exist, the std_out, and the
        std_err. The final dictionary can contain extra information recorded by the processing function.
    :rtype: Tuple[str, str, List[bool], str, str, dict]
    """
    # Either a single path or a list of paths can be passed to check - I make sure that the checking process only
    #  ever has to deal with a list
    if isinstance(check_path, str):
        check_path = [check_path]

    # ----------------------------- MISSION/PLATFORM SPECIFIC CHECKS LIVE HERE -----------------------------

    # This chunk is a fix for problems with eSASS (eROSITA package) finding the correct libraries on Apple ARM based
    #  systems, and just creates a new environment variable so it can locate them, if necessary
    sys_env = os.environ.copy()
    if sys.platform == 'darwin':
        if "LD_LIBRARY_PATH" in sys_env:
            cmd = f"export LD_LIBRARY_PATH={sys_env['LD_LIBRARY_PATH']} && {cmd}"
        if "DYLD_LIBRARY_PATH" in sys_env:
            cmd = f"export DYLD_LIBRARY_PATH={sys_env['DYLD_LIBRARY_PATH']} && {cmd}"

    # The eROSITA toolset eSASS is also available in a Docker container for macOS and Windows users
    # CURRENTLY we do not support it and actually an exception will be raised in the backend check, but we might
    #  support it in the future, in which case we may well need to do something differently here, and might need
    #  reminding of that by a handy not implemented error
    if 'esass_in_docker' in extra_info and extra_info['esass_in_docker']:
        raise NotImplementedError("The use of eSASS through Docker has not been implemented.")

    # If the mission we're working on here is Chandra, we create an extra command that should reset the
    #  parameter file values
    if 'chandra' in miss_name:
        extra_cmd = 'punlearn; '
    else:
        extra_cmd = ''
    # ------------------------------------------------------------------------------------------------------

    # Keep an original copy of this variable for later
    og_cmd = cmd
    # We're going to make a temporary pfiles directory which is a) local to the DAXA processing directory, and thus
    #  sure to be on the same filesystem (can be a performance issue for HPCs I think), and b) is unique to a
    #  particular process, so there shouldn't be any clashes. The temporary file name is randomly generated
    # Some processes may not need this however, in which case the working directory will be None

    if extra_info['working_dir'] is not None:
        # This just makes a new pfiles directory in our randomly generated working directory for this process
        new_pfiles = os.path.join(os.path.dirname(extra_info['working_dir']), 'pfiles/')
        os.makedirs(new_pfiles)

        # Now add the altered PFILES env variable to the beginning of the cmd - doesn't matter that
        #  I won't change it back as this spawns a new shell which then disappears at the end. Including the
        #  existing PFILES path entry is apparently very important - this path needs to contain the directory
        #  where all the blank template par files live, or it can't make a new one in the temporary directory
        # The HEADAS environment variables stop any processes trying to redirect to /dev/null
        cmd = ("export HEADASNOQUERY=; export HEADASPROMPT=/dev/null; " + extra_cmd +
               'export PFILES="{}:$PFILES"; '.format(new_pfiles) + cmd)

    # Starts the process running on a shell
    cmd_proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # This makes sure the process is killed if it does timeout
    try:
        out, err = cmd_proc.communicate(timeout=timeout)
    except TimeoutExpired:
        cmd_proc.kill()
        out, err = cmd_proc.communicate()
        warn("An {mn} process for {rid} has timed out".format(mn=miss_name, rid=rel_id), stacklevel=2)

    # Decodes the stdout and stderr from the binary encoding it currently exists in. The errors='ignore' flag
    #  means that it doesn't throw errors if there is a character it doesn't recognize
    out = out.decode("UTF-8", errors='ignore')
    err = err.decode("UTF-8", errors='ignore')

    # We also add the command string to the beginning of the stdout - this is for logging purposes
    out = cmd + '\n\n' + out

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


def create_dirs(obs_archive: Archive, miss_name: str):
    """
    This general function will set up the 'archive name'/processed_data/'ObsID' and 'archive name'/failed_data/'ObsID'
    directory structures in which the processed data for a mission (whether generated by DAXA implementations of
    missions processing, or pre-processed data downloaded from the mission data archive) are stored.

    :param Archive obs_archive: The Archive for which 'processed' and 'failed' data directories are to be created.
    :param str miss_name: The specific mission for which the 'processed' and 'failed' data directories are to
        be created. The mission name must refer to one of the missions associated with the Archive.
    """

    # We retrieve the actual mission object corresponding to the name we've been given
    miss = obs_archive[miss_name]

    # We make sure that the archive directory has folders to store each ObsID of the specified mission - the
    #  'processed_data' directories have all the ObsID directories initially, but any failed ObsIDs will be
    #  transferred over to the 'failed_data' directory which we create further down
    for obs_id in miss.filtered_obs_ids:
        stor_dir = obs_archive.construct_processed_data_path(miss, obs_id)
        if not os.path.exists(stor_dir):
            os.makedirs(stor_dir)

        # We also make a directory within the storage directory, specifically for logs
        if not os.path.exists(stor_dir + 'logs'):
            os.makedirs(stor_dir + 'logs')
        # Same deal but for different types of files that could be produced
        if not os.path.exists(stor_dir + 'images'):
            os.makedirs(stor_dir + 'images')
        if not os.path.exists(stor_dir + 'background'):
            os.makedirs(stor_dir + 'background')
        if not os.path.exists(stor_dir + 'events'):
            os.makedirs(stor_dir + 'events')
        if not os.path.exists(stor_dir + 'cleaning'):
            os.makedirs(stor_dir + 'cleaning')
        # As there can be a lot of variation between files generated for different telescopes, and what we want
        #  to keep may also differ between them, we also create a catch-all 'misc' directory
        if not os.path.exists(stor_dir + 'misc'):
            os.makedirs(stor_dir + 'misc')

    # We also ensure that an overall directory for failed processing observations exists - this will give
    #  observation directories which have no useful data in (i.e. they do not have a successful final
    #  processing step) somewhere to be moved to (see daxa.process._cleanup._last_process). Any pre-processed data
    #  downloaded from the online archives that somehow appears to have problems will also be moved over to the
    #  'failed_data' directory.
    fail_proc_dir = obs_archive.construct_failed_data_path(miss, None).format(oi='')[:-1]
    if not os.path.exists(fail_proc_dir):
        os.makedirs(fail_proc_dir)
