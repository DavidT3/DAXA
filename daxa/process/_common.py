#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 17/10/2024, 13:29. Copyright (c) The Contributors

import glob
import os.path
from subprocess import Popen, PIPE, TimeoutExpired
from typing import Tuple, List
from warnings import warn


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

    # ----------------------------- MISSION SPECIFIC CHECKS LIVE HERE -----------------------------

    # The eROSITA toolset eSASS is also available in a Docker container for macOS and Windows users
    # CURRENTLY we do not support it and actually an exception will be raised in the backend check, but we might
    #  support it in the future, in which case we may well need to do something differently here, and might need
    #  reminding of that by a handy not implemented error
    if 'esass_in_docker' in extra_info and extra_info['esass_in_docker']:
        raise NotImplementedError("The use of eSASS through Docker has not been implemented.")

    # ---------------------------------------------------------------------------------------------

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


