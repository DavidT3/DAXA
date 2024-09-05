#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 04/09/2024, 14:55. Copyright (c) The Contributors

import glob
import os.path
from enum import Flag
from functools import wraps
from inspect import signature, Parameter
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE, TimeoutExpired
from typing import Tuple, List
from warnings import warn

from astropy.units import UnitConversionError
from exceptiongroup import ExceptionGroup
from tqdm import tqdm

from daxa.archive.base import Archive
from daxa.exceptions import NoEROSITAMissionsError, DAXADeveloperError
from daxa.process._backend_check import find_esass
from daxa.process.general import create_dirs

ALLOWED_EROSITA_MISSIONS = ['erosita_calpv', 'erosita_all_sky_de_dr1']


# TODO Make this compliant with how I normally do docstrings
class _eSASS_Flag(Flag):
    """
    This class was written by Toby Wallage found on Github @TobyWallage.
    It throws a ValueError when an invalid eSASS Flag is declared with this class.
    For use in the cleaned_evt_lists function to check the user input.

    Class derived from Enum/Flag containing possible flags that can be used in
    evtool.
    
    Descriptions found here:
    https://erosita.mpe.mpg.de/edr/DataAnalysis/prod_descript/EventFiles_edr.html

    Args
        :param int value: An integer repesenting a valid flag value. The flag
        value will determine which type of events will be discarded by the
        cleaned_evt_lists function.
    
    Raises
        :raises ValueError: If value is NOT a valid flag.
        :raises TypeError: If value is not an integer or integer-like.

    Examples

        A flag may be constructed with a valid integer hexadecimal value
        >>> some_flag = _eSASS_Flag(0xc0008000)

        A flag may also be constructed from individual events
        >>> some_new_flag = _eSASS_Flag.DEFAULT_FLAG | _eSASS_Flag.OUT_OF_FOV

        These flags are identical
    
    """
    # Values copied and pasted from eSASS docs
    MPE_OWNER = 0x1
    IKI_OWNER = 0x2
    TRAILING_EVENT = 0x10
    NEXT_TO_BORDER = 0x20
    NEXT_TO_ONBOARD_BADPIX = 0x100
    NEXT_TO_BRIGHT_PIX = 0x200
    NEXT_TO_DEAD_PIX = 0x400
    NEXT_TO_FLICKERING = 0x800
    ON_FLICKERING = 0x1000
    ON_BADPIX = 0x2000
    ON_DEADPIX = 0x4000
    OUT_OF_FOV = 0x8000
    OUTSIDE_QUALGTI = 0x10000
    OUTSIDE_GTI = 0x20000
    PRECEDING_MIP = 0x40000
    MIP_ASSOCIATED = 0x80000
    PHA_QUALITY_1 = 0x1000000
    PHA_QUALITY_2 = 0x2000000
    PHA_QUALITY_3 = 0x4000000
    CORRUPT_EVENT = 0x40000000
    CORRUPT_FRAME = 0x80000000

    # DEFAULT_FLAG is equivalent to 0xc0000000
    DEFAULT_FLAG = CORRUPT_EVENT | CORRUPT_FRAME

    def get_hex(self):
        return hex(self.value)


def _is_valid_flag(flag):
    """
    This function is to be called within the cleaned_evt_lists function to check that the user has
    input a valid eSASS flag to filter event with. 

    :param flag Flag: The user input of the flag parameter in the cleaned_evt_list function.
        This may be in hexidecimal or its equivalent decimal format, both are accepted by evtool. 
    :return: True for valid eSASS flags, and False for invalid. 
    """
    try:
        # If the flag is valid then it will declare the class without an error
        _eSASS_Flag(flag)
        return True

    except ValueError:
        # If the flag is invalid then a ValueError is thrown
        return False


def _make_flagsel_keword(flag, invert=True):
    """
    This function is to be called within the cleaned_evt_lists function to generate the correct
    header keyword based on the user's input eSASS flag. This is a workaround a bug within eSASS.

    :param flag Flag: The user input of the flag parameter in the cleaned_evt_list function.
        This may be in hexidecimal or its equivalent decimal format, both are accepted by evtool.
    """

    #TODO I think that the pattern selection might effect the FLAGSEL keyword - need to check

    if invert:
        value = _eSASS_Flag(flag).value
    
    else:
        # This returns a flag containing all the bits apart from those specified by the user
        value = ~_eSASS_Flag(flag).value

    return value


def _esass_process_setup(obs_archive: Archive) -> bool:
    """
    This function is to be called at the beginning of eROSITA specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :return: A bool indicating whether eSASS is being used via Docker or not, set to True if Docker is being used.
    :rtype: Bool
    """

    # This makes sure that eSASS is installed on the host system, and also idenitifies whether
    # it is within a Docker container or just on the system.
    esass_in_docker = find_esass()

    if not isinstance(obs_archive, Archive):
        raise TypeError('The passed obs_archive must be an instance of the Archive class, which is made up of one '
                        'or more mission class instances.')
    
    # Now we ensure that the passed observation archive actually contains eROSITA mission(s)
    erosita_miss = [mission for mission in obs_archive if mission.name in ALLOWED_EROSITA_MISSIONS]
    if len(erosita_miss) == 0:
        raise NoEROSITAMissionsError("None of the missions that make up the passed observation archive are "
                                     "eROSITA missions, and thus this eROSITA-specific function cannot continue.")
    else:
        processed = [em.processed for em in erosita_miss]
        if any(processed):
            warn("One or more eROSITA missions have already been fully processed")

    # This section generates the storage directory structure for each eROSITA mission
    for miss in erosita_miss:
        create_dirs(obs_archive, miss.name)

    return esass_in_docker


def execute_cmd(cmd: str, esass_in_docker: bool, rel_id: str, miss_name: str, check_path: str,
                extra_info: dict, timeout: float = None) -> Tuple[str, str, List[bool], str, str, dict]:
    """
    This is a simple function designed to execute eSASS commands either through Docker or the command line
    for the processing and reduction of eROSITA mission data. It will collect the stdout and stderr values 
    for each command and return them too for the process of logging. Finally, it checks that a specified 'final file' 
    (or a set of 'final files') actually exists at the expected path, as a final check of the success of whatever 
    process has been run.

    :param str cmd: The command that should be executed in a bash shell.
    :param Bool esass_in_docker: Set to True if eSASS is being used via Docker.
    :param str rel_id: Whatever ID has been attached to the particular command (it could be an ObsID, or an ObsID
        + instrument combination depending on the task).
    :param str miss_name: The specific eROSITA mission name that this task belongs to.
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

    # eSASS is also released in a Docker container for Mac OS and Windows users, which is not yet supported in DAXA.
    if esass_in_docker:
        raise NotImplementedError("The use of eSASS through Docker has not been implemented.")

    # Starts the process running on a shell
    cmd_proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # This makes sure the process is killed if it does timeout
    try:
        out, err = cmd_proc.communicate(timeout=timeout)
    except TimeoutExpired:
        cmd_proc.kill()
        out, err = cmd_proc.communicate()
        warn("An eROSITA process for {} has timed out".format(rel_id), stacklevel=2)
    
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


def esass_call(esass_func):
    """
    This is used as a decorator for functions that produce eSASS command strings.
    """

    @wraps(esass_func)
    def wrapper(*args, **kwargs):
        # This is here to avoid a circular import issue
        from daxa.process.erosita.setup import _prepare_erosita_info

        # The first argument of all the eSASS processing functions will be an archive instance, and pulling
        #  that out of the arguments will be useful later
        obs_archive = args[0]
        obs_archive: Archive  # Just for autocomplete purposes in my IDE

        func_sig = signature(esass_func)
        all_arg_names = [key for key in func_sig.parameters.keys()]
        run_args = {k: v.default for k, v in func_sig.parameters.items() if v.default is not Parameter.empty}
        run_args = {k: kwargs[k] if k in kwargs else v for k, v in run_args.items()}
        if len(args) != 1:
            for ind in range(1, len(args)):
                rel_key = all_arg_names[ind]
                run_args[rel_key] = args[ind]

        # Seeing if any of the erosita missions in the archive have had any processing done yet
        erosita_miss = [mission for mission in obs_archive if mission.name in ALLOWED_EROSITA_MISSIONS]
        for miss in erosita_miss:
            # Getting the process_logs for each mission
            process_logs = obs_archive._process_logs[miss.name]

            # We need to run the _prepare_erositacalpv_info function.
            #   This will fill out the mission observation summaries, which are needed for later
            #   processing functions. It will also populate the _process_extra_info dictionary for the archive
            #   with top level keys of the erositacalpv mission and lower level keys of obs_ids with
            #   lower level keys of 'path', which will store the raw data path for that obs id.
            _prepare_erosita_info(obs_archive, miss)

        # This is the output from whatever function this is a decorator for
        (miss_cmds, miss_final_paths, miss_extras, process_message, cores, disable, timeout,
         esass_in_docker) = esass_func(*args, **kwargs)

        # Converting the timeout from whatever time units it is in, to seconds - but first checking that the user
        #  hasn't been daft and passed a non-time quantity
        if timeout is not None and not timeout.unit.is_equivalent('s'):
            raise UnitConversionError("The value of timeout must be convertible to seconds.")
        elif timeout is not None:
            timeout = timeout.to('s').value

        # This just acts as a check for any newly implemented functions as a reminder that they need to be in that
        #  dictionary, otherwise loading an archive, updating it, and processing the new data will not work
        from daxa.process import PROC_LOOKUP
        for mn in miss_cmds:
            if esass_func.__name__ not in PROC_LOOKUP[mn]:
                raise DAXADeveloperError("The {p} process does not have an entry in process.PROC_FILTER for "
                                         "{mn}.".format(p=esass_func.__name__, mn=mn))

        # This just sets up a dictionary of how many tasks there are for each mission
        num_to_run = {mn: len(miss_cmds[mn]) for mn in miss_cmds}

        # The first dictionary is to store boolean flags for each task, True if they succeeded (i.e. no errors +
        #  the final file exists), False if they didn't. The second dictionary is to store raised errors. The
        #  top level keys are mission names, the lower level keys are whatever was used for the task being run (i.e.
        #  either ObsID or ObsID+Inst, depending on the task).
        success_flags = {}
        process_raw_stderrs = {}  # Specifically the unparsed stderr
        # The std outs recorded for each task, keys are the same as the two dictionaries above
        process_stdouts = {}
        # This is for the extra information which can be passed from processing functions
        process_einfo = {}
        # Here we setup another dictionary to store the processing configuration in - all this will be though
        #  is one layer deeper than the existing run_args dictionary, with mission names as keys on the top level
        process_cinfo = {}

        # Observation information, parsed from the header and event lists from the raw data file, will be stored in
        #  this dictionary and eventually passed into the archive. As such this dictionary will only be used
        #  if the task esass_call is wrapping is flaregti
        parsed_obs_info = {}

        # I do not love this solution, but this will be what any python errors that are thrown during execute_cmd
        #  are stored in. In theory, because execute_cmd is so simple, there shouldn't be Python errors thrown.
        python_errors = []

        # Iterating through the missions (there may only one but as the dictionary will have mission name as the top
        #  level key regardless this is valid for one or multiple eROSITA missions).
        for miss_name in miss_cmds:
            # Set up top level (mission name) keys for the output storage dictionaries
            success_flags[miss_name] = {}
            process_raw_stderrs[miss_name] = {}
            process_stdouts[miss_name] = {}
            process_einfo[miss_name] = {}
            process_cinfo[miss_name] = {}
            parsed_obs_info[miss_name] = {}

            # There's no point setting up a Pool etc. if there are no tasks to run for the current mission, so
            #  we check how many there are
            if num_to_run[miss_name] > 0:
                # Use the mission name to grab the relevant mission object out from the observation archive
                rel_miss = obs_archive[miss_name]

                # Set up a tqdm progress bar, as well as a Pool for multiprocessing (using the number of cores
                #  specified in the eSASS task that this decorator wraps. We want to parallelize these tasks because
                #  they tend to be embarrassingly parallelise
                with tqdm(total=num_to_run[miss_name], desc=rel_miss.pretty_name + ' - ' + process_message,
                          disable=disable) as gen, Pool(cores) as pool:

                    # This 'callback' function is triggered when the parallelized function completes successfully
                    #  and returns.
                    def callback(results_in: Tuple[str, str, List[bool], str, str, dict]):
                        """
                        Callback function for the apply_async pool method, gets called when a task finishes
                        and something is returned.

                        :param Tuple[str, str, List[bool], str, str, dict] results_in: The output of execute_cmd.
                        """
                        # The progress bar will need updating
                        nonlocal gen
                        # Need to make sure we have access to these dictionaries to store information on process
                        #  success, stderr, and stdout
                        nonlocal success_flags
                        nonlocal process_raw_stderrs
                        nonlocal process_stdouts
                        nonlocal process_einfo

                        nonlocal parsed_obs_info
                        nonlocal python_errors

                        # Just unpack the results in for clarity's sake
                        relevant_id, mission_name, does_file_exist, proc_out, proc_err, proc_extra_info = results_in
                        
                        # TODO would like to parse and identify eSASS errors like we can with SAS

                        # We consider the task successful if all the final files exist and there are no entries in
                        #  the parsed std_err output
                        if all(does_file_exist):
                            success_flags[mission_name][relevant_id] = True
                        else:
                            success_flags[mission_name][relevant_id] = False
                            process_raw_stderrs[mission_name][relevant_id] = proc_err

                        # Store the stdout for logging purposes
                        process_stdouts[mission_name][relevant_id] = proc_out

                        # If there is extra information then we shall store it in the dictionary which will
                        #  eventually be fed to the archive
                        if len(proc_extra_info) != 0:
                            process_einfo[mission_name][relevant_id] = proc_extra_info

                        # Make sure to update the progress bar
                        gen.update(1)

                    # This other 'callback' function is triggered when Python inside the parallelised function
                    #  raises an exception rather than completing successfully
                    def err_callback(err):
                        """
                        The callback function for errors that occur inside a task running in the pool.
                        :param err: An error that occurred inside a task.
                        """
                        nonlocal python_errors
                        nonlocal gen

                        if err is not None:
                            # Rather than throwing an error straight away I append them all to a list for later.
                            python_errors.append(err)
                        # Still want to update the progress bar if an error has occurred
                        gen.update(1)

                    for rel_id, cmd in miss_cmds[miss_name].items():
                        # Grab the relevant information for the current mission and ID
                        rel_fin_path = miss_final_paths[miss_name][rel_id]
                        rel_einfo = miss_extras[miss_name][rel_id]

                        pool.apply_async(execute_cmd, args=(cmd, esass_in_docker, rel_id, miss_name, rel_fin_path,
                                                            rel_einfo, timeout),
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            # This uses the new ExceptionGroup class to raise a set of python errors (if there are any raised
            #  during the execute_cmd function calls)
            if len(python_errors) != 0:
                raise ExceptionGroup("Python errors raised during eSASS commands", python_errors)

        obs_archive.process_success = (esass_func.__name__, success_flags)
        obs_archive.raw_process_errors = (esass_func.__name__, process_raw_stderrs)
        obs_archive.process_logs = (esass_func.__name__, process_stdouts)
        obs_archive.process_extra_info = (esass_func.__name__, process_einfo)
        obs_archive.process_configurations = (esass_func.__name__, process_cinfo)

        # We automatically save after every process run
        obs_archive.save()

    return wrapper
