#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/12/2022, 14:25. Copyright (c) The Contributors

import os.path
from functools import wraps
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from typing import Tuple
from warnings import warn

from exceptiongroup import ExceptionGroup
from packaging.version import Version
from tqdm import tqdm

from daxa.archive.base import Archive
from daxa.exceptions import NoXMMMissionsError
from daxa.process._backend_check import find_sas

ALLOWED_XMM_MISSIONS = ['xmm_pointed', 'xmm_slew']


def _sas_process_setup(obs_archive: Archive) -> Version:
    """
    This function is to be called at the beginning of XMM specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :return: The version number of the SAS install located on the system, as a 'packaging' Version instance.
    :rtype: Version
    """
    # This makes sure that SAS is installed on the host system, and also identifies the version
    sas_vers = find_sas()

    if not isinstance(obs_archive, Archive):
        raise TypeError('The passed obs_archive must be an instance of the Archive class, which is made up of one '
                        'or more mission class instances.')

    # Now we ensure that the passed observation archive actually contains XMM mission(s)
    xmm_miss = [mission for mission in obs_archive if mission.name in ALLOWED_XMM_MISSIONS]
    if len(xmm_miss) == 0:
        raise NoXMMMissionsError("None of the missions that make up the passed observation archive are "
                                 "XMM missions, and thus this XMM-specific function cannot continue.")
    else:
        processed = [xm.processed for xm in xmm_miss]
        if any(processed):
            warn("One or more XMM missions have already been fully processed")

    # TODO Remove this when XMM slew is implemented and SAS procedures have been verified as working
    if any([m.name == 'xmm_slew' for m in xmm_miss]):
        raise NotImplementedError("This process has not yet been implemented/tested for slew "
                                  "XMM observations.")

    for miss in xmm_miss:
        # We make sure that the archive directory has folders to store the processed XMM data that will eventually
        #  be created by most functions that call this _sas_process_setup function
        for obs_id in miss.filtered_obs_ids:
            stor_dir = obs_archive.get_processed_data_path(miss, obs_id)
            if not os.path.exists(stor_dir):
                os.makedirs(stor_dir)

    return sas_vers


def execute_cmd(cmd: str, rel_id: str, miss_name: str, check_path: str) -> Tuple[str, str, bool, str, str]:
    """
    This is a simple function designed to execute cmd line SAS commands for the processing and reduction of
    XMM mission data. It will collect the stdout and stderr values for each command and return them too for the
    process of logging. Finally, it checks that a specified 'final file' actually exists at the expected path, as
    a final check of the success of whatever process has been run.

    :param str cmd: The command that should be executed in a bash shell.
    :param str rel_id: Whatever ID has been attached to the particular command (it could be an ObsID, or an ObsID
        + instrument combination depending on the task).
    :param str miss_name: The specific XMM mission name that this task belongs to.
    :param str check_path: The path where a 'final file' should exist, used for the purposes of checking
        that it exists.
    :return: The rel_id, a boolean flag indicating whether the final file exists, the std_out, and the std_err.
    :rtype: Tuple[str, str, bool, str, str]
    """

    # Starts the process running on a shell, connects to the process and waits for it to terminate, and collects
    #  the stdout and stderr
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    # Decodes the stdout and stderr from the binary encoding it currently exists in. The errors='ignore' flag
    #  means that it doesn't throw errors if there is a character it doesn't recognize
    out = out.decode("UTF-8", errors='ignore')
    err = err.decode("UTF-8", errors='ignore')

    # Simple check on whether the 'final file' passed into this function actually exists or not
    if os.path.exists(check_path):
        file_exists = True
    else:
        file_exists = False

    return rel_id, miss_name, file_exists, out, err


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings.
    """

    @wraps(sas_func)
    def wrapper(*args, **kwargs):

        # The first argument of all the SAS processing functions will be an archive instance, and pulling
        #  that out of the arguments will be useful later
        obs_archive = args[0]

        # This is the output from whatever function this is a decorator for
        miss_cmds, miss_final_paths, miss_extras, process_message, cores, disable = sas_func(*args, **kwargs)

        # This just sets up a dictionary of how many tasks there are for each mission
        num_to_run = {mn: len(miss_cmds[mn]) for mn in miss_cmds}

        # The first dictionary is to store boolean flags for each task, True if they succeeded (i.e. no errors +
        #  the final file exists), False if they didn't. The second dictionary is to store raised errors. The
        #  top level keys are mission names, the lower level keys are whatever was used for the task being run (i.e.
        #  either ObsID or ObsID+Inst, depending on the task).
        success_flags = {}
        process_stderrs = {}
        # The std outs recorded for each task, keys are the same as the two dictionaries above
        process_stdouts = {}
        # I do not love this solution, but this will be what any python errors that are thrown during execute_cmd
        #  are stored in. In theory, because execute_cmd is so simple, there shouldn't be Python errors thrown.
        #  SAS errors will be stored in process_stderrs
        python_errors = []

        # Iterating through the missions (there may only one but as the dictionary will have mission name as the top
        #  level key regardless this is valid for one or multiple XMM missions).
        for miss_name in miss_cmds:
            # Set up top level (mission name) keys for the output storage dictionaries
            success_flags[miss_name] = {}
            process_stderrs[miss_name] = {}
            process_stdouts[miss_name] = {}

            # There's no point setting up a Pool etc. if there are no tasks to run for the current mission, so
            #  we check how many there are
            if num_to_run[miss_name] > 0:
                # Use the mission name to grab the relevant mission object out from the observation archive
                rel_miss = obs_archive[miss_name]

                # Set up a tqdm progress bar, as well as a Pool for multiprocessing (using the number of cores
                #  specified in the SAS task that this decorator wraps. We want to parallelize these tasks because
                #  they tend to be embarrassingly parallelise
                with tqdm(total=num_to_run[miss_name], desc=rel_miss.pretty_name + ' - ' + process_message,
                          disable=disable) as gen, Pool(cores) as pool:

                    # This 'callback' function is triggered when the parallelized function completes successfully
                    #  and returns.
                    def callback(results_in: Tuple[str, str, bool, str, str]):
                        """
                        Callback function for the apply_async pool method, gets called when a task finishes
                        and something is returned.

                        :param Tuple[str, str, bool, str, str] results_in: The output of execute_cmd.
                        """
                        # The progress bar will need updating
                        nonlocal gen
                        # Need to make sure we have access to these dictionaries to store information on process
                        #  success, stderr, and stdout
                        nonlocal success_flags
                        nonlocal process_stderrs
                        nonlocal process_stdouts

                        # Just unpack the results in for clarity's sake
                        relevant_id, mission_name, does_file_exist, proc_out, proc_err = results_in

                        # We consider the task successful if the final file exists and there is nothing
                        #  in the stderr output
                        if does_file_exist and len(proc_err) == 0:
                            success_flags[mission_name][relevant_id] = True
                        else:
                            success_flags[mission_name][relevant_id] = False
                            process_stderrs[mission_name][relevant_id] = proc_err

                        # Store the stdout for logging purposes
                        process_stdouts[mission_name][relevant_id] = proc_out

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
                        gen.update(1)

                    for rel_id, cmd in miss_cmds[miss_name].items():
                        rel_fin_path = miss_final_paths[miss_name][rel_id]

                        pool.apply_async(execute_cmd, args=(cmd, rel_id, miss_name, rel_fin_path),
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            # This uses the new ExceptionGroup class to raise a set of python errors (if there are any raised
            #  during the execute_cmd function calls)
            if len(python_errors) != 0:
                raise ExceptionGroup("Python errors raised during SAS commands", python_errors)

        return success_flags, process_stderrs, process_stdouts
    return wrapper


