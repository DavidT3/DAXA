#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 08/11/2024, 16:36. Copyright (c) The Contributors

from functools import wraps
from inspect import signature, Parameter
from multiprocessing.dummy import Pool
from typing import Tuple, List
from warnings import warn

from astropy.units import UnitConversionError
from exceptiongroup import ExceptionGroup
from packaging.version import Version
from tqdm import tqdm

from daxa import BaseMission
from daxa.archive.base import Archive
from daxa.exceptions import DAXADeveloperError
from daxa.exceptions import NoValidMissionsError
from daxa.process._backend_check import find_ciao
from daxa.process._common import create_dirs
from daxa.process._common import execute_cmd

ALLOWED_CHANDRA_MISSIONS = ['chandra']


def _ciao_process_setup(obs_archive: Archive, make_dirs: bool = True) -> Tuple[Version, Version, List[BaseMission]]:
    """
    This function is to be called at the beginning of CIAO specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :param bool make_dirs: A boolean variable that controls whether the setup process should ensure that the
        storage directories for the future processed Chandra data are made or not. Default is True.
    :return: The version numbers of the CIAO and CALDB installs located on the system, as well as the relevant missions.
    :rtype: Tuple[Version, Version, List[BaseMission]]
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

    # One function (prepare_chandra_info) wants most of this preparation function, without necessarily creating
    #  directories, so we now allow the option of not making directories
    if make_dirs:
        # This bit creates the storage directories for Chandra missions
        for miss in chandra_miss:
            create_dirs(obs_archive, miss.name)

    return ciao_vers, caldb_vers, chandra_miss


def ciao_call(ciao_func):
    """
    This is used as a decorator for functions that produce CIAO command strings.
    """

    @wraps(ciao_func)
    def wrapper(*args, **kwargs):

        # The first argument of all the CIAO processing functions will be an archive instance, and pulling
        #  that out of the arguments will be useful later
        obs_archive = args[0]
        obs_archive: Archive  # Just for autocomplete purposes in my IDE

        func_sig = signature(ciao_func)
        all_arg_names = [key for key in func_sig.parameters.keys()]
        run_args = {k: v.default for k, v in func_sig.parameters.items() if v.default is not Parameter.empty}
        run_args = {k: kwargs[k] if k in kwargs else v for k, v in run_args.items()}
        if len(args) != 1:
            for ind in range(1, len(args)):
                rel_key = all_arg_names[ind]
                run_args[rel_key] = args[ind]

        # This is the output from whatever function this is a decorator for
        (miss_cmds, miss_final_paths, miss_extras, process_message, cores, disable,
            timeout) = ciao_func(*args, **kwargs)

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
            if ciao_func.__name__ not in PROC_LOOKUP[mn]:
                raise DAXADeveloperError("The {p} process does not have an entry in process.PROC_LOOKUP for "
                                         "{mn}.".format(p=ciao_func.__name__, mn=mn))

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

        # Iterating through the missions (there will probably only ever be one for Chandra but it'll still iterate as
        #  this goes through the dictionary keys, which are mission names).
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
                #  specified in the CIAO task that this decorator wraps. We want to parallelize these tasks because
                #  they tend to be embarrassingly parallel
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

                        # TODO would like to parse and identify CIAO errors like we can with SAS

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

                        pool.apply_async(execute_cmd, args=(cmd, rel_id, miss_name, rel_fin_path, rel_einfo, timeout),
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            # This uses the new ExceptionGroup class to raise a set of python errors (if there are any raised
            #  during the execute_cmd function calls)
            if len(python_errors) != 0:
                raise ExceptionGroup("Python errors raised during CIAO commands", python_errors)

        obs_archive.process_success = (ciao_func.__name__, success_flags)
        obs_archive.raw_process_errors = (ciao_func.__name__, process_raw_stderrs)
        obs_archive.process_logs = (ciao_func.__name__, process_stdouts)
        obs_archive.process_extra_info = (ciao_func.__name__, process_einfo)
        obs_archive.process_configurations = (ciao_func.__name__, process_cinfo)

        # We automatically save after every process run
        obs_archive.save()

    return wrapper
