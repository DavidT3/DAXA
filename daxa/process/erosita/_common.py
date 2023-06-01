# This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
# Last modified by David J Turner (turne540@msu.edu) Thu Apr 13 2023, 15:16. Copyright (c) The Contributors
import glob
from typing import Tuple, List
from warnings import warn
import os.path
from subprocess import Popen, PIPE
from functools import wraps
from multiprocessing.dummy import Pool

from astropy.units import UnitConversionError
from tqdm import tqdm
from exceptiongroup import ExceptionGroup

from daxa.archive.base import Archive
from daxa.exceptions import NoEROSITAMissionsError
from daxa.process._backend_check import find_esass
from daxa.process.erosita.setup import prepare_erositacalpv_info

ALLOWED_EROSITA_MISSIONS = ['erosita_calpv']

def _esass_process_setup(obs_archive: Archive) -> bool:
    """
    This function is to be called at the beginning of eROSITA specific processing functions, and contains several
    checks to ensure that passed data common to multiple process function calls is suitable.

    :param Archive obs_archive: The observation archive passed to the processing function that called this function.
    :return: A bool indicating whether or not eSASS is being used via Docker or not, set to True if Docker is being used. 
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

    for miss in erosita_miss:
        # We make sure that the archive directory has folders to store the processed eROSITA data that will eventually
        #  be created by most functions that call this _esass_process_setup function
        for obs_id in miss.filtered_obs_ids:
            stor_dir = obs_archive.get_processed_data_path(miss, obs_id)
            if not os.path.exists(stor_dir):
                os.makedirs(stor_dir)
    
        # We also ensure that an overall directory for failed processing observations exists - this will give
        #  observation directories which have no useful data in (i.e. they do not have a successful final
        #  processing step) somewhere to be copied to (see daxa.process._cleanup._last_process).
        # This is the overall path, there might not ever be anything in it, so we don't pre-make ObsID sub-directories
        fail_proc_dir = obs_archive.get_failed_data_path(miss, None).format(oi='')[:-1]
        if not os.path.exists(fail_proc_dir):
            os.makedirs(fail_proc_dir)

    return esass_in_docker

def execute_cmd(cmd: str, esass_in_docker: bool, rel_id: str, miss_name: str, check_path: str,
                extra_info: dict) -> Tuple[str, str, List[bool], str, str, dict]:
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
    :return: The rel_id, a list of boolean flags indicating whether the final files exist, the std_out, and the
        std_err. The final dictionary can contain extra information recorded by the processing function.
    :rtype: Tuple[str, str, List[bool], str, str, dict]
    """
    # Either a single path or a list of paths can be passed to check - I make sure that the checking process only
    #  ever has to deal with a list
    if isinstance(check_path, str):
        check_path = [check_path]

    # eSASS is also released in a Docker container for Mac OS and Windows users, which is not yet supported in DAXA.
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
    print('excecute_cmd done')
    return rel_id, miss_name, files_exist, out, err, extra_info

def esass_call(esass_func):
    """
    This is used as a decorator for functions that produce eSASS command strings.
    """

    @wraps(esass_func)
    def wrapper(*args, **kwargs):
        # This is here to avoid a circular import issue
        from daxa.process.erosita.setup import prepare_erositacalpv_info

        # The first argument of all the eSASS processing functions will be an archive instance, and pulling
        #  that out of the arguments will be useful later
        # DAVID_QUESTION Arent there a mixture of mission types in here?
        obs_archive = args[0]
        obs_archive: Archive  # Just for autocomplete purposes in my IDE

        # Seeing if any of the erosita missions in the archive have had any processing done yet
        # DAVID_QUESTION if this is in esass call, is it guarenteed there is an erosita mission
        erosita_miss = [mission for mission in obs_archive if mission.name in ALLOWED_EROSITA_MISSIONS]
        print('WRAPPER gone into the wrapper')
        for miss in erosita_miss:
            # Getting the process_logs for each mission
            process_logs = obs_archive._process_logs[miss.name]
            if len(process_logs) == 0:
                # If no processing has been done yet, we need to run the prepare_erositacalpv_info function.
                #   This will fill out the mission observation summaries, which are needed for later 
                #   processing functions. It will also populate the _process_extra_info dictionary for the archive
                #   with top level keys of the erositacalpv mission and lower level keys of obs_ids with lower level keys
                #   of 'path', which will store the raw data path for that obs id.
                prepare_erositacalpv_info(obs_archive, miss)
                print('WRAPPER done the prepare_erositacalpv_info function in the wrapper')

        # This is the output from whatever function this is a decorator for
        miss_cmds, miss_final_paths, miss_extras, process_message, cores, disable, timeout, esass_in_docker = esass_func(*args, **kwargs)

        # Converting the timeout from whatever time units it is in, to seconds - but first checking that the user
        #  hasn't been daft and passed a non-time quantity
        if timeout is not None and not timeout.unit.is_equivalent('s'):
            raise UnitConversionError("The value of timeout must be convertible to seconds.")
        elif timeout is not None:
            timeout = timeout.to('s').value

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

        # Observation information, parsed from the header and event lists from the raw data file, will be stored in
        #  this dictionary and eventually passed into the archive. As such this dictionary will only be used
        #  if the task esass_call is wrapping is flaregti
        parsed_obs_info = {}

        '''
        # I do not love this solution, but this will be what any python errors that are thrown during execute_cmd
        #  are stored in. In theory, because execute_cmd is so simple, there shouldn't be Python errors thrown.
        #  SAS errors will be stored in process_parsed_stderrs
        python_errors = []

        # Iterating through the missions (there may only one but as the dictionary will have mission name as the top
        #  level key regardless this is valid for one or multiple eROSITA missions).
        for miss_name in miss_cmds:
            # Set up top level (mission name) keys for the output storage dictionaries
            success_flags[miss_name] = {}
            process_raw_stderrs[miss_name] = {}
            process_stdouts[miss_name] = {}
            process_einfo[miss_name] = {}
            parsed_obs_info[miss_name] = {}

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
                        # This processes the stderr output to try and differentiate between warnings and actual
                        #  show-stopping errors
                        sas_err, sas_warn, other_err = parse_stderr(proc_err)

                        # We consider the task successful if all the final files exist and there are no entries in
                        #  the parsed std_err output
                        if all(does_file_exist) and len(sas_err) == 0:
                            success_flags[mission_name][relevant_id] = True
                        else:
                            success_flags[mission_name][relevant_id] = False
                            if not does_file_exist:
                                sas_err.append('Final file not found raised by DAXA')
                            # We store both the parsed and unparsed stderr for debugging purposes
                            process_raw_stderrs[mission_name][relevant_id] = proc_err
                            process_parsed_stderrs[mission_name][relevant_id] = sas_err

                        # If there are any warnings, we don't consider them an indication of the total failure of
                        #  the process, but we do make sure to store them
                        if len(sas_warn) > 0:
                            process_parsed_stderr_warns[mission_name][relevant_id] = sas_warn

                        # Store the stdout for logging purposes
                        process_stdouts[mission_name][relevant_id] = proc_out

                        # If there is extra information then we shall store it in the dictionary which will
                        #  eventually be fed to the archive
                        if len(proc_extra_info) != 0:
                            process_einfo[mission_name][relevant_id] = proc_extra_info

                        # If the tested-for output file exists, and we know that the current task is odf_ingest
                        #  we're going to do an extra post-processing step and parse the output SAS summary file
                        if all(does_file_exist) and run_odf_sum_parse:
                            try:
                                parsed_obs_info[mission_name][relevant_id] = parse_odf_sum(proc_extra_info['sum_path'],
                                                                                           relevant_id)
                            # Possible that this parsing doesn't go our way however, so we have to be able to catch
                            #  an exception.
                            except ValueError as err:
                                python_errors.append(err)

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
                raise ExceptionGroup("Python errors raised during SAS commands", python_errors)
        '''

        # I do not love this solution, but this will be what any python errors that are thrown during execute_cmd
        #  are stored in. In theory, because execute_cmd is so simple, there shouldn't be Python errors thrown.
        #  SAS errors will be stored in process_parsed_stderrs
        python_errors = []

        # Iterating through the missions (there may only one but as the dictionary will have mission name as the top
        #  level key regardless this is valid for one or multiple eROSITA missions).
        for miss_name in miss_cmds:
            # Set up top level (mission name) keys for the output storage dictionaries
            success_flags[miss_name] = {}
            process_raw_stderrs[miss_name] = {}
            process_stdouts[miss_name] = {}
            process_einfo[miss_name] = {}
            parsed_obs_info[miss_name] = {}

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
                        # This processes the stderr output to try and differentiate between warnings and actual
                        #  show-stopping errors
                        #sas_err, sas_warn, other_err = parse_stderr(proc_err)

                        # We consider the task successful if all the final files exist and there are no entries in
                        #  the parsed std_err output
                        if all(does_file_exist) == 0:
                            success_flags[mission_name][relevant_id] = True
                        else:
                            success_flags[mission_name][relevant_id] = False
                            #if not does_file_exist:
                                #sas_err.append('Final file not found raised by DAXA')
                            # We store both the parsed and unparsed stderr for debugging purposes
                            process_raw_stderrs[mission_name][relevant_id] = proc_err
                            #process_parsed_stderrs[mission_name][relevant_id] = sas_err

                        # If there are any warnings, we don't consider them an indication of the total failure of
                        #  the process, but we do make sure to store them
                        #if len(sas_warn) > 0:
                        #    process_parsed_stderr_warns[mission_name][relevant_id] = sas_warn

                        # Store the stdout for logging purposes
                        process_stdouts[mission_name][relevant_id] = proc_out

                        # If there is extra information then we shall store it in the dictionary which will
                        #  eventually be fed to the archive
                        if len(proc_extra_info) != 0:
                            process_einfo[mission_name][relevant_id] = proc_extra_info

                        # If the tested-for output file exists, and we know that the current task is odf_ingest
                        #  we're going to do an extra post-processing step and parse the output SAS summary file
                        #if all(does_file_exist):
                          #  try:
                          #      parsed_obs_info[mission_name][relevant_id] = parse_odf_sum(proc_extra_info['sum_path'],
                          #                                                                 relevant_id)
                            # Possible that this parsing doesn't go our way however, so we have to be able to catch
                            #  an exception.
                           # except ValueError as err:
                          #      python_errors.append(err)

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

                        pool.apply_async(execute_cmd, args=(cmd, esass_in_docker, rel_id, miss_name, rel_fin_path, rel_einfo, timeout),
                                         error_callback=err_callback, callback=callback)
                        print('WRAPPER Command excecuted')
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            # This uses the new ExceptionGroup class to raise a set of python errors (if there are any raised
            #  during the execute_cmd function calls)
            if len(python_errors) != 0:
                raise ExceptionGroup("Python errors raised during SAS commands", python_errors)

        obs_archive.process_success = (esass_func.__name__, success_flags)
        #obs_archive.process_errors = (esass_func.__name__, process_parsed_stderrs)
        #obs_archive.process_warnings = (esass_func.__name__, process_parsed_stderr_warns)
        obs_archive.raw_process_errors = (esass_func.__name__, process_raw_stderrs)
        obs_archive.process_logs = (esass_func.__name__, process_stdouts)
        obs_archive.process_extra_info = (esass_func.__name__, process_einfo)
        print('WRAPPER added the process logs to the archive')

    return wrapper