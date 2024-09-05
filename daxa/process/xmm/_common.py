#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/09/2024, 12:15. Copyright (c) The Contributors

import glob
import os.path
from functools import wraps
from inspect import signature, Parameter
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE, TimeoutExpired
from typing import Tuple, List, Dict
from warnings import warn

from astropy.units import UnitConversionError
from exceptiongroup import ExceptionGroup
from packaging.version import Version
from tqdm import tqdm

from daxa.archive.base import Archive
from daxa.config import SASERROR_LIST, SASWARNING_LIST
from daxa.exceptions import NoXMMMissionsError, DAXADeveloperError
from daxa.process._backend_check import find_sas
from daxa.process.general import create_dirs

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
            warn("One or more XMM missions have already been fully processed", stacklevel=2)

    # TODO Remove this when XMM slew is implemented and SAS procedures have been verified as working
    if any([m.name == 'xmm_slew' for m in xmm_miss]):
        raise NotImplementedError("This process has not yet been implemented/tested for slew "
                                  "XMM observations.")

    # This bit creates the storage directories for XMM missions
    for miss in xmm_miss:
        create_dirs(obs_archive, miss.name)

    return sas_vers


def parse_stderr(unprocessed_stderr: str) -> Tuple[List[str], List[Dict], List]:
    """
    A function to parse the stderr output from SAS tasks which attempts to identify salient parts of the output
    by matching to known SAS errors and warnings. The identified errors/warnings are returned, and will inform
    DAXA whether a particular call of a particular SAS process was successful or not.

    :return: A list of dictionaries containing parsed, confirmed SAS errors, another containing SAS warnings,
        and another list of unidentifiable errors that occured in the stderr.
    :rtype: Tuple[List[Dict], List[Dict], List]
    """
    def find_sas_error(split_stderr: list, err_type: str) -> Tuple[List[dict], List[str]]:
        """
        Function to search for and parse SAS errors and warnings.

        :param list split_stderr: The stderr string split on line endings.
        :param str err_type: Should this look for errors or warnings?
        :return: Returns the dictionary of parsed errors/warnings, as well as all lines
            with SAS errors/warnings in.
        :rtype: Tuple[List[dict], List[str]]
        """
        parsed_sas = []
        # This is a crude way of looking for SAS error/warning strings ONLY
        sas_lines = [line for line in split_stderr if "** " in line and ": {}".format(err_type) in line]
        for err in sas_lines:
            try:
                # This tries to split out the SAS task that produced the error
                originator = err.split("** ")[-1].split(":")[0]
                # And this should split out the actual error name
                err_ident = err.split(": {} (".format(err_type))[-1].split(")")[0]
                # Actual error message
                err_body = err.split("({})".format(err_ident))[-1].strip("\n").strip(", ").strip(" ")

                if err_type == "error":
                    # Checking to see if the error identity is in the list of SAS errors
                    sas_err_match = [sas_err for sas_err in SASERROR_LIST if err_ident.lower()
                                     in sas_err.lower()]
                elif err_type == "warning":
                    # Checking to see if the error identity is in the list of SAS warnings
                    sas_err_match = [sas_err for sas_err in SASWARNING_LIST if err_ident.lower()
                                     in sas_err.lower()]

                if len(sas_err_match) != 1:
                    originator = ""
                    err_ident = ""
                    err_body = ""
            except IndexError:
                originator = ""
                err_ident = ""
                err_body = ""

            parsed_sas.append({"originator": originator, "name": err_ident, "message": err_body})
        return parsed_sas, sas_lines

    # Defined as empty as they are returned by this method
    sas_errs_msgs = []
    parsed_sas_warns = []
    other_err_lines = []
    # err_str being "" is ideal, hopefully means that nothing has gone wrong
    if unprocessed_stderr != "":
        # Errors will be added to the error summary, then raised later
        # That way if people try except the error away the object will have been constructed properly
        err_lines = [e for e in unprocessed_stderr.split('\n') if e != '']
        # Fingers crossed each line is a separate error
        parsed_sas_errs, sas_err_lines = find_sas_error(err_lines, "error")
        parsed_sas_warns, sas_warn_lines = find_sas_error(err_lines, "warning")

        sas_errs_msgs = ["{e} raised by {t} - {b}".format(e=e["name"], t=e["originator"], b=e["message"])
                         for e in parsed_sas_errs]

        # These are impossible to predict the form of, so they won't be parsed
        other_err_lines = [line for line in err_lines if line not in sas_err_lines
                           and line not in sas_warn_lines and line != "" and "warn" not in line]
        # Adding some advice
        for e_ind, e in enumerate(other_err_lines):
            if 'seg' in e.lower() and 'fault' in e.lower():
                other_err_lines[e_ind] += ' - Try examining an image of the cluster with regions subtracted, ' \
                                          'and have a look at where your coordinate lies.'

    return sas_errs_msgs, parsed_sas_warns, other_err_lines


def execute_cmd(cmd: str, rel_id: str, miss_name: str, check_path: str, extra_info: dict,
                timeout: float = None) -> Tuple[str, str, List[bool], str, str, dict]:
    """
    This is a simple function designed to execute cmd line SAS commands for the processing and reduction of
    XMM mission data. It will collect the stdout and stderr values for each command and return them too for the
    process of logging. Finally, it checks that a specified 'final file' (or a set of 'final files') actually
    exists at the expected path, as a final check of the success of whatever process has been run.

    :param str cmd: The command that should be executed in a bash shell.
    :param str rel_id: Whatever ID has been attached to the particular command (it could be an ObsID, or an ObsID
        + instrument combination depending on the task).
    :param str miss_name: The specific XMM mission name that this task belongs to.
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

    # Starts the process running on a shell
    cmd_proc = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    # This makes sure the process is killed if it does timeout
    try:
        out, err = cmd_proc.communicate(timeout=timeout)
    except TimeoutExpired:
        cmd_proc.kill()
        out, err = cmd_proc.communicate()
        warn("An XMM process for {} has timed out".format(rel_id), stacklevel=2)

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


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings.
    """

    @wraps(sas_func)
    def wrapper(*args, **kwargs):
        # This is here to avoid a circular import issue
        from daxa.process.xmm.setup import parse_odf_sum

        # The first argument of all the SAS processing functions will be an archive instance, and pulling
        #  that out of the arguments will be useful later
        obs_archive = args[0]
        obs_archive: Archive  # Just for autocomplete purposes in my IDE

        func_sig = signature(sas_func)
        all_arg_names = [key for key in func_sig.parameters.keys()]
        run_args = {k: v.default for k, v in func_sig.parameters.items() if v.default is not Parameter.empty}
        run_args = {k: kwargs[k] if k in kwargs else v for k, v in run_args.items()}
        if len(args) != 1:
            for ind in range(1, len(args)):
                rel_key = all_arg_names[ind]
                run_args[rel_key] = args[ind]

        # This is the output from whatever function this is a decorator for
        miss_cmds, miss_final_paths, miss_extras, process_message, cores, disable, timeout = sas_func(*args, **kwargs)

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
            if sas_func.__name__ not in PROC_LOOKUP[mn]:
                raise DAXADeveloperError("The {p} process does not have an entry in process.PROC_FILTER for "
                                         "{mn}.".format(p=sas_func.__name__, mn=mn))

        # This just sets up a dictionary of how many tasks there are for each mission
        num_to_run = {mn: len(miss_cmds[mn]) for mn in miss_cmds}

        # The first dictionary is to store boolean flags for each task, True if they succeeded (i.e. no errors +
        #  the final file exists), False if they didn't. The second dictionary is to store raised errors. The
        #  top level keys are mission names, the lower level keys are whatever was used for the task being run (i.e.
        #  either ObsID or ObsID+Inst, depending on the task).
        success_flags = {}
        process_raw_stderrs = {}  # Specifically the unparsed stderr
        process_parsed_stderrs = {}  # These two are for errors and warnings extracted by parsing the stderr
        process_parsed_stderr_warns = {}
        # The std outs recorded for each task, keys are the same as the two dictionaries above
        process_stdouts = {}
        # This is for the extra information which can be passed from processing functions
        process_einfo = {}
        # Here we setup another dictionary to store the processing configuration in - all this will be though
        #  is one layer deeper than the existing run_args dictionary, with mission names as keys on the top level
        process_cinfo = {}

        # Observation information, parsed from the output summary file created by ODF ingest, will be stored in
        #  this dictionary and eventually passed into the archive. As such this dictionary will only be used
        #  if the task sas_call is wrapping is odf_ingest
        parsed_obs_info = {}
        # In the same vein, I define a simple boolean to tell the callback function (where parsing will take place)
        #  whether or not it needs to run the parsing function. I could do this by checking for the existence
        #  of the 'sum_path' key in the extra information dictionary, but I think this way is safer. Just in case
        # I accidentally re-use that key somewhere else
        run_odf_sum_parse = sas_func.__name__ == 'odf_ingest'

        # I do not love this solution, but this will be what any python errors that are thrown during execute_cmd
        #  are stored in. In theory, because execute_cmd is so simple, there shouldn't be Python errors thrown.
        #  SAS errors will be stored in process_parsed_stderrs
        python_errors = []

        # Iterating through the missions (there may only one but as the dictionary will have mission name as the top
        #  level key regardless this is valid for one or multiple XMM missions).
        for miss_name in miss_cmds:
            # Set up top level (mission name) keys for the output storage dictionaries
            success_flags[miss_name] = {}
            process_raw_stderrs[miss_name] = {}
            process_parsed_stderrs[miss_name] = {}
            process_parsed_stderr_warns[miss_name] = {}
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
                        nonlocal process_parsed_stderrs
                        nonlocal process_parsed_stderr_warns
                        nonlocal process_stdouts
                        nonlocal process_einfo
                        nonlocal run_odf_sum_parse
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

            # Adding an entry of the run arguments for this processing step under the current mission name
            process_cinfo[miss_name] = run_args

        obs_archive.process_success = (sas_func.__name__, success_flags)
        obs_archive.process_errors = (sas_func.__name__, process_parsed_stderrs)
        obs_archive.process_warnings = (sas_func.__name__, process_parsed_stderr_warns)
        obs_archive.raw_process_errors = (sas_func.__name__, process_raw_stderrs)
        obs_archive.process_logs = (sas_func.__name__, process_stdouts)
        obs_archive.process_extra_info = (sas_func.__name__, process_einfo)
        obs_archive.process_configurations = (sas_func.__name__, process_cinfo)

        # If the task we just ran is odf ingest, that means we've parsed the summary files to provide us with some
        #  information on the data we have - that information is in the parsed_obs_info dictionary and needs to be
        #  added to the observation_summaries property of the archive
        if run_odf_sum_parse:
            obs_archive.observation_summaries = parsed_obs_info

        # We automatically save after every process run
        obs_archive.save()

    return wrapper


