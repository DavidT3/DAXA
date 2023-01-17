#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 17/01/2023, 16:37. Copyright (c) The Contributors
import os
from random import randint
from typing import Union, List
from warnings import warn

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.xmm._common import ALLOWED_XMM_MISSIONS, _sas_process_setup, sas_call


@sas_call
def emanom(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    This function runs the SAS emanom function, which attempts to identify when MOS CCDs are have operated in an
    'anomalous' state, where the  background at E < 1 keV is strongly enhanced. Data above 2 keV are unaffected, so
    CCDs in anomalous states used for science where the soft X-rays are unnecessary do not need to be excluded.

    The emanom task calculates the (2.5-5.0 keV)/(0.4-0.8 keV) hardness ratio from the corner data to determine
    whether a chip is in an anomalous state. However, it should be noted that the "anonymous" anomalous state of
    MOS1 CCD#4 is not always detectable from the unexposed corner data.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with MOS observations for
        which emchain should be run. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool]
    """
    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # Define the form of the emchain command that must be run to check for anomalous states in MOS CCDs
    emanom_cmd = "cd {d}; export SAS_CCF={ccf}; emanom eventfile={ef} keepcorner=no; mv {of} ../; cd ..; rm -r {d}"

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information that might be useful to provide to the next step in the generation process
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # Just grabs the XMM missions, we already know there will be at least one because otherwise _sas_process_setup
    #  would have thrown an error
    xmm_miss = [mission for mission in obs_archive if mission.name in ALLOWED_XMM_MISSIONS]
    # We are iterating through XMM missions (options could include xmm_pointed and xmm_slew for instance).
    for miss in xmm_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # Need to check to see whether ANY of ObsID-instrument-subexposure combos have had emchain run for them, as
        #  it is a requirement for this processing function. There will probably be a more elegant way of checkinf
        #  at some point in the future, generalised across all SAS functions
        if 'emchain' not in obs_archive.process_success[miss.name]:
            raise NoDependencyProcessError("The emchain step has not been run for the {m} mission in the {a} "
                                           "archive, it is a requirement to use "
                                           "emanom.".format(m=miss.name, a=obs_archive.archive_name))
        # If every emchain run was a failure then we warn the user and move onto the next XMM mission (if there
        #  is one).
        elif all([v is False for v in obs_archive.process_success[miss.name]['emchain'].values()]):
            warn("Every emchain run for the {m} mission in the {a} archive is reporting as a failure, skipping "
                 "process.".format(m=miss.name, a=obs_archive.archive_name), stacklevel=2)
            continue
        else:
            # This fetches those IDs for which emchain has reported success, and these are what we will iterate
            #  through to ensure that we only act upon data that is in a final event list form.
            valid_ids = [k for k, v in obs_archive.process_success[miss.name]['emchain'].items() if v]

        # We iterate through the valid IDs rather than nest ObsID and instrument for loops - as we can use the emchain
        #  success information to determine which can be processed further.
        for val_id in valid_ids:
            # TODO Review this if I change the IDing system as I was pondering in issue #44
            if 'M1' in val_id:
                obs_id, exp_id = val_id.split('M1')
                # The form of this inst is different to the standard in DAXA/SAS (M1), because emanom log files
                #  are named with mos1 and mos2 rather than M1 and M2
                inst = 'mos1'
            elif 'M2' in val_id:
                obs_id, exp_id = val_id.split('M2')
                # The form of this inst is different to the standard in DAXA/SAS (M2), because emanom log files
                #  are named with mos1 and mos2 rather than M1 and M2
                inst = 'mos2'
            else:
                raise ValueError("Somehow there is no instance of M1 or M2 in that storage key, this should be "
                                 "impossible!")

            evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            # Checking for the output anom file created by the process (unless turned off with an argument)
            log_name = "{i}{eid}-anom.log".format(i=inst, eid=exp_id)
            final_path = dest_dir + log_name

            # If it doesn't already exist then we will create commands to generate it
            # TODO Decide whether this is the route I really want to follow for this (see issue #28)
            if not os.path.exists(final_path):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = emanom_cmd.format(d=temp_dir, ccf=ccf_path, ef=evt_list_file, of=log_name)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                miss_final_paths[miss.name][val_id] = final_path
                # Make sure to store the log file path so it can be parsed later to see which CCDs to keep
                miss_extras[miss.name][val_id] = {'log_path': final_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Checking for MOS CCD anomalous states'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress


def parse_emanom_out(log_file_path: str, acceptable_states: Union[List[str], str] = ('G', 'I', 'U')) -> List[int]:
    """
    A simple function to parse the output file created by the emanom SAS task, which seeks to identify which MOS
    CCDs are in 'anomalous states'. The most relevant pieces of information contained in the output file are the
    CCD IDs, and the state code; the 'state' values have the following meanings:

    Status- G is good at all energies
          - I is intermediate for E<1 keV
          - B is bad for E<1 keV
          - O is off, chip not in use
          - U is undetermined (low band counts <= 0)

    Users can specify which states they would like to keep using the 'acceptable_states' parameter.

    :param str log_file_path: The path to the output file created by emanom.
    :param List[str]/str acceptable_states: The CCD states which should be accepted. If a CCD is accepted then its
        ID will be returned by this function.
    :return: A list of CCD IDs which are in 'acceptable' states.
    :rtype: List[int]
    """
    # Have to make sure that the file is actually there first.
    if not os.path.exists(log_file_path):
        raise FileNotFoundError("That path to an emanom log file is not valid.")

    # It is possible for the user to just pass a single state, in which case we make it a list to allow
    #  for everything later on to be consistent
    if isinstance(acceptable_states, str):
        acceptable_states = [acceptable_states]

    # This just checks that all the user-passed states are actually valid states that can be generated by emanom.
    val_states = ['G', 'I', 'B', 'O', 'U']
    if not all([state in val_states for state in acceptable_states]):
        raise ValueError("The values in 'acceptable_states' must be part of the following set: "
                         "{}".format(', '.join(val_states)))

    # Open the file and read the lines out into a list
    with open(log_file_path, 'r') as loggo:
        file_lines = loggo.readlines()

    # We iterate through the lines of the list, splitting so that the CCD ID becomes a dictionary key and the
    #  status code becomes the dictionary value. This sort of hard coded approach is not ideal, but oh well
    proc_lines = {line.split('CCD: ')[-1].split(' Hard')[0]: line.split('Status: ')[-1].strip('\n')
                  for line in file_lines if 'CCD' in line}

    # Now we iterate through and just select those CCD IDs which are in the list of states provided by the user.
    the_chosen = [int(ccd_id) for ccd_id in proc_lines if proc_lines[ccd_id] in val_states]

    return the_chosen


