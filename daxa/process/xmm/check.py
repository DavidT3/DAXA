#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/09/2024, 12:54. Copyright (c) The Contributors
import os
from random import randint
from typing import Union, List
from warnings import warn

import numpy as np
from astropy.units import Quantity
from packaging.version import Version

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm._common import ALLOWED_XMM_MISSIONS, _sas_process_setup, sas_call


@sas_call
def emanom(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False, timeout: Quantity = None):
    """
    This function runs the SAS emanom function, which attempts to identify when MOS CCDs are have operated in an
    'anomalous' state, where the  background at E < 1 keV is strongly enhanced. Data above 2 keV are unaffected, so
    CCDs in anomalous states used for science where the soft X-rays are unnecessary do not need to be excluded.

    The emanom task calculates the (2.5-5.0 keV)/(0.4-0.8 keV) hardness ratio from the corner data to determine
    whether a chip is in an anomalous state. However, it should be noted that the "anonymous" anomalous state of
    MOS1 CCD#4 is not always detectable from the unexposed corner data.

    This functionality is only usable if you have SAS v19.0.0 or higher - a version check will be performed and
    a warning raised (though no error will be raised) if you use this function with an earlier SAS version.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with MOS observations for
        which emchain should be run. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire emanom process, but a timeout for individual
        ObsID-subexposure processes.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool, Quantity]
    """
    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # As it turns out, emanom was only introduced in v19.0.0. Thankfully emanom is optional in processing XMM, so
    #  I don't have to change the required SAS version - I'll just put a version check here
    if sas_version < Version('19.0.0'):
        warn("The emanom task was introduced in SAS v19.0.0, you have SAS {} - skipping "
             "emanom.".format(str(sas_version)), stacklevel=2)
        return {}, {}, {}, '', num_cores, disable_progress, timeout

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

        # The loop of instruments is necessary because it is possible, if unlikely, that the user only selected
        #  one of the MOS instruments when setting up the mission
        rel_obs_info = []
        for inst in [i for i in miss.chosen_instruments if i[0] == 'M']:
            rel_obs_info += obs_archive.get_obs_to_process(miss.name, inst)

        # Here we check that emchain ran - if it didn't then we can hardly search the MOS event lists for a badly
        #  behaved CCD!
        good_em = obs_archive.check_dependence_success(miss.name, rel_obs_info, 'emchain')

        # Now we start to cycle through the relevant data
        for obs_info in np.array(rel_obs_info)[good_em]:
            # This is the valid id that allows us to retrieve the specific event list for this ObsID-M1/2-SubExp
            #  combination
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # emanom has a different instrument naming convention in its files (because of course it does), so we
            #  need to be able to catch that.
            if inst == 'M1':
                alt_inst = 'mos1'
            else:
                alt_inst = 'mos2'

            # Grab the relevant event list from the extra information of the emchain (process that created the
            #  event list) process
            evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"

            # Checking for the output anom file created by the process (unless turned off with an argument)
            log_name = "{i}{eid}-anom.log".format(i=alt_inst, eid=exp_id)
            final_path = dest_dir + log_name

            # If it doesn't already exist then we will create commands to generate it
            if ('emanom' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['emanom']):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = emanom_cmd.format(d=temp_dir, ccf=ccf_path, ef=evt_list_file, of=log_name)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                miss_final_paths[miss.name][val_id] = final_path
                # Make sure to store the log file path, so it can be parsed later to see which CCDs to keep
                miss_extras[miss.name][val_id] = {'log_path': final_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Checking for MOS CCD anomalous states'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


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
        raise FileNotFoundError("That path ({}) to an emanom log file is not valid.".format(log_file_path))

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


