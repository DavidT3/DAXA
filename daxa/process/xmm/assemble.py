#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 12/12/2022, 12:50. Copyright (c) The Contributors
import os
from random import randint

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm._common import _sas_process_setup, sas_call, ALLOWED_XMM_MISSIONS


def epchain():
    pass


@sas_call
def emchain(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False):
    """

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

    # Define the form of the odfingest command that must be run to create an ODF summary file
    # odf_cmd = "cd {d}; export SAS_CCF={ccf}; echo $SAS_CCF; odfingest odfdir={odf_dir} outdir={out_dir}
    #  withodfdir=yes"
    em_cmd = "cd {d}; export SAS_CCF={ccf}; emchain odf={odf} instruments={i}; mv *EVLI*.FIT ../; mv *ATTTSR*.FIT ../; "
    # TODO Restore the deleting part
             # "cd ..; rm -r {d}"

    # TODO Once summary parser is built (see issue #34) we can make this an unambiguous path
    #  rather than a pattern matching path
    # The event list pattern that we want to check for at the end of the process
    evt_list_name = "P{o}{i}S*MIEVLI*.FIT"

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

        # Now we're iterating through the ObsIDs that have been selected for the current mission
        for obs_id in miss.filtered_obs_ids:
            for inst in miss.chosen_instruments:
                if inst in ['M1', 'M2']:
                    # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
                    #  files will be written to.
                    dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
                    ccf_path = dest_dir + 'ccf.cif'

                    odf_dir = miss.raw_data_path + obs_id + '/'

                    # Set up a temporary directory to work in (probably not really necessary in this case, but will be
                    #  in other processing functions).
                    temp_name = "tempdir_{}".format(randint(0, 1e+8))
                    temp_dir = dest_dir + temp_name + "/"

                    # This is where the final output calibration file will be stored
                    final_path = dest_dir + evt_list_name.format(o=obs_id, i=inst)

                    # If it doesn't already exist then we will create commands to generate it
                    # TODO Decide whether this is the route I really want to follow for this (see issue #28)
                    if not os.path.exists(final_path):
                        # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)

                        # Format the blank command string defined near the top of this function with information
                        #  particular to the current mission and ObsID
                        cmd = em_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, i=inst)

                        # Now store the bash command, the path, and extra info in the dictionaries
                        miss_cmds[miss.name][obs_id+inst] = cmd
                        miss_final_paths[miss.name][obs_id+inst] = final_path
                        miss_extras[miss.name][obs_id+inst] = {}

    # This is just used for populating a progress bar during generation
    process_message = 'Assembling MOS event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress
