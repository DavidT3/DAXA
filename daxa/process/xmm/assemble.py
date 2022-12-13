#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 13/12/2022, 12:09. Copyright (c) The Contributors
import os
from random import randint

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm._common import _sas_process_setup, sas_call, ALLOWED_XMM_MISSIONS


@sas_call
def epchain(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    This function runs the epchain SAS process on XMM missions in the passed archive, which assembles the
    PN-specific ODFs into combined photon event lists - rather than the per CCD files that existed before. A run of
    epchain for out of time (OOT) events is also performed as part of this function call. The epchain manual can be
    found here (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/epchain.pdf) and gives detailed
    explanations of the process.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with PN observations for
        which epchain should be run. This function will fail if no XMM missions are present in the archive.
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
    ep_cmd = "cd {d}; export SAS_CCF={ccf}; epchain odf={odf} odfaccess=odf exposure={e} schedule={s}; " \
             "mv *EVLI*.FIT ../; mv *ATTTSR*.FIT ../;cd ..; rm -r {d}"

    # TODO Once summary parser is built (see issue #34) we can make this an unambiguous path
    #  rather than a pattern matching path
    # The event list pattern that we want to check for at the end of the process
    evt_list_name = "P{o}PN{eid}PIEVLI*.FIT"

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
                if inst == 'PN':
                    # The location of the raw data
                    odf_dir = miss.raw_data_path + obs_id + '/'

                    # TODO Update this when I have built a SAS summary file parser (see issue #34)
                    # Try to figure out how many PN exposures there were, as epchain will not automatically
                    #  loop through them, it has to be run separately for each I think (unlike emchain)
                    pn_exp = list(set([f.split(obs_id)[1].split('PN')[1][:4] for f in os.listdir(odf_dir)
                                  if 'PNS' in f or 'PNU' in f]))
                    # Find just the 'scheduled' observations for now, this will be altered later on
                    sch_pn_exp = [pe for pe in pn_exp if pe[0] == 'S']

                    for exp_ind, exp_id in enumerate(sch_pn_exp):
                        # TODO Again update this after SAS summary parser (issue 34), because we currently try to
                        #  process everything as imaging mode (see issue #40)

                        # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
                        #  files will be written to.
                        dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
                        ccf_path = dest_dir + 'ccf.cif'

                        # Set up a temporary directory to work in (probably not really necessary in this case, but will be
                        #  in other processing functions).
                        temp_name = "tempdir_{}".format(randint(0, 1e+8))
                        temp_dir = dest_dir + temp_name + "/"
                        # This is where the final output calibration file will be stored
                        final_path = dest_dir + evt_list_name.format(o=obs_id, eid=exp_id)

                        # If it doesn't already exist then we will create commands to generate it
                        # TODO Decide whether this is the route I really want to follow for this (see issue #28)
                        if not os.path.exists(final_path):
                            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)

                            # Format the blank command string defined near the top of this function with information
                            #  particular to the current mission and ObsID
                            # TODO If unscheduled observations are supported, will need to alter the s= part
                            cmd = ep_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, e=exp_id[1:], s='S')

                            # Now store the bash command, the path, and extra info in the dictionaries
                            miss_cmds[miss.name][obs_id + inst + exp_id] = cmd
                            miss_final_paths[miss.name][obs_id + inst + exp_id] = final_path
                            miss_extras[miss.name][obs_id + inst + exp_id] = {}

    # This is just used for populating a progress bar during generation
    process_message = 'Assembling PN event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress


@sas_call
def emchain(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    This function runs the emchain SAS process on XMM missions in the passed archive, which assembles the
    MOS-specific ODFs into combined photon event lists - rather than the per CCD files that existed before. The
    emchain manual can be found here (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/emchain.pdf) and
    gives detailed explanations of the process.

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
    em_cmd = "cd {d}; export SAS_CCF={ccf}; emchain odf={odf} instruments={i}; mv *EVLI*.FIT ../; " \
             "mv *ATTTSR*.FIT ../; cd ..; rm -r {d}"

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
