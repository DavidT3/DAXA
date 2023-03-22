#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 22/03/2023, 11:51. Copyright (c) The Contributors

# This part of DAXA is for wrapping SAS functions that are relevant to the processing of XMM data, but don't directly
#  assemble/clean event lists etc.

import os
from datetime import datetime
from random import randint
from typing import Union, Tuple

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.process.xmm._common import _sas_process_setup, ALLOWED_XMM_MISSIONS, sas_call


@sas_call
def cif_build(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
              analysis_date: Union[str, datetime] = 'now', timeout: Quantity = None) \
        -> Tuple[dict, dict, dict, str, int, bool, Quantity]:
    """
    A DAXA Python interface for the SAS cifbuild command, used to generate calibration files for XMM observations
    prior to processing. The observation date is supplied by the XMM mission instance(s), and is the date when the
    observation was started (as acquired from the XSA).

    :param Archive obs_archive: An Archive instance containing XMM mission instances for which observation calibration
        files should be generated. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param str/datetime analysis_date: The analysis date for which to generate calibration file. The default is
        'now', but this parameter can be used to create calibration files as they would have been on a past date.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire cif_build process, but a timeout for individual
        ObsID processes.
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

    # This string contains the bash code to run cifbuild, and will be filled in for each
    #  observation within each XMM mission
    # TODO DECIDE WHETHER TO KEEP FULLPATH=YES OR NOT
    cif_cmd = "cd {d}; cifbuild calindexset=ccf.cif withobservationdate=yes " \
              "observationdate={od} analysisdate={ad} fullpath=yes; mv * ../; cd ..; rm -r {n}"

    if isinstance(analysis_date, datetime):
        # If an analysis date object, we need to convert it to the right string format for cifbuild
        analysis_date = analysis_date.strftime('%Y-%m-%d')
    elif analysis_date == 'now':
        # The cifbuild SAS tool will take 'now' as input, but I would rather use that flag to trigger grabbing the
        #  current date from datetime - as then the actual date used will be in the memory of this Python module
        analysis_date = datetime.today().strftime('%Y-%m-%d')
    else:
        raise ValueError("The analysis_date argument must either be a valid datetime object, or a "
                         "'now' string.")

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

        # Grabs the Pandas dataframe of observation information for those observations that have been selected
        #  by the mission - makes a copy just to be safe (I don't think its probably necessary but I'm doing it
        #  anyway).
        filtered_obs_info = miss.filtered_obs_info.copy()

        # Now we're iterating through the ObsIDs that have been selected for the current mission
        for obs_id in miss.filtered_obs_ids:
            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.get_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            # Grab the start date of the observation from the observation info dataframe - it is a Pandas datetime
            #  object and thus we can use strftime to output a string in the format that we need
            obs_date = filtered_obs_info[filtered_obs_info['ObsID'] == obs_id].iloc[0]['start'].strftime('%Y-%m-%d')

            # This is where the final output calibration file will be stored
            final_path = dest_dir + "ccf.cif"

            # If it doesn't already exist then we will create commands to generate it
            # TODO Decide whether this is the route I really want to follow for this (see issue #28)
            if not os.path.exists(final_path):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = cif_cmd.format(d=temp_dir, od=obs_date, n=temp_name, ad=analysis_date)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id] = cmd
                miss_final_paths[miss.name][obs_id] = final_path
                miss_extras[miss.name][obs_id] = {'obs_date': obs_date, 'analysis_date': analysis_date}

    # This is just used for populating a progress bar during generation
    process_message = 'Generating calibration files'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def odf_ingest(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
               timeout: Quantity = None):
    """
    This function runs the SAS odfingest task, which creates a summary of the raw data available in the ODF
    directory, and is used by many SAS processing tasks.

    :param Archive obs_archive: An Archive instance containing XMM mission instances for which observation summary
        files should be generated. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire odf_ingest process, but a timeout for individual
        ObsID processes.
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

    # Define the form of the odfingest command that must be run to create an ODF summary file
    odf_cmd = "cd {d}; export SAS_CCF={ccf}; odfingest odfdir={odf_dir} outdir={out_dir} withodfdir=yes"

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

        # Grabs the Pandas dataframe of observation information for those observations that have been selected
        #  by the mission - makes a copy just to be safe (I don't think its probably necessary but I'm doing it
        #  anyway).
        filtered_obs_info = miss.filtered_obs_info.copy()

        # Now we're iterating through the ObsIDs that have been selected for the current mission
        for obs_id in miss.filtered_obs_ids:
            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            # dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
            raw_dir = miss.raw_data_path + obs_id + '/'
            proc_dir = obs_archive.get_processed_data_path(miss, obs_id)
            ccf_path = proc_dir + 'ccf.cif'

            rev = filtered_obs_info[filtered_obs_info['ObsID'] == obs_id].iloc[0]['revolution']
            rev = str(rev).zfill(4)

            # This is where the final output calibration file will be stored
            final_path = raw_dir + "{r}_{o}_SCX00000SUM.SAS".format(r=rev, o=obs_id)
            # This file should be deleted if it already exists
            if os.path.exists(final_path):
                os.remove(final_path)

            # The path to the ODF (raw data) for this ObsID
            odf_path = miss.raw_data_path + obs_id + '/'

            # Construct the command with relevant information
            cmd = odf_cmd.format(d=proc_dir, ccf=ccf_path, odf_dir=odf_path, out_dir=odf_path)

            # Now store the bash command, the path, and extra info in the dictionaries
            miss_cmds[miss.name][obs_id] = cmd
            miss_final_paths[miss.name][obs_id] = final_path
            miss_extras[miss.name][obs_id] = {'sum_path': final_path}

            # This is just used for populating a progress bar during generation
        process_message = 'Generating ODF summary files'

        return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout
