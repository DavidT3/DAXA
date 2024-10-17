#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 11/10/2024, 17:07. Copyright (c) The Contributors

# This part of DAXA is for wrapping SAS functions that are relevant to the processing of XMM data, but don't directly
#  assemble/clean event lists etc.

import os
from datetime import datetime
from random import randint
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoProcessingError
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
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"

            # Grab the start date of the observation from the observation info dataframe - it is a Pandas datetime
            #  object and thus we can use strftime to output a string in the format that we need
            obs_date = filtered_obs_info[filtered_obs_info['ObsID'] == obs_id].iloc[0]['start'].strftime('%Y-%m-%d')

            # This is where the final output calibration file will be stored
            final_path = dest_dir + "ccf.cif"

            # As this is the first process in the chain, we need to account for the fact that nothing has been run
            #  before, and using the process_success property might raise an exception
            try:
                check_dict = obs_archive.process_success[miss.name]['cif_build']
            except (NoProcessingError, KeyError):
                check_dict = {}

            # If it doesn't already exist then we will create commands to generate it
            if obs_id not in check_dict:
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

        # This allows us to get a boolean array corresponding to the ObsIDs letting us know which cifbuilds
        #  worked, though tbh they should all work or none of them in my experience.
        cif_good = obs_archive.check_dependence_success(miss.name, [[o] for o in filtered_obs_info['ObsID'].values],
                                                        'cif_build')

        # Now we're iterating through the ObsIDs that have been selected for the current mission
        for obs_id in miss.filtered_obs_ids[cif_good]:
            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            # dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
            raw_dir = miss.raw_data_path + obs_id + '/'
            proc_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = proc_dir + 'ccf.cif'

            rev = filtered_obs_info[filtered_obs_info['ObsID'] == obs_id].iloc[0]['revolution']
            rev = str(rev).zfill(4)

            # This is where the final output calibration file will be stored
            final_path = raw_dir + "{r}_{o}_SCX00000SUM.SAS".format(r=rev, o=obs_id)
            # This file should be deleted if it already exists - IF IT IS THE ORIGINAL THAT WAS DOWNLOADED. Hence
            #  why I've included the clunky extra logic. If a previous run of odf_ingest was successful then we don't
            #  need to redo anything
            if os.path.exists(final_path) and ('odf_ingest' not in obs_archive.process_success[miss.name] or
                                               obs_id not in obs_archive.process_success[miss.name]['odf_ingest'] or
                                               not obs_archive.process_success[miss.name]['odf_ingest'][obs_id]):
                os.remove(final_path)

            # If it doesn't already exist then we will create commands to generate it
            if ('odf_ingest' not in obs_archive.process_success[miss.name] or
                    obs_id not in obs_archive.process_success[miss.name]['odf_ingest']):
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


def parse_odf_sum(sum_path: str, obs_id: str = None):
    """
    A function that takes a path to an XMM SAS summary file generated by the odf ingest command. The file will be
    filtered and parsed so that data relevant to DAXA processing valid scientific observations can be extracted.
    This includes things like whether a particular instrument was active, the number of sub-exposures, whether those
    sub-exposures were in a science mode that produces data useful for the study of astrophysical objects (i.e.
    not in a calibration or diagnosis mode). Data relevant to SAS procedures that calibrate and construct exposure
    lists is not included in the output of this function.

    :param str sum_path: The path to the odf_ingest-generated summary file that is to be parsed into a dictionary
        of relevant information.
    :param str obs_id: Optionally, the observation ID that goes with this summary file can be passed, purely to
        make a possible error message more useful.
    :return: Multi-level dictionary of information, with top-level keys being instrument names. Next level contains
        information on whether the instrument was active and the number of exposures. This level has an 'exposures'
        key which is for a dictionary where the keys are all the exp_ids available for this instrument. Those keys
        are for dictionaries of exposure specific information, including mode, whether the exposure was
        scheduled, and the modes of the individual CCDs.
    :rtype: dict
    """

    def inst_sec_parser(sec_lines: List[str], inst: str):
        """
        This is an internal function to extract necessary information for specific instruments from a SAS summary
        file generated by odf ingest. It returns a multi-level dictionary with the relevant information, excluding
        the majority of the data in the SAS summary file as it is of no use to DAXA.

        :param List[str] sec_lines: A list of summary file lines for the instrument section.
        :param str inst: The instrument which we're looking at.
        :return: Multi-level dictionary of information, top level key is the inst passed to the function, so this
            dict can be added to an external dict easily, next level contains information on whether the instrument
            was active and the number of exposures. This level has an 'exposures' key which is for a dictionary
            where the keys are all the exp_ids available for this instrument. Those keys are for dictionaries of
            exposure specific information, including mode, whether the exposure was scheduled, and the modes of the
            individual CCDs.
        :rtype: dict
        """
        # This dictionary will store configuration and observation information about the instrument that has
        #  been passed into this parsing function
        info_dict = {}

        # First of all we check that the instrument was actually active, no point proceeding further if that
        #  isn't true - we do still return a dictionary with active False if it was turned off though
        if sec_lines[0][0] == 'Y':
            info_dict['active'] = True
        else:
            info_dict['active'] = False
            # The eventual return has the instrument as the top level key, so I have to do that here were we're
            #  exiting early
            return {inst: info_dict}

        # Turning the instrument section lines list into a Pandas series, purely because it has the str.contains
        #  method which makes it way easier to find the lines I'm looking for.
        sec_lines = pd.Series(sec_lines)
        # First of all find the number of exposures specified in the top level of the instrument section
        info_dict['num_exp'] = int(sec_lines[sec_lines.str.contains('Number of exposures for '
                                                                    'this instrument')].iloc[0].split(' ')[0])

        # The sub-levels of this instrument section are specific exposure sections - we locate the indices where they
        #  begin. This will allow us to create sub-sections to parse from
        exp_sec_inds = np.where(sec_lines == 'EXPOSURE')[0]
        # Check to see whether the number of exposure sections matches up to the number specified at the top. I hope
        #  it always does, but if not we'll get some warning.
        if len(exp_sec_inds) != info_dict['num_exp'] and obs_id is None:
            raise ValueError("{i} SAS summary file number of exposure headers ({eh}) is different from stated "
                             "number of exposures ({ne}).".format(i=inst, eh=len(exp_sec_inds),
                                                                  ne=info_dict['num_exp']))
        elif len(exp_sec_inds) != info_dict['num_exp'] and obs_id is not None:
            raise ValueError("{i} SAS summary file number of exposure headers ({eh}) is different from stated "
                             "number of exposures ({ne}) - for {oi} summary file.".format(i=inst, eh=len(exp_sec_inds),
                                                                                          ne=info_dict['num_exp'],
                                                                                          oi=obs_id))

        # Need to add an index on the end so the last exposure section has an endpoint, I make it the last line (so
        #  the end of the instrument section).
        exp_sec_inds = np.append(exp_sec_inds, -1)

        # Now we create specific sub-exposure section sets of lines. The keys in this case are extracted in such
        #  a way that they have the S or U prefix, specifying scheduled or unscheduled. As the numerical IDs after
        #  S and U (i.e. 001, 002, 003, 004, ...) aren't necessarily unique between scheduled and unscheduled
        #  exposures, we are being safe.
        exp_secs = {sec_lines[exp_sec_inds[esi]+1].split('[also ')[-1].split(']')[0]:
                        sec_lines[exp_sec_inds[esi]+2: exp_sec_inds[esi+1]] for esi in range(0, len(exp_sec_inds)-1)}

        # Adding a key/dictionary to the overall info dict to information about individual sub-exposures
        info_dict['exposures'] = {}
        # Now iterating through specific sub-exposures, remembering that e_sec in this case will be the unique
        #  sub-exposure ID; e.g. S001, U004, etc.
        for e_sec in exp_secs:
            # Grab the information for this exposure to minimise line lengths in the rest of this as much as possible
            cur_s = exp_secs[e_sec]
            # Extract some of the information we want for this exposure - scheduled can be defined via the
            #  first character of the exposure ID, the type (e.g. SCIENCE) and mode (e.g. PRIME FULL WINDOW) are
            #  extracted from their entries.
            exp_info = {'scheduled': e_sec[0] == 'S',
                        'type': cur_s[cur_s.str.contains('/ Exposure Type')].iloc[0].split(' ')[0],
                        'mode': cur_s[cur_s.str.contains('/ Instrument '
                                                         'configuration')].iloc[0].split('= ')[-1].split(' ')[0]}

            # Not every exposure sub-section will have a filter entry - for instance diagnostic modes don't, and
            #  RGS exposures never do, even for usable exposures.
            filt_search = cur_s[cur_s.str.contains('FILTER = ')]
            # If there are some lines part-matching FILTER  = then we know there is a filter entry
            if len(filt_search) != 0:
                # Grab the filter, e.g. thin, medium, etc.
                exp_info['filter'] = filt_search.iloc[0].split(' = ')[-1].split(' ')[0]
            # If no filter entry, then we set the dictionary entry to None - better to have a null entry than
            #  have to check if there is an entry there
            else:
                exp_info['filter'] = None

            # We're also going to record the CCD modes, because there are some observing modes (I think) for MOS
            #  cameras where they can differ - also I'm hoping this will tell us when some CCDs are turned off.
            exp_info['ccd_modes'] = {int(ccd.split('DATA_MODE_')[-1].split(' = ')[0]):
                                         ccd.split(' = ')[-1].split(' /')[0]
                                     for ccd in cur_s[cur_s.str.contains('/ Data mode for CCD')]}
            # This particular sub-exposure is added to the greater overall dictionary
            info_dict['exposures'][e_sec] = exp_info

        # Return the dictionary we just assembled as part of another dictionary - that means this return can be
        #  added to an external storage dictionary for all instruments more easily.
        return {inst: info_dict}

    # We open up the SAS summary file that has been generated by ODF ingest
    with open(sum_path, 'r') as reado:
        # Reading out the lines, we strip out any which are purely comment lines (i.e. begin with // ) and make
        #  this list an array for future ease
        sum_lines = np.array([line for line in reado.readlines() if line[:3] != '// '])

    # First off, find which line indices are just a break with a newline
    sec_div_ind = np.where(sum_lines == '//\n')[0]
    # Find the index I want to start at, which is where the summary starts to list what files are in the ODF
    start_ind = np.where(sum_lines == 'FILES\n')[0][0]-1
    # Select only those break lines which occur after the start index I've just defined
    sec_div_ind = sec_div_ind[sec_div_ind > start_ind]
    # Then add that start index onto the beginning of the break line index array - we want to be able to bracket
    #  sections with indices
    sec_div_ind = np.concatenate(([start_ind], sec_div_ind))
    # Then I try to strip some 'sections' defined by the break lines which don't actually have anything in them,
    #  and are really just two or more break lines one after the other
    ind_diff = np.ediff1d(sec_div_ind, len(sum_lines)-sec_div_ind[-1])
    sec_div_ind = sec_div_ind[np.where(ind_diff != 1)[0]]

    # We find those indices of break lines which are for an instrument section, which is what we really care about
    #  in terms of extracting info that is useful to DAXA
    inst_head = sec_div_ind[np.where(sum_lines[sec_div_ind+1] == 'INSTRUMENT\n')[0]]
    # Need to add an endpoint here, so that the last instrument section index in the above array can be used
    #  with the next index to bracket the section
    inst_head = np.append(inst_head, -1)

    # Just stripping the end lines of off all the strings - makes my life easier from here on
    sum_lines = np.array([sl.strip('\n') for sl in sum_lines])

    # Build separate sets of lines for the different instrument sections - they will be parsed individually to
    #  create a (very) multi-leveled dictionary of relevant information
    inst_secs = {sum_lines[inst_head[i]+2]: sum_lines[inst_head[i]+3: inst_head[i + 1]]
                 for i in range(0, len(inst_head) - 1)}

    # This dictionary stores the information dictionaries extracted for each of instruments - the returned dicts
    #  have an instrument key so we can just use update to add them to this empty dictionary
    sum_sas_dict = {}
    for inst_name in inst_secs:
        sum_sas_dict.update(inst_sec_parser(inst_secs[inst_name], inst_name))

    return sum_sas_dict

