#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/09/2024, 15:41. Copyright (c) The Contributors

import os
from copy import deepcopy
from random import randint
from typing import Union, List, Tuple

import numpy as np
from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process._backend_check import find_lcurve
from daxa.process._cleanup import _last_process
from daxa.process.xmm._common import _sas_process_setup, sas_call, ALLOWED_XMM_MISSIONS
from daxa.process.xmm.check import parse_emanom_out


# TODO YOU ALSO NEED TO MAKE SURE THAT THE DOCUMENTATION REFLECTS ALL THE CHANGES YOU HAVE MADE


@sas_call
def epchain(obs_archive: Archive, process_unscheduled: bool = True, num_cores: int = NUM_CORES,
            disable_progress: bool = False, timeout: Quantity = None):
    """
    This function runs the epchain SAS process on XMM missions in the passed archive, which assembles the
    PN-specific ODFs into combined photon event lists - rather than the per CCD files that existed before. A run of
    epchain for out of time (OOT) events is also performed as part of this function call. The epchain manual can be
    found here (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/epchain.pdf) and gives detailed
    explanations of the process.

    Per the advice of the SAS epchain manual, the OOT event list epchain call is performed first, and its intermediate
    files are saved and then used for the normal call to epchain.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with PN observations for
        which epchain should be run. This function will fail if no XMM missions are present in the archive.
    :param bool process_unscheduled: Whether this function should also process sub-exposures marked 'U', for
        unscheduled. Default is True, in which case they will be processed.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire epchain process, but a timeout for individual
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

    # Define the form of the odfingest command that must be run to create an ODF summary file
    # Per the advice of the SAS epchain manual, the OOT event list epchain call is performed first, and its intermediate
    #  files are saved and then used for the normal call to epchain.
    ep_cmd = "cd {d}; export SAS_CCF={ccf}; epchain odf={odf} odfaccess=odf exposure={e} schedule={s} ccds={c} " \
             "runbackground=N keepintermediate=raw withoutoftime=Y; epchain odf={odf} odfaccess=odf exposure={e} " \
             "schedule={s} ccds={c} runatthkgen=N runepframes=N runbadpixfind=N runbadpix=N; mv *EVLI*.FIT ../; " \
             "mv *ATTTSR*.FIT ../;cd ..; rm -r {d}; mv {oge} {fe}; mv {ogoote} {foote}"

    # The event list pattern that that should exist after the SAS process, which we will rename to our convention
    prod_evt_list_name = "P{o}PN{eid}PIEVLI0000.FIT"
    prod_oot_evt_list_name = "P{o}PN{eid}OOEVLI0000.FIT"

    # These represent the final names and resting places of the event lists
    evt_list_name = "obsid{o}-inst{i}-subexp{se}-events.fits"
    oot_evt_list_name = "obsid{o}-inst{i}-subexp{se}-ootevents.fits"

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

        # This method will fetch the valid data (ObsID, Instrument, and sub-exposure) that we need to process. When
        #  given PN as a search term (as this is epchain, we don't need to include MOS) only PN identifiers will be
        #  returned.
        # I prefer this to how I originally wrote this, as it saves multiple layers of for loops/if statements, which
        #  can be a little tricky to decipher
        rel_obs_info = obs_archive.get_obs_to_process(miss.name, 'PN')

        # Don't just launch straight into the loop however, as the user can choose NOT to process unscheduled
        #  observations. In that case we clean that rel_obs_info list.
        if not process_unscheduled:
            # Select only those sub exposures that have an ident that doesn't start with a U
            rel_obs_info = [roi for roi in rel_obs_info if roi[2][0] != 'U']

        # Here we check that the previous required processes ran, mainly to be consistent. I know that odf ingest
        #  worked if we have rel_obs_info data, because odf_ingest is what populated the information get_obs_to_process
        #  uses for XMM.
        good_odf = obs_archive.check_dependence_success(miss.name, [[roi[0]] for roi in rel_obs_info], 'odf_ingest')

        # Now we start to cycle through the relevant data
        for obs_info in np.array(rel_obs_info)[good_odf]:
            # Unpack the observation information provided by the
            obs_id, inst, exp_id = obs_info

            # The location of the raw data
            odf_dir = miss.raw_data_path + obs_id + '/'

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Want to identify which CCDs can be processed, i.e. were turned on and were in imaging mode rather
            #  than timing mode - perhaps timing mode will be supported by DAXA later on.
            ccd_modes = obs_archive.observation_summaries[miss.name][obs_id][inst]['exposures'][exp_id]['ccd_modes']
            ccd_ids = [ccd_id for ccd_id, ccd_mode in ccd_modes.items() if ccd_mode.upper() == 'IMAGING']

            # Then turn into a string so as they can be passed to the epchain command we're constructing
            ccd_str = ",".join([str(c_id) for c_id in sorted(ccd_ids)])

            # Set up a temporary directory to work in
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"
            # This is where the processes will output the event list files
            og_out_path = os.path.join(dest_dir, prod_evt_list_name.format(o=obs_id, eid=exp_id))
            og_oot_out_path = os.path.join(dest_dir, prod_oot_evt_list_name.format(o=obs_id, eid=exp_id))

            # This is where the final output event list file will be stored - after moving and renaming
            final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, se=exp_id, i='PN'))
            oot_final_path = os.path.join(dest_dir, 'events', oot_evt_list_name.format(o=obs_id, se=exp_id, i='PN'))

            # If it doesn't already exist then we will create commands to generate it - there are no options for
            #  epchain that could be changed between runs (other than processing unscheduled, but we're looping
            #  through those commands separately), so it's safe to take what has already been generated.
            # We check to see if the process has been run (whether it was a success or failure) for the current
            #  data for the archive
            if ('epchain' not in obs_archive.process_success[miss.name] or
                    (obs_id + inst + exp_id) not in obs_archive.process_success[miss.name]['epchain']):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = ep_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, e=exp_id[1:], s=exp_id[0],
                                    c=ccd_str, oge=og_out_path, ogoote=og_oot_out_path, fe=final_path,
                                    foote=oot_final_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id + inst + exp_id] = cmd
                # The SAS wrapping functionality can check that multiple final files exist
                miss_final_paths[miss.name][obs_id + inst + exp_id] = [final_path, oot_final_path]
                miss_extras[miss.name][obs_id + inst + exp_id] = {'evt_list': final_path,
                                                                  'oot_evt_list': oot_final_path}

    # This is just used for populating a progress bar during generation
    process_message = 'Assembling PN and PN-OOT event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def emchain(obs_archive: Archive, process_unscheduled: bool = True, num_cores: int = NUM_CORES,
            disable_progress: bool = False, timeout: Quantity = None):
    """
    This function runs the emchain SAS process on XMM missions in the passed archive, which assembles the
    MOS-specific ODFs into combined photon event lists - rather than the per CCD files that existed before. The
    emchain manual can be found here (https://xmm-tools.cosmos.esa.int/external/sas/current/doc/emchain.pdf) and
    gives detailed explanations of the process.

    The DAXA wrapper does not allow emchain to automatically loop through all the sub-exposures for a given
    ObsID-MOSX combo, but rather creates a separate process call for each of them. This allows for greater
    parallelisation (if on a system with a significant core count), but also allows the same level of granularity
    in the logging of processing of different sub-exposures as in DAXA's epchain implementation.

    The particular CCDs to be processed are not specified in emchain, unlike in epchain, because it can sometimes
    have unintended consequences. For instance processing a MOS observation in FastUncompressed mode, with timing
    on CCD 1 and imaging everywhere else, can cause emchain to fail (even though no actual failure occurs) because
    the submode is set to Unknown, rather than FastUncompressed.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with MOS observations for
        which emchain should be run. This function will fail if no XMM missions are present in the archive.
    :param bool process_unscheduled: Whether this function should also process sub-exposures marked 'U', for
        unscheduled. Default is True, in which case they will be processed.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire emchain process, but a timeout for individual
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

    # We run a backend check for the lcurve tool, as emchain seems to require it to complete successfully. This will
    #  throw an error if it does not find lcurve
    heasoft_version = find_lcurve()

    # Define the form of the emchain command that must be run to create a combined MOS1/2 event list - the exposures
    #  argument will only ever be set with one exposure at a time. emchain does loop through sub-exposures
    #  automatically, but I'm attempting to normalise the behaviours between emchain and epchain in how DAXA calls
    #  them. Issue #42 discusses this.
    # addtaglenoise and makeflaregti are disabled because DAXA already has equivalents, emanom and espfilt
    em_cmd = "cd {d}; export SAS_CCF={ccf}; emchain odf={odf} instruments={i} exposures={ei} addtaglenoise=no " \
             "makeflaregti=no; mv *MIEVLI*.FIT ../; mv *ATTTSR*.FIT ../; cd ..; rm -r {d}; mv {oge} {fe}"

    # The event list name that we want to check for at the end of the process - the zeros at the end seem to always
    #  be there for emchain-ed event lists, which is why I'm doing it this way rather than with a wildcard * at the
    #  end (which DAXA does support in the sas_call stage). This is the name of the file produced by the SAS call
    prod_evt_list_name = "P{o}{i}{ei}MIEVLI0000.FIT"
    # This represents the final names and resting places of the event lists
    evt_list_name = "obsid{o}-inst{i}-subexp{se}-events.fits"

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

        # This method will fetch the valid data (ObsID, Instrument, and sub-exposure) that we need to process. When
        #  given M1 (for instance) as a search term only MOS1 identifiers will be returned. As emchain needs to
        #  process MOS1 and MOS2 data, I just run it twice and add the results together
        # I prefer this to how I originally wrote this, as it saves multiple layers of for loops/if statements, which
        #  can be a little tricky to decipher.
        # The loop of instruments is necessary because it is possible, if unlikely, that the user only selected
        #  one of the MOS instruments when setting up the mission
        rel_obs_info = []
        for inst in [i for i in miss.chosen_instruments if i[0] == 'M']:
            rel_obs_info += obs_archive.get_obs_to_process(miss.name, inst)

        # Don't just launch straight into the loop however, as the user can choose NOT to process unscheduled
        #  observations. In that case we clean that rel_obs_info list.
        if not process_unscheduled:
            # Select only those sub exposures that have an ident that doesn't start with a U
            rel_obs_info = [roi for roi in rel_obs_info if roi[2][0] != 'U']

        # Here we check that the previous required processes ran, mainly to be consistent. I know that odf ingest
        #  worked if we have rel_obs_info data, because odf_ingest is what populated the information get_obs_to_process
        #  uses for XMM.
        good_odf = obs_archive.check_dependence_success(miss.name, [[roi[0]] for roi in rel_obs_info], 'odf_ingest')

        # Now we start to cycle through the relevant data
        for obs_info in np.array(rel_obs_info)[good_odf]:
            # Unpack the observation information provided by the
            obs_id, inst, exp_id = obs_info

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Grab the path to the ODF directory, we shall need it
            odf_dir = miss.raw_data_path + obs_id + '/'

            # ATTENTION - this was left here because it may be useful in the future, but specifying which CCDs to
            #  process can have some unintended consequences in emchain that I do not care to fully explore right now
            #  As such I'm going to allow emchain itself to figure out which CCDs are in timing mode etc.

            # Want to identify which CCDs can be processed, i.e. were turned on and were in imaging mode rather
            #  than timing mode - perhaps timing mode will be supported by DAXA later on. This same check
            #  happens exactly the same way in epchain
            # ccd_modes = obs_archive.observation_summaries[miss.name][obs_id][inst]['exposures'][exp_id]['ccd_modes']
            # ccd_ids = [ccd_id for ccd_id, ccd_mode in ccd_modes.items() if ccd_mode.upper() == 'IMAGING']
            #
            # # Then turn into a string so as they can be passed to the epchain command we're constructing - the
            # #  construction of this CCD list is NOT the same as in epchain, because the two tasks require different
            # #  formats for lists... (and they don't seem to define them anywhere!!)
            # ccd_str = "'" + " ".join([str(c_id) for c_id in sorted(ccd_ids)]) + "'"

            # Set up a temporary directory to work in
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"

            # This is where the final output event list file will be stored
            og_out_path = dest_dir + prod_evt_list_name.format(o=obs_id, i=inst, ei=exp_id)
            # This is where the final output event list file will be stored - after moving and renaming
            final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, se=exp_id, i=inst))

            # If it doesn't already exist then we will create commands to generate it - there are no options for
            #  emchain that could be changed between runs (other than processing unscheduled, but we're looping
            #  through those commands separately), so it's safe to take what has already been generated.
            if ('emchain' not in obs_archive.process_success[miss.name] or
                    (obs_id + inst + exp_id) not in obs_archive.process_success[miss.name]['emchain']):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = em_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, i=inst, ei=exp_id, fe=final_path,
                                    oge=og_out_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id + inst + exp_id] = cmd
                miss_final_paths[miss.name][obs_id + inst + exp_id] = final_path
                miss_extras[miss.name][obs_id + inst + exp_id] = {'evt_list': final_path}

    # This is just used for populating a progress bar during generation
    process_message = 'Assembling MOS event lists'
    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def rgs_events(obs_archive: Archive, process_unscheduled: bool = True,  num_cores: int = NUM_CORES,
               disable_progress: bool = False, timeout: Quantity = None):
    """
    This function runs the first step of the SAS RGS processing pipeline, rgsproc. This should prepare the RGS event
    lists by calibrating and combining the separate CCD event lists into RGS events. This happens separately for RGS1
    and RGS2, and for each sub-exposure of the two instruments.

    None of the calculations performed in this step should be affected by the choice of source, the first step where
    the choice of primary source should be taken into consideration is the next step, rgs_angles; though as DAXA
    processes data to be generally useful we will not define a primary source, that is for the user in the future as
    the aspect drift calculations can be re-run.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with RGS observations for
        which RGS processing should be run. This function will fail if no XMM missions are present in the archive.
    :param bool process_unscheduled: Whether this function should also process sub-exposures marked 'U', for
        unscheduled. Default is True, in which case they will be processed.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire rgs_events process, but a timeout for individual
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

    # Define the form of the rgsproc command that will be executed this function. This DAXA function, rgs_events, only
    #  deals with the first stage of processing, hence why entrystage and finalstage are both one.
    # As we are effectively splitting up an existing pipeline, I actually leave the temporary directories (and final
    #  files) in place until later in the chain
    rgp_cmd = "cd {d}; export SAS_CCF={ccf}; export SAS_ODF={odf}; rgsproc entrystage=1:events finalstage=1:events " \
              "withinstexpids=true instexpids={ei}"  # ; mv *.FIT ../; cd ..; rm -r {d}

    # The event list name that we want to check for at the end of the process - the zeros at the end seem to always
    #  be there for rgsproc-ed event lists, which is why I'm doing it this way rather than with a wildcard * at the
    #  end (which DAXA does support in the sas_call stage).
    evt_list_name = "P{o}{i}{ei}merged0000.FIT"

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

        # This method will fetch the valid data (ObsID, Instrument, and sub-exposure) that we need to process. When
        #  given R1 (for instance) as a search term only RGS1 identifiers will be returned. As this function needs to
        #  process RGS1 and RGS2 data, I just run it twice and add the results together
        # I prefer this to how I originally wrote this, as it saves multiple layers of for loops/if statements, which
        #  can be a little tricky to decipher
        # The loop of instruments is necessary because it is possible, if unlikely, that the user only selected
        #  one of the RGS instruments when setting up the mission
        rel_obs_info = []
        for inst in [i for i in miss.chosen_instruments if i[0] == 'R']:
            rel_obs_info += obs_archive.get_obs_to_process(miss.name, inst)

        # Don't just launch straight into the loop however, as the user can choose NOT to process unscheduled
        #  observations. In that case we clean that rel_obs_info list.
        if not process_unscheduled:
            # Select only those sub exposures that have an ident that doesn't start with a U
            rel_obs_info = [roi for roi in rel_obs_info if roi[2][0] != 'U']

        # Here we check that the previous required processes ran, mainly to be consistent. I know that odf ingest
        #  worked if we have rel_obs_info data, because odf_ingest is what populated the information get_obs_to_process
        #  uses for XMM.
        good_odf = obs_archive.check_dependence_success(miss.name, [[roi[0]] for roi in rel_obs_info], 'odf_ingest')

        # Now we start to cycle through the relevant data
        for obs_info in np.array(rel_obs_info)[good_odf]:
            # Unpack the observation information provided by the
            obs_id, inst, exp_id = obs_info

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Grab the path to the ODF directory, we shall need it
            odf_dir = miss.raw_data_path + obs_id + '/'

            # Set up a temporary directory to work in
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"

            # This is where the final output event list file will be stored
            final_path = dest_dir + evt_list_name.format(o=obs_id, i=inst, ei=exp_id)
            # But as I leave the temporary directory and files in place for now, because otherwise I'd just be
            #  moving them back for the next stage, I also define a temporary final path to check for the existence
            temp_final_path = temp_dir + evt_list_name.format(o=obs_id, i=inst, ei=exp_id)

            # If it doesn't already exist then we will create commands to generate it - there are no options for
            #  rgsproc that could be changed between runs (other than processing unscheduled, but we're looping
            #  through those commands separately), so it's safe to take what has already been generated.
            # We check to see if the process has been run (whether it was a success or failure) for the current
            #  data for the archive
            if ('rgs_events' not in obs_archive.process_success[miss.name] or
                    (obs_id + inst + exp_id) not in obs_archive.process_success[miss.name]['rgs_events']):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = rgp_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, i=inst, ei=inst+exp_id)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id + inst + exp_id] = cmd
                # We set the temporary final path here as that is what the checking stage looks at to verify stage
                #  success
                miss_final_paths[miss.name][obs_id + inst + exp_id] = temp_final_path
                miss_extras[miss.name][obs_id + inst + exp_id] = {'evt_list': final_path, 'temp_dir': temp_dir}

        # This is just used for populating a progress bar during generation
    process_message = 'Assembling RGS event lists'
    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def rgs_angles(obs_archive: Archive,  num_cores: int = NUM_CORES, disable_progress: bool = False,
               timeout: Quantity = None):
    """
    This function runs the second step of the SAS RGS processing pipeline, rgsproc. This should calculate aspect drift
    corrections for some 'uninformative' source, and should likely be refined later when these data are used to analyse
    a specific source. This happens separately for RGS1 and RGS2, and for each sub-exposure of the two instruments.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with RGS observations for
        which RGS processing should be run. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire rgs_events process, but a timeout for individual
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

    # Define the form of the rgsproc command that will be executed this function. This DAXA function, rgs_angles, only
    #  deals with the second stage of processing, hence why entrystage and finalstage are both two.
    # As we are effectively splitting up an existing pipeline, I actually leave the temporary directories (and final
    #  files) in place until later in the chain
    rgp_cmd = "cd {d}; export SAS_CCF={ccf}; export SAS_ODF={odf}; rgsproc entrystage=2:angles finalstage=2:angles " \
              "withinstexpids=true instexpids={ei}; "

    # mv *.FIT ../; cd ..; rm -r {d}

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

        # This method will fetch the valid data (ObsID, Instrument, and sub-exposure) that we need to process. When
        #  given R1 (for instance) as a search term only RGS1 identifiers will be returned. As this function needs to
        #  process RGS1 and RGS2 data, I just run it twice and add the results together
        # I prefer this to how I originally wrote this, as it saves multiple layers of for loops/if statements, which
        #  can be a little tricky to decipher
        # The loop of instruments is necessary because it is possible, if unlikely, that the user only selected
        #  one of the RGS instruments when setting up the mission
        rel_obs_info = []
        for inst in [i for i in miss.chosen_instruments if i[0] == 'R']:
            rel_obs_info += obs_archive.get_obs_to_process(miss.name, inst)

        # Here we check that the previous RGS processing step ran, if not then we don't use that particular RGS
        #  sub-exposure - I don't know how common failure of the last step is likely to be, but DAXA has the machinery
        #  to keep track of that stuff so we should use it!
        good_odf = obs_archive.check_dependence_success(miss.name, rel_obs_info, 'rgs_events')

        # Now we start to cycle through the relevant data
        for obs_info in np.array(rel_obs_info)[good_odf]:
            # Unpack the observation information provided by the
            obs_id, inst, exp_id = obs_info

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Grab the path to the ODF directory, we shall need it
            odf_dir = miss.raw_data_path + obs_id + '/'

            # We check to see if the process has been run (whether it was a success or failure) for the current
            #  data for the archive
            if ('rgs_angles' not in obs_archive.process_success[miss.name] or
                    (obs_id + inst + exp_id) not in obs_archive.process_success[miss.name]['rgs_angles']):
                # We don't need to set-up a temporary directory, as we use the one from the last step
                temp_dir = obs_archive.process_extra_info[miss.name]['rgs_events'][obs_id + inst + exp_id]['temp_dir']

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = rgp_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, i=inst, ei=inst + exp_id)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id + inst + exp_id] = cmd
                # There are no file outputs from this stage, it just modifies the existing event list
                miss_final_paths[miss.name][obs_id + inst + exp_id] = []
                miss_extras[miss.name][obs_id + inst + exp_id] = {}

        # This is just used for populating a progress bar during generation
    process_message = 'Correcting RGS for aspect drift'
    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def cleaned_rgs_event_lists(obs_archive: Archive,  num_cores: int = NUM_CORES, disable_progress: bool = False,
                            timeout: Quantity = None):
    """
    This function runs the third step of the SAS RGS processing pipeline, rgsproc. Here we filter the events to only
    those which should be useful for scientific analysis. The attitude and house-keeping GTIs are also applied. This
    happens separately for RGS1 and RGS2, and for each sub-exposure of the two instruments.

    Unfortunately it seems that combining sub-exposure event lists for a given ObsID-Instrument combo is not
    supported/recommended, combinations of data are generally done after spectrum generation, and even then they
    don't exactly recommend it - of course spectrum generation doesn't get done in DAXA. As such this function
    will produce individual event lists for RGS sub-exposures.

    :param Archive obs_archive: An Archive instance containing XMM mission instances with RGS observations for
        which RGS processing should be run. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire rgs_events process, but a timeout for individual
        ObsID-subexposure processes.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool, Quantity]
    """

    # TODO allow the application of soft proton filtering somehow

    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # Define the form of the rgsproc command that will be executed this function. This DAXA
    #  function, cleaned_rgs_event_lists, only deals with the first stage of processing, hence why entrystage
    #  and finalstage are both three.
    # As we are effectively splitting up an existing pipeline, I actually leave the temporary directories (and final
    #  files) in place until later in the chain
    rgp_cmd = "cd {d}; export SAS_CCF={ccf}; export SAS_ODF={odf}; rgsproc entrystage=3:filter finalstage=3:filter " \
              "withinstexpids=true instexpids={ei}; mv *EVENLI0000.FIT ../; cd ..; rm -r {d}; mv {oge} {fe}"

    # The event list name that we want to check for at the end of the process - a copy of the original event list
    #  but with the filtering of events applied - this is what is produced by the SAS call
    prod_evt_list_name = "P{o}{i}{ei}EVENLI0000.FIT"

    # These represent the final names and resting places of the event lists
    evt_list_name = "obsid{o}-inst{i}-subexp{se}-finalevents.fits"

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

        # This method will fetch the valid data (ObsID, Instrument, and sub-exposure) that we need to process. When
        #  given R1 (for instance) as a search term only RGS1 identifiers will be returned. As this function needs to
        #  process RGS1 and RGS2 data, I just run it twice and add the results together
        # I prefer this to how I originally wrote this, as it saves multiple layers of for loops/if statements, which
        #  can be a little tricky to decipher
        # The loop of instruments is necessary because it is possible, if unlikely, that the user only selected
        #  one of the RGS instruments when setting up the mission
        rel_obs_info = []
        for inst in [i for i in miss.chosen_instruments if i[0] == 'R']:
            rel_obs_info += obs_archive.get_obs_to_process(miss.name, inst)

        # Here we check that the previous required processes ran, mainly to be consistent. I know that odf ingest
        #  worked if we have rel_obs_info data, because odf_ingest is what populated the information get_obs_to_process
        #  uses for XMM.
        good_odf = obs_archive.check_dependence_success(miss.name, rel_obs_info, 'rgs_angles')

        # Now we start to cycle through the relevant data
        for obs_info in np.array(rel_obs_info)[good_odf]:
            # Unpack the observation information provided by the
            obs_id, inst, exp_id = obs_info

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Grab the path to the ODF directory, we shall need it
            odf_dir = miss.raw_data_path + obs_id + '/'

            # We don't need to set-up a temporary directory, as we use the one from the first step of RGS processing
            temp_dir = obs_archive.process_extra_info[miss.name]['rgs_events'][obs_id + inst + exp_id]['temp_dir']

            # This is where the final output event list file will be stored
            og_out_path = dest_dir + prod_evt_list_name.format(o=obs_id, i=inst, ei=exp_id)
            # This is where the final output event list file will be stored - after moving and renaming
            final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, se=exp_id, i=inst))

            # If it doesn't already exist then we will create commands to generate it - there are no options for
            #  rgsproc that could be changed between runs (other than processing unscheduled, but we're looping
            #  through those commands separately), so it's safe to take what has already been generated.
            # We check to see if the process has been run (whether it was a success or failure) for the current
            #  data for the archive
            if ('cleaned_rgs_event_lists' not in obs_archive.process_success[miss.name] or
                    (obs_id + inst + exp_id) not in obs_archive.process_success[miss.name]['cleaned_rgs_event_lists']):
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Format the blank command string defined near the top of this function with information
                #  particular to the current mission and ObsID
                cmd = rgp_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, i=inst, ei=inst + exp_id, fe=final_path,
                                     oge=og_out_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][obs_id + inst + exp_id] = cmd
                # There are no file outputs from this stage, it just modifies the existing event list
                miss_final_paths[miss.name][obs_id + inst + exp_id] = final_path
                miss_extras[miss.name][obs_id + inst + exp_id] = {'evt_clean_path': final_path}

        # This is just used for populating a progress bar during generation
    process_message = 'Cleaning RGS event lists'
    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def cleaned_evt_lists(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                      pn_filt_expr: Union[str, List[str]] = ("#XMMEA_EP", "(PATTERN <= 4)", "(FLAG .eq. 0)"),
                      mos_filt_expr: Union[str, List[str]] = ("#XMMEA_EM", "(PATTERN <= 12)", "(FLAG .eq. 0)"),
                      filt_mos_anom_state: bool = False, acc_mos_anom_states: Union[List[str], str] = ('G', 'I', 'U'),
                      num_cores: int = NUM_CORES, disable_progress: bool = False, timeout: Quantity = None):
    """
    This function is used to apply the soft-proton filtering (along with any other filtering you may desire, including
    the setting of energy limits) to XMM-Newton event lists, resulting in the creation of sets of cleaned event lists
    which are ready to be analysed (or merged together, if there are multiple exposures for a particular
    observation-instrument combination).

    :param Archive obs_archive: An Archive instance containing XMM mission instances for which cleaned event lists
        should be created. This function will fail if no XMM missions are present in the archive.
    :param Quantity lo_en: The lower bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'lo_en' is set to an Astropy Quantity, then 'hi_en' must be as well. Default is None, in which case no
        energy filter is applied.
    :param Quantity hi_en: The upper bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'hi_en' is set to an Astropy Quantity, then 'lo_en' must be as well. Default is None, in which case no
        energy filter is applied.
    :param str/List[str]/Tuple[str] pn_filt_expr: The filter expressions to be applied to EPIC-PN event lists. Either
        a single string expression can be passed, or a list/tuple of separate expressions, which will be combined
        using '&&' logic before being used as the expression for evselect. Other expression components can be
        added during the process of the function, such as GTI filtering and energy filtering.
    :param str/List[str]/Tuple[str] mos_filt_expr: The filter expressions to be applied to EPIC-MOS event lists. Either
        a single string expression can be passed, or a list/tuple of separate expressions, which will be combined
        using '&&' logic before being used as the expression for evselect. Other expression components can be
        added during the process of the function, such as GTI filtering, energy filtering, and anomalous state CCD
        filtering.
    :param bool filt_mos_anom_state: Whether this function should use the results of an 'emanom' run
        to identify and remove MOS CCDs that are in anomolous states. If 'False' is passed then no such filtering
        will be applied.
    :param List[str]/str acc_mos_anom_states: A list/tuple of acceptable MOS CCD status codes found by emanom
        (status- G is good at all energies, I is intermediate for E<1 keV, B is bad for E<1 keV, O is off, chip
        not in use, U is undetermined (low band counts <= 0)).
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire cleaned_evt_lists process, but a timeout for individual
        ObsID-Inst-subexposure processes.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool, Quantity]
    """

    # Have to make sure that the filter expressions are a list, as we want to append to them (if necessary), and then
    #  join them into a final filter string
    if isinstance(pn_filt_expr, str):
        pn_filt_expr = [pn_filt_expr]
    elif isinstance(pn_filt_expr, tuple):
        pn_filt_expr = list(pn_filt_expr)

    # Same deal here with the MOS filter expressions
    if isinstance(mos_filt_expr, str):
        mos_filt_expr = [mos_filt_expr]
    elif isinstance(mos_filt_expr, tuple):
        mos_filt_expr = list(mos_filt_expr)

    # Here we are making sure that the input energy limits are legal and sensible
    en_check = [en is not None for en in [lo_en, hi_en]]
    if not all(en_check) and any(en_check):
        raise ValueError("If one energy limit is set (e.g. 'lo_en') then the other energy limit must also be set.")
    elif (lo_en is not None and not lo_en.unit.is_equivalent('eV')) or \
            (hi_en is not None and not hi_en.unit.is_equivalent('eV')):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units "
                                  "that can be converted to eV.")
    # Obviously the upper limit can't be lower than the lower limit, or equal to it.
    elif hi_en is not None and lo_en is not None and hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")

    # Make sure we're converted to the right unit
    if all(en_check):
        lo_en = lo_en.to('eV').value
        hi_en = hi_en.to('eV').value
        pn_filt_expr.append("(PI in [{l}:{u}])".format(l=lo_en, u=hi_en))
        mos_filt_expr.append("(PI in [{l}:{u}])".format(l=lo_en, u=hi_en))
        # This is added into the filtered event list name, but only if energy limits are applied
        en_ident = '_{l}_{h}keV'.format(l=lo_en.value, h=hi_en.value)
    else:
        lo_en = ''
        hi_en = ''
        en_ident = ''

    # This command has two versions, one for M1 and M2, and another for PN which includes a second evselect command
    #  for the OOT events. I could make the OOT events a separate command, but that doesn't really work given the
    #  storage structure for recording processing success/logs. There is an echo in the PN command to give something
    #  to search for in the logs to see where the OOT processing begins
    ev_inst_cmd = {'mos': "cd {d}; export SAS_CCF={ccf}; evselect table={ae} filteredset={fe} expression={expr} "
                          "updateexposure=yes; cd ../; rm -r {d}",
                   'pn': "cd {d}; export SAS_CCF={ccf}; evselect table={ae} filteredset={fe} expression={expr} "
                         "updateexposure=yes; echo OOT EVSELECT; evselect table={ootae} "
                         "filteredset={ootfe} expression={expr} updateexposure=yes; cd ../; rm -r {d}"}

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

        # This method will fetch the valid data M1/2 and PN (ObsID, Instrument, and sub-exposure) that can be
        #  processed - then we can narrow it down to only those observations that had em/epchain run successfully
        # The loop of instruments is necessary because it is possible, if unlikely, that the user selected a subset
        #  of the PN, MOS1, or MOS2 instruments that this function can be used on. We select instruments beginning
        #  with M or P, not R (which would indicate RGS, which cannot be used with this function).
        rel_obs_info = []
        for inst in miss.chosen_instruments:
            if inst[0] == 'P':
                rel_p_obs = obs_archive.get_obs_to_process(miss.name, inst)
                # Have to ensure that there are actually some observations for this instrument - if there aren't then
                #  we'll skip over it
                if len(rel_p_obs) == 0:
                    continue
                # This checks that espfilt ran successfully for the PN data
                ef_pn_good = obs_archive.check_dependence_success(miss.name, rel_p_obs, 'espfilt',
                                                                  no_success_error=False)
                rel_obs_info.append(np.array(rel_p_obs)[ef_pn_good])

            elif inst[0] == 'M':
                rel_m_obs = obs_archive.get_obs_to_process(miss.name, inst)
                # Have to ensure that there are actually some observations for this instrument - if there aren't then
                #  we'll skip over it
                if len(rel_m_obs) == 0:
                    continue
                # This is why we're treating PN and MOS separately here, if the user wants to exclude certain CCD states
                #  then we have to make sure that emanom was run (and run successfully) for the MOS data.
                if filt_mos_anom_state:
                    ef_mos_good = obs_archive.check_dependence_success(miss.name, rel_m_obs, ['emanom', 'espfilt'],
                                                                       no_success_error=False)
                else:
                    ef_mos_good = obs_archive.check_dependence_success(miss.name, rel_m_obs, 'espfilt',
                                                                       no_success_error=False)
                rel_obs_info.append(np.array(rel_m_obs)[ef_mos_good])

        # We combine the obs information for PN and MOS, taking only those that we have confirmed have had successful
        #  emchain or epchain runs
        all_obs_info = np.vstack(rel_obs_info)

        # We check to see if any data remain in all_obs_info - normally check_dependence_success would raise an error
        #  if there weren't any, but as we're checking PN and MOS separately (and I want cleaned_evt_lists to run
        #  even if all data for PN or MOS hasn't made it this far) I passed no_success_error=False and instead check
        #  for absolute failure here
        if len(all_obs_info) == 0:
            raise NoDependencyProcessError("No observations have had successful espfilt runs, so cleaned_evt_lists "
                                           "cannot be run.")

        # We iterate through the valid identifying information which has had a successful espfilt (and possibly
        #  emanom, for MOS and if the user wants to exclude anomalous CCD states) run
        for obs_info in all_obs_info:
            # This is the valid id that allows us to retrieve the specific event list for this ObsID-M1/2-SubExp
            #  combination
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # Default value of this is None, so I don't have to set it for the two MOS cameras, only overwrite for PN
            oot_evt_list_file = None
            if inst == 'M1':
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                # Makes a copy of the MOS selection expression, as we might be adding to it during this
                #  part of the function
                cur_sel_expr = deepcopy(mos_filt_expr)
                # This chooses the correct command template
                ev_cmd = ev_inst_cmd['mos']
            elif inst == 'M2':
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                # Makes a copy of the MOS selection expression, as we might be adding to it during this
                #  part of the function
                cur_sel_expr = deepcopy(mos_filt_expr)
                # This chooses the correct command template
                ev_cmd = ev_inst_cmd['mos']
            elif inst == 'PN':
                evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['evt_list']
                # We do actually have an OOT file for PN, so we overwrite it
                oot_evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['oot_evt_list']
                # Make a copy of the PN selection expression to add to throughout this function
                cur_sel_expr = deepcopy(pn_filt_expr)
                # This chooses the correct command template, PN has an addition to process the OOT events as well
                ev_cmd = ev_inst_cmd['pn']
            else:
                raise ValueError("Somehow there is no instance of M1, M2, or PN in that storage key, this should be "
                                 "impossible!")

            # This is only triggered if the user WANTS to filter out anomalous states, and has actually run
            #  the emanom task - only MOS data with a successful emanom would have got to this point if the user
            #  does want to filter out the anomalous states
            if inst in ['M1', 'M2'] and filt_mos_anom_state is not False:
                log_path = obs_archive.process_extra_info[miss.name]['emanom'][val_id]['log_path']
                allow_ccds = [str(c_id) for c_id in parse_emanom_out(log_path, acceptable_states=acc_mos_anom_states)]
                ccd_expr = "CCDNR in {}".format(','.join(allow_ccds))
                # We add it to the list of selection expression components that we have been constructing
                cur_sel_expr.append(ccd_expr)

            # Read out where the GTIs created by espfilt live, and then create a filtering expression for the
            #  current mission-observation-instrument-subexposure (what a mouthful...)
            gti_path = obs_archive.process_extra_info[miss.name]['espfilt'][val_id]['gti_path']
            cur_sel_expr.append("GTI({}, TIME)".format(gti_path))

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
            temp_dir = dest_dir + temp_name + "/"

            # Setting up the path to the event file
            filt_evt_name = "obsid{o}-inst{i}-subexp{se}-en{en_id}-cleanevents.fits".format(i=inst, se=exp_id,
                                                                                            en_id=en_ident, o=obs_id)
            filt_evt_path = os.path.join(dest_dir, 'events', filt_evt_name)
            # And the same deal with the OOT event file, though of course that is only relevant to PN data
            filt_oot_evt_name = "obsid{o}-inst{i}-subexp{se}-en{en_id}-cleanootevents.fits".format(i=inst, se=exp_id,
                                                                                                   en_id=en_ident,
                                                                                                   o=obs_id)
            filt_oot_evt_path = os.path.join(dest_dir, 'events', filt_oot_evt_name)

            # The default final_paths declaration
            final_paths = [filt_evt_path]
            # The default extra information to store after the command has been construct
            to_store = {'evt_clean_path': filt_evt_path, 'en_key': en_ident}

            # We check to see if the process has been run (whether it was a success or failure) for the current
            #  data for the archive
            if ('cleaned_evt_lists' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['cleaned_evt_lists']):
                # As OOT events are only relevant to PN, we only add the variable to the paths-to-check if we're
                #  processing some PN data right now. The OOT events path also gets added to the extra information
                if inst == 'PN':
                    final_paths.append(filt_oot_evt_path)
                    to_store['oot_evt_clean_path'] = filt_oot_evt_path

                # If it doesn't already exist then we will create commands to generate it
                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                final_expression = "'" + " && ".join(cur_sel_expr) + "'"
                cmd = ev_cmd.format(d=temp_dir, ccf=ccf_path, ae=evt_list_file, fe=filt_evt_path, expr=final_expression,
                                    ootae=oot_evt_list_file, ootfe=filt_oot_evt_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                miss_final_paths[miss.name][val_id] = final_paths
                miss_extras[miss.name][val_id] = to_store

    # This is just used for populating a progress bar during the process run
    process_message = 'Generating cleaned PN/MOS event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@_last_process(ALLOWED_XMM_MISSIONS, 2)
@sas_call
def merge_subexposures(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                       timeout: Quantity = None):
    """
    A function to identify cases where an instrument for a particular XMM observation has multiple
    sub-exposures, for which the event lists can be merged. This produces a final event list, which is a
    combination of the sub-exposures.

    For those observation-instrument combinations with only a single
    exposure, this function will rename the cleaned event list so that the naming convention is comparable
    to the merged event list naming convention (i.e. sub-exposure identifier will be removed).

    :param Archive obs_archive: An Archive instance containing XMM mission instances for which cleaned event lists
        should be created. This function will fail if no XMM missions are present in the archive.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire merge_subexposures process, but a timeout for individual
        ObsID-Inst processes.
    :return: Information required by the SAS decorator that will run commands. Top level keys of any dictionaries are
        internal DAXA mission names, next level keys are ObsIDs. The return is a tuple containing a) a dictionary of
        bash commands, b) a dictionary of final output paths to check, c) a dictionary of extra info (in this case
        obs and analysis dates), d) a generation message for the progress bar, e) the number of cores allowed, and
        f) whether the progress bar should be hidden or not.
    :rtype: Tuple[dict, dict, dict, str, int, bool, Quantity]
    """

    # These commands get filled in by various stages of this function - in most of the other reduction wrapper
    #  functions there is only one of these template commands, but as each observation-instrument combo can have a
    #  different number of sub-exposures, and you can only merge two at once, the 'merge_cmd' part could be repeated
    #  multiple times, so it is separate.
    # The rename command is for those observation-instrument combos which DON'T have multiple sub-exposures to be
    #  merged  but instead will have their cleaned event list renamed to a filename consistent with the merged events.
    # I could have done this using a Python function (and did at first), but doing it this way means that there
    #  is an entry regarding this change in the log dictionaries.
    # The different instruments need different commands to deal with the fact that PN has OOT event lists as well
    inst_cmds = {'mos': {"merge": "merge set1={e_one} set2={e_two} outset={e_fin}",
                         "clean": "mv {ft} {fe}; cd ../ ; rm -r {d}",
                         "rename": "mv {cne} {nne}"},
                 'pn': {"merge": "merge set1={e_one} set2={e_two} outset={e_fin}; echo OOT MERGE; merge set1={oote_one}"
                                 " set2={oote_two} outset={oote_fin}",
                        "clean": "mv {ft} {fe}; mv {ootft} {ootfe}; cd ../ ; rm -r {d}",
                        "rename": "mv {cne} {nne}; mv {ootcne} {ootnne}"},
                 'setup': "cd {d}"}

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

        # Identifiers for all the valid data are fetched, and will be narrowed down later so that only those which
        #  had cleaned event lists generated successfully are selected
        rel_obs_info = obs_archive.get_obs_to_process(miss.name)

        # Here we check that cleaned_evt_lists ran - if it didn't then we aren't going to be merging anything
        good_ce = obs_archive.check_dependence_success(miss.name, rel_obs_info, 'cleaned_evt_lists')
        val_obs_info = np.array(rel_obs_info)[good_ce]

        # This dictionary will have top level keys of observation-instrument combinations, and with the values
        #  being lists of event lists that need to be combined
        to_combine = {}
        # We iterate through the valid observation info, so only data that has been processed all the way through
        #  to this stage is considered
        for obs_info in val_obs_info:
            # This sets up the observation ID, sub-exposure ID, and instrument
            obs_id, inst, exp_id = obs_info
            # Combine all that info into a single valid ID
            val_id = ''.join(obs_info)

            # The 'cleaned_evt_lists' function stores path info in the extra information dictionary, so we can
            #  just go there and grab the details about where the cleaned event list for this particular
            #  observation-instrument-exposure combination lives, as well as the energy range applied by the
            #  user in the cleaning/filtering step.
            filt_evt = obs_archive.process_extra_info[miss.name]['cleaned_evt_lists'][val_id]['evt_clean_path']
            en_key = obs_archive.process_extra_info[miss.name]['cleaned_evt_lists'][val_id]['en_key']

            # If the instrument is PN then we also need to know where the filtered out of time events live
            if inst == 'PN':
                filt_oot_evt = obs_archive.process_extra_info[miss.name]['cleaned_evt_lists'][val_id]['oot_evt_'
                                                                                                      'clean_path']
            else:
                filt_oot_evt = None

            # Combines just the observation and instrument into a top-level key for the dictionary that is used
            #  to identify which event lists needed to be added together
            oi_id = obs_id + '_' + inst
            # If there isn't already an entry then we make a new list, storing both the path to the filtered
            #  even list, and its energy range key
            if oi_id not in to_combine:
                # The OOT event list comprehension is slightly cheeky, but basically if it is set to None (for MOS)
                #  then we'll just be adding an empty list. It saves on more if statements with instrument checking
                to_combine[oi_id] = [[en_key, filt_evt] + [el for el in [filt_oot_evt] if el is not None]]
            # If there IS an entry, then we append the filtered event list path + energy key information
            else:
                # The OOT event list comprehension is slightly cheeky, but basically if it is set to None (for MOS)
                #  then we'll just be adding an empty list. It saves on more if statements with instrument checking
                to_combine[oi_id].append([en_key, filt_evt] + [el for el in [filt_oot_evt] if el is not None])

        # We've gone through all the observation-instrument-exposures that we have for the current mission and now
        #  we cycle through the ObsID-instrument combinations and start adding event lists together
        for oi in to_combine:
            # It seems very cyclical but ah well, we immediately split the storage key so we have the ObsID+instrument
            #  information back again
            obs_id, inst = oi.split('_')
            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Setting up the path to the final combined event file
            final_evt_name = "obsid{o}-inst{i}-subexpALL-en{en_id}-finalevents.fits".format(o=obs_id, i=inst,
                                                                                            en_id=to_combine[oi][0][0])
            final_path = os.path.join(dest_dir, 'events', final_evt_name)
            # And a possible accompanying OOT combined events file, not used if the instrument isn't PN but
            #  always defined because its easier
            final_oot_evt_name = ("obsid{o}-inst{i}-subexpALL-en{en_id}-finalootevents."
                                  "fits").format(o=obs_id, i=inst, en_id=to_combine[oi][0][0])
            final_oot_path = os.path.join(dest_dir, 'events', final_oot_evt_name)

            # We check if we've already run this for the current ObsID + instrument combo, as we don't need to do
            #  it again in that case - this has inverted logic compared to most of these, as we are checking if
            #  we want to break off this loop rather than actually execute the rest of it
            if ('merge_subexposures' in obs_archive.process_success[miss.name] and
                    (obs_id + inst) in obs_archive.process_success[miss.name]['merge_subexposures']):
                continue

            # If there is only one event list for a particular ObsID-instrument combination, then obviously merging
            #  is impossible/unnecessary, so in that case we just rename the file (which will have sub-exposure ID
            #  info in the name) to the same style of the merged files
            if len(to_combine[oi]) == 1 and inst == 'PN':
                # In this case we make sure to move the OOT of time event list file as well, using the PN skew
                #  of the rename command
                cmd = inst_cmds['pn']['rename'].format(cne=to_combine[oi][0][1], nne=final_path,
                                                       ootcne=to_combine[oi][0][2], ootnne=final_oot_path)
            elif len(to_combine[oi]) == 1:
                cmd = inst_cmds['mos']['rename'].format(cne=to_combine[oi][0][1], nne=final_path)
            else:

                # Set up a temporary directory to work in (probably not really necessary in this case, but will be
                #  in other processing functions).
                temp_name = "tempdir_{}".format(randint(0, int(1e+8)))
                temp_dir = dest_dir + temp_name + "/"

                # As the merge command won't overwrite an existing file name, and we don't know how many times the loop
                #  below will iterate, we create temporary file names based on the iteration number of the loop
                temp_evt_name = "{i}{en_id}_clean_temp{ind}.fits"
                # We also make one for OOT events, though it will only be used when merging PN events
                temp_oot_evt_name = "{i}{en_id}_oot_clean_temp{ind}.fits"

                # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # If we've got to this point then merging will definitely occur, so we start with the setup command,
                #  which just moves us to the working directory - it's in a list because said list will contain all
                #  the stages of this command, and will be joined at the end of this process into a single bash
                #  command.
                cur_merge_cmds = [inst_cmds['setup'].format(d=temp_dir)]
                # Now we can iterate through the files for merging - using enumerate so we get an index for the current
                #  event path, which we can add one too to retrieve the next event list along - i.e. what we will be
                #  merging into. This is why we slice the event file list so that we only iterate up to the penultimate
                #  file, because that file will be accessed by adding one to the last evt_ind.
                # Frankly I probably should have used a while loop here, but ah well
                for evt_ind, evt_path in enumerate(to_combine[oi][:-1]):
                    # If we haven't iterated yet then we use the current event list name as the
                    #  first event list.
                    if evt_ind == 0 and len(evt_path) == 2:
                        first_evt = evt_path[1]
                    # There will be three entries in evt_path if the current instrument is PN, and I thought this
                    #  was easier to read versus nesting another if statement in the one above
                    elif evt_ind == 0 and len(evt_path) == 3:
                        first_evt = evt_path[1]
                        first_oot_evt = evt_path[2]
                    # However if we HAVE iterated before, then the first event list should actually be the output of the
                    #  last merging step, not the CURRENT value of evt_path (as that has already been added into the
                    #  merged list).
                    elif len(evt_path) == 2:
                        # This is a bit cheeky, but this will never be used before its defined - it will always use the
                        #  value defined in the last iteration around
                        first_evt = cur_t_name
                    # This is the same as above, but again accounting for the fact that for PN we need to be merging
                    #  OOT events as well
                    else:
                        first_evt = cur_t_name
                        first_oot_evt = cur_oot_t_name

                    # The output of the merge has to be given a temporary name, as the merge command won't allow it to
                    #  have the same name as an existing file
                    cur_t_name = temp_evt_name.format(i=inst, en_id=to_combine[oi][0][0], ind=evt_ind)
                    if inst == 'PN':
                        cur_oot_t_name = temp_oot_evt_name.format(i=inst, en_id=to_combine[oi][0][0], ind=evt_ind)
                        # This populated the command with the event list paths and output path (note where we add
                        #  1 to the evt_ind value). Also includes the OOT paths
                        cur_cmd = inst_cmds['pn']['merge'].format(e_one=first_evt,
                                                                  e_two=to_combine[oi][evt_ind + 1][1],
                                                                  e_fin=cur_t_name,
                                                                  oote_one=first_oot_evt,
                                                                  oote_two=to_combine[oi][evt_ind + 1][2],
                                                                  oote_fin=cur_oot_t_name)
                    else:
                        # This populated the command with the event list paths and output path (note where we add
                        #  1 to the evt_ind value).
                        cur_cmd = inst_cmds['mos']['merge'].format(e_one=first_evt,
                                                                   e_two=to_combine[oi][evt_ind+1][1],
                                                                   e_fin=cur_t_name)
                    # Then the command is added to the command list
                    cur_merge_cmds.append(cur_cmd)

                # The final command added to the cmd list is a cleanup step, removing the temporary working directory
                #  (and all the transient part merged event lists that might have been created along the way).
                # Again have to account for PN having OOT event lists as well
                if inst == 'PN':
                    cur_merge_cmds.append(inst_cmds['pn']['clean'].format(ft=cur_t_name, fe=final_path,
                                                                          ootft=cur_oot_t_name,
                                                                          ootfe=final_oot_path,
                                                                          d=temp_dir))
                else:
                    cur_merge_cmds.append(inst_cmds['mos']['clean'].format(ft=cur_t_name, fe=final_path,
                                                                           d=temp_dir))
                # Finally the list of commands is all joined together so it is one, like the commands of the rest
                #  of the SAS wrapper functions
                cmd = '; '.join(cur_merge_cmds)

            # Now store the bash command, the path, and extra info in the dictionaries
            miss_cmds[miss.name][obs_id+inst] = cmd
            miss_final_paths[miss.name][obs_id+inst] = final_path
            # Again accounting for whether a OOT merged event list has been produced here or not
            if inst == 'PN':
                miss_extras[miss.name][obs_id+inst] = {'final_evt': final_path, 'final_oot_evt': final_oot_path}
            else:
                miss_extras[miss.name][obs_id+inst] = {'final_evt': final_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Generating final PN/MOS event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout
