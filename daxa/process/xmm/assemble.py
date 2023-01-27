#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 27/01/2023, 16:33. Copyright (c) The Contributors
import os
from copy import deepcopy
from random import randint
from typing import Union, List, Tuple
from warnings import warn

from astropy.io import fits
from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive.base import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.xmm._common import _sas_process_setup, sas_call, ALLOWED_XMM_MISSIONS
from daxa.process.xmm.check import parse_emanom_out


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
    :rtype: Tuple[dict, dict, dict, str, int, bool]
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
             "mv *ATTTSR*.FIT ../;cd ..; rm -r {d}"

    # The event list pattern that we want to check for at the end of the process
    evt_list_name = "P{o}PN{eid}PIEVLI0000.FIT"
    oot_evt_list_name = "P{o}PN{eid}OOEVLI0000.FIT"

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

                    # Clean out the list of exposures here rather than add another layer of nesting to the
                    #  for loop below - this removes unscheduled exposures if the user has decided that they don't
                    #  want to process them.
                    if not process_unscheduled:
                        pn_exp = [pe for pe in pn_exp if 'U' not in pe]

                    for exp_id in pn_exp:
                        # TODO Again update this after SAS summary parser (issue 34), because we currently try to
                        #  process everything as imaging mode (see issue #40)

                        # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
                        #  files will be written to.
                        dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
                        ccf_path = dest_dir + 'ccf.cif'

                        # The idea here is to figure out which CCDs where active for a particular observation - this
                        #  is to avoid errors thrown by epchain about missing CCD data. Those errors are non-fatal,
                        #  but they contaminate the stderr output which is parsed by DAXA, and cause us to consider
                        #  processing small-window mode data a failure when it isn't, as only one CCD is turned on
                        search_term = '{o}_{i}{e}'.format(o=obs_id, i=inst, e=exp_id)
                        # TODO Again update this after SAS summary parser (issue 34) is implemented
                        # Lists the ODFs which match the search term above, finding separate imaging mode files
                        #  for each CCD of the current sub-exposure
                        ccd_files = [f for f in os.listdir(odf_dir) if search_term in f and 'IME' in f]

                        # As an aside, at this point we can check to see if there are any files in this list, as
                        #  if there are none then we can decide that this sub-exposure is not in an imaging data mode
                        #  and as such we won't process it
                        # TODO This will have to change when I start processing non-imaging data modes
                        if len(ccd_files) == 0:
                            continue

                        # Now the files are processed to just retrieve the list of CCDs active for this sub-exposure
                        #  of the current observation
                        ccd_ids = sorted([int(f.split(search_term)[-1].split('IME')[0]) for f in ccd_files])
                        # The stupid conversion from str to int to str again is an inelegant way of removing leading
                        #  zeros from the CCD ID numbers in their string form - also need to sort whilst they
                        #  are in integer form
                        ccd_ids = [str(c_id) for c_id in ccd_ids]
                        ccd_str = ",".join(ccd_ids)

                        # Set up a temporary directory to work in
                        temp_name = "tempdir_{}".format(randint(0, 1e+8))
                        temp_dir = dest_dir + temp_name + "/"
                        # This is where the final output event list file will be stored
                        final_path = dest_dir + evt_list_name.format(o=obs_id, eid=exp_id)
                        oot_final_path = dest_dir + oot_evt_list_name.format(o=obs_id, eid=exp_id)

                        # If it doesn't already exist then we will create commands to generate it
                        # TODO Decide whether this is the route I really want to follow for this (see issue #28)
                        if not os.path.exists(final_path):
                            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)

                            # Format the blank command string defined near the top of this function with information
                            #  particular to the current mission and ObsID
                            cmd = ep_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, e=exp_id[1:], s=exp_id[0],
                                                c=ccd_str)

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
    :rtype: Tuple[dict, dict, dict, str, int, bool]
    """
    # Run the setup for SAS processes, which checks that SAS is installed, checks that the archive has at least
    #  one XMM mission in it, and shows a warning if the XMM missions have already been processed
    sas_version = _sas_process_setup(obs_archive)

    # Define the form of the emchain command that must be run to create a combined MOS1/2 event list - the exposures
    #  argument will only ever be set with one exposure at a time. emchain does loop through sub-exposures
    #  automatically, but I'm attempting to normalise the behaviours between emchain and epchain in how DAXA calls
    #  them. Issue #42 discusses this.
    em_cmd = "cd {d}; export SAS_CCF={ccf}; emchain odf={odf} instruments={i} exposures={ei}; mv *MIEVLI*.FIT ../; " \
             "mv *ATTTSR*.FIT ../; cd ..; rm -r {d}"

    # The event list name that we want to check for at the end of the process - the zeros at the end seem to always
    #  be there for emchain-ed event lists, which is why I'm doing it this way rather than with a wildcard * at the
    #  end (which DAXA does support in the sas_call stage).
    evt_list_name = "P{o}{i}{ei}MIEVLI0000.FIT"

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

                    # Grab the path to the ODF directory, we shall need it
                    odf_dir = miss.raw_data_path + obs_id + '/'

                    # List available sub-exposures for the current ObsID-instrument combo
                    m_exp = list(set([f.split(obs_id)[1].split(inst)[1][:4] for f in os.listdir(odf_dir)
                                      if '{i}S'.format(i=inst) in f or '{i}U'.format(i=inst) in f]))

                    # Clean out the list of exposures here rather than add another layer of nesting to the
                    #  for loop below - this removes unscheduled exposures if the user has decided that they don't
                    #  want to process them.
                    if not process_unscheduled:
                        m_exp = [me for me in m_exp if 'U' not in me]

                    for exp_id in m_exp:
                        # Set up a temporary directory to work in
                        temp_name = "tempdir_{}".format(randint(0, 1e+8))
                        temp_dir = dest_dir + temp_name + "/"

                        # This is where the final output event list file will be stored
                        final_path = dest_dir + evt_list_name.format(o=obs_id, i=inst, ei=exp_id)

                        # If it doesn't already exist then we will create commands to generate it
                        # TODO Decide whether this is the route I really want to follow for this (see issue #28)
                        if not os.path.exists(final_path):
                            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
                            if not os.path.exists(temp_dir):
                                os.makedirs(temp_dir)

                            # Format the blank command string defined near the top of this function with information
                            #  particular to the current mission and ObsID
                            cmd = em_cmd.format(d=temp_dir, odf=odf_dir, ccf=ccf_path, i=inst, ei=exp_id)

                            # Now store the bash command, the path, and extra info in the dictionaries
                            miss_cmds[miss.name][obs_id+inst+exp_id] = cmd
                            miss_final_paths[miss.name][obs_id+inst+exp_id] = final_path
                            miss_extras[miss.name][obs_id+inst+exp_id] = {'evt_list': final_path}

    # This is just used for populating a progress bar during generation
    process_message = 'Assembling MOS event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def cleaned_evt_lists(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                      pn_filt_expr: Union[str, List[str]] = ("#XMMEA_EP", "(PATTERN <= 4)", "(FLAG .eq. 0)"),
                      mos_filt_expr: Union[str, List[str]] = ("#XMMEA_EM", "(PATTERN <= 12)", "(FLAG .eq. 0)"),
                      filt_mos_anom_state: Union[List[str], str, bool] = ('G', 'I', 'U'), num_cores: int = NUM_CORES,
                      disable_progress: bool = False, timeout: Quantity = None):
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
        filtering..
    :param List[str]/str/bool filt_mos_anom_state: Whether this function should use the results of an 'emanom' run
        to identify and remove MOS CCDs that are in anomolous states. If 'False' is passed then no such filtering
        will be applied, with the same behaviour occuring if emanom has not been run on the passed archive. Otherwise
        a list/tuple of acceptable status codes can be passed (status- G is good at all energies, I is intermediate
        for E<1 keV, B is bad for E<1 keV, O is off, chip not in use, U is undetermined (low band counts <= 0)).
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
    :rtype: Tuple[dict, dict, dict, str, int, bool]
    """
    #
    if isinstance(pn_filt_expr, str):
        pn_filt_expr = [pn_filt_expr]
    elif isinstance(pn_filt_expr, tuple):
        pn_filt_expr = list(pn_filt_expr)

    if isinstance(mos_filt_expr, str):
        mos_filt_expr = [mos_filt_expr]
    elif isinstance(mos_filt_expr, tuple):
        mos_filt_expr = list(mos_filt_expr)

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

    ev_cmd = "cd {d}; export SAS_CCF={ccf}; evselect table={ae} filteredset={fe} expression={expr} " \
             "updateexposure=yes; mv {fe} ../; cd ../; rm -r {d}"

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

        # TODO Generalise this for god's sake, it shouldn't need to be repeated in any form
        # Need to check to see whether ANY of ObsID-instrument-subexposure combos have had emchain run for them, as
        #  it is a requirement for this processing function. There will probably be a more elegant way of checking
        #  at some point in the future, generalised across all SAS functions
        if 'emchain' not in obs_archive.process_success[miss.name] \
                and 'epchain' not in obs_archive.process_success[miss.name]:
            raise NoDependencyProcessError("Neither the emchain (for MOS) nor the epchain (for PN) step has been run "
                                           "for the {m} mission in the {a} archive. At least one of these must have "
                                           "been run to use cleaned_evt_lists.".format(m=miss.name,
                                                                                       a=obs_archive.archive_name))
        # If every emchain run was a failure then we warn the user and move onto the next XMM mission (if there
        #  is one).
        elif ('emchain' in obs_archive.process_success[miss.name] and all(
                [v is False for v in obs_archive.process_success[miss.name]['emchain'].values()])) and (
                'epchain' in obs_archive.process_success[miss.name] and all(
            [v is False for v in obs_archive.process_success[miss.name]['epchain'].values()])):

            warn("Every emchain and epchain run for the {m} mission in the {a} archive is reporting as a "
                 "failure, skipping process.".format(m=miss.name, a=obs_archive.archive_name), stacklevel=2)
            continue
        else:
            # This fetches those IDs for which emchain has reported success, and these are what we will iterate
            #  through to ensure that we only act upon data that is in a final event list form.
            valid_ids = [k for k, v in obs_archive.process_success[miss.name]['emchain'].items() if v] + \
                        [k for k, v in obs_archive.process_success[miss.name]['epchain'].items() if v]

        # We iterate through the valid IDs rather than nest ObsID and instrument for loops
        for val_id in valid_ids:
            # TODO Review this if I change the IDing system as I was pondering in issue #44
            if 'M1' in val_id:
                obs_id, exp_id = val_id.split('M1')
                inst = 'M1'
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                oot_evt_list_file = None
                # Makes a copy of the MOS selection expression, as we might be adding to it during this
                #  part of the function
                cur_sel_expr = deepcopy(mos_filt_expr)
            elif 'M2' in val_id:
                obs_id, exp_id = val_id.split('M2')
                inst = 'M2'
                evt_list_file = obs_archive.process_extra_info[miss.name]['emchain'][val_id]['evt_list']
                oot_evt_list_file = None
                # Makes a copy of the MOS selection expression, as we might be adding to it during this
                #  part of the function
                cur_sel_expr = deepcopy(mos_filt_expr)
            elif 'PN' in val_id:
                obs_id, exp_id = val_id.split('PN')
                inst = 'PN'
                evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['evt_list']
                oot_evt_list_file = obs_archive.process_extra_info[miss.name]['epchain'][val_id]['oot_evt_list']
                cur_sel_expr = deepcopy(pn_filt_expr)
            else:
                raise ValueError("Somehow there is no instance of M1, M2, or PN in that storage key, this should be "
                                 "impossible!")

            # This should read in the header so that we can grab filter information from it
            evt_hdr = fits.getheader(evt_list_file)
            # If the filter is either CalClosed or Closed then we do not care to process it any further.
            # TODO Consider changing this if I add the SAS summary file parser, and use it upstream
            if evt_hdr['FILTER'] in ['CalClosed', 'Closed']:
                continue

            if inst in ['M1', 'M2'] and val_id in obs_archive.process_extra_info[miss.name]['emanom'] \
                    and filt_mos_anom_state is not False:
                log_path = obs_archive.process_extra_info[miss.name]['emanom'][val_id]['log_path']
                allow_ccds = [str(c_id) for c_id in parse_emanom_out(log_path, acceptable_states=filt_mos_anom_state)]
                ccd_expr = "CCDNR in {}".format(','.join(allow_ccds))
                # We add it to the list of selection expression components that we have been constructing
                cur_sel_expr.append(ccd_expr)

            # Read out where the GTIs created by espfilt live, and then create a filtering expression for the
            #  current mission-observation-instrument-subexposure (what a mouthful...)
            gti_path = obs_archive.process_extra_info[miss.name]['espfilt'][val_id]['gti_path']
            cur_sel_expr.append("GTI({}, TIME)".format(gti_path))

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.get_processed_data_path(miss, obs_id)
            ccf_path = dest_dir + 'ccf.cif'

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            # Setting up the path to the event file
            filt_evt_name = "{i}{exp_id}{en_id}_clean.fits".format(i=inst, exp_id=exp_id, en_id=en_ident)
            filt_evt_path = dest_dir + filt_evt_name
            final_paths = [filt_evt_path]

            # If it doesn't already exist then we will create commands to generate it
            # TODO Need to decide which file to check for here to see whether the command has already been run
            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            final_expression = "'" + " && ".join(cur_sel_expr) + "'"
            cmd = ev_cmd.format(d=temp_dir, ccf=ccf_path, ae=evt_list_file, fe=filt_evt_path, expr=final_expression)

            # Now store the bash command, the path, and extra info in the dictionaries
            miss_cmds[miss.name][val_id] = cmd
            miss_final_paths[miss.name][val_id] = final_paths
            miss_extras[miss.name][val_id] = {'evt_clean_path': filt_evt_path, 'en_key': en_ident}

    # This is just used for populating a progress bar during the process run
    process_message = 'Generating cleaned PN/MOS event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@sas_call
def merge_subexposures(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                       timeout: Quantity = None):
    """
    A function to identify cases where an instrument for a particular XMM observation has multiple
    sub-exposures, for which the event lists can be merged. This produces a final event list, which is a
    combination of the sub-exposures. For those observation-instrument combinations with only a single
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
    :rtype: Tuple[dict, dict, dict, str, int, bool]
    """

    # These commands get filled in by various stages of this function - in most of the other reduction wrapper
    #  functions there is only one of these template commands, but as each observation-instrument combo can have a
    #  different number of sub-exposures, and you can only merge two at once, the 'merge_cmd' part could be repeated
    #  multiple times, so it is separate.
    setup_cmd = "cd {d}"
    merge_cmd = "merge set1={e_one} set2={e_two} outset={e_fin}"
    cleanup_cmd = "mv {ft} ../{fe}; cd ../"  # ; rm -r {d}

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

        # TODO Generalise this for god's sake, it shouldn't need to be repeated in any form
        # Need to check to see whether ANY of ObsID-instrument-subexposure combos have had cleaned_evt_lists run for
        #  them, as it is a requirement for this processing function.
        if 'cleaned_evt_lists' not in obs_archive.process_success[miss.name]:
            raise NoDependencyProcessError("The cleaned_evt_lists step has not been run for the {m} mission in the "
                                           "{a} archive. Filtered event lists must be available for sub-exposures to "
                                           "be merged.".format(m=miss.name, a=obs_archive.archive_name))
        # If every cleaned_evt_lists run was a failure then we warn the user and move onto the next
        #  XMM mission (if there is one).
        elif 'cleaned_evt_lists' in obs_archive.process_success[miss.name] and \
                all([v is False for v in obs_archive.process_success[miss.name]['cleaned_evt_lists'].values()]):
            warn("Every cleaned_evt_lists run for the {m} mission in the {a} archive is reporting as a "
                 "failure, skipping process.".format(m=miss.name, a=obs_archive.archive_name), stacklevel=2)
            continue
        else:
            # This fetches those IDs for which cleaned_evt_lists has reported success, and these are what we will
            #  iterate through to ensure that we only act upon data that is in a final event list form.
            valid_ids = [k for k, v in obs_archive.process_success[miss.name]['cleaned_evt_lists'].items() if v]

        # This dictionary will have top level keys of observation-instrument combinations, and with the values
        #  being lists of event lists that need to be combined
        to_combine = {}
        # We iterate through the valid IDs rather than nest ObsID and instrument for loops
        for val_id in valid_ids:
            # This sets up the observation ID, sub-exposure ID, and instrument
            # TODO Review this if I change the IDing system as I was pondering in issue #44
            if 'M1' in val_id:
                obs_id, exp_id = val_id.split('M1')
                inst = 'M1'
            elif 'M2' in val_id:
                obs_id, exp_id = val_id.split('M2')
                inst = 'M2'
            elif 'PN' in val_id:
                obs_id, exp_id = val_id.split('PN')
                inst = 'PN'
                # filt_oot_evt = obs_archive.process_extra_info[miss.name]['cleaned_evt_lists'][val_id]['oot_evt_list']
            else:
                raise ValueError("Somehow there is no instance of M1, M2, or PN in that storage key, this should be "
                                 "impossible!")

            # The 'cleaned_evt_lists' function stores path info in the extra information dictionary, so we can
            #  just go there and grab the details about where the cleaned event list for this particular
            #  observation-instrument-exposure combination lives, as well as the energy range applied by the
            #  user in the cleaning/filtering step.
            filt_evt = obs_archive.process_extra_info[miss.name]['cleaned_evt_lists'][val_id]['evt_clean_path']
            en_key = obs_archive.process_extra_info[miss.name]['cleaned_evt_lists'][val_id]['en_key']

            # Combines just the observation and instrument into a top-level key for the dictionary that is used
            #  to identify which event lists needed to be added together
            oi_id = obs_id + '_' + inst
            # If there isn't already an entry then we make a new list, storing both the path to the filtered
            #  even list, and its energy range key
            if oi_id not in to_combine:
                to_combine[oi_id] = [[filt_evt, en_key]]
            # If there IS an entry, then we append the filtered event list path + energy key information
            else:
                to_combine[oi_id].append([filt_evt, en_key])

        # We've gone through all the observation-instrument-exposures that we have for the current mission and now
        #  we cycle through the ObsID-instrument combinations and start adding event lists together
        for oi in to_combine:
            # It seems very cyclical but ah well, we immediately split the storage key so we have the ObsID+instrument
            #  information back again
            obs_id, inst = oi.split('_')
            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.get_processed_data_path(miss, obs_id)

            # Setting up the path to the final combined event file
            final_evt_name = "{i}{en_id}_clean.fits".format(i=inst, en_id=to_combine[oi][0][1])
            final_path = dest_dir + final_evt_name

            # If there is only one event list for a particular ObsID-instrument combination, then obviously merging
            #  is impossible/unnecessary, so in that case we just rename the file (which will have sub-exposure ID
            #  info in the name) to the same style of the merged files
            if len(to_combine[oi]) == 1:
                os.rename(to_combine[oi][0][0], final_path)
                continue
            elif os.path.exists(final_path):
                continue

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            temp_name = "tempdir_{}".format(randint(0, 1e+8))
            temp_dir = dest_dir + temp_name + "/"

            # As the merge command won't overwrite an existing file name, and we don't know how many times the loop
            #  below will iterate, we create temporary file names based on the iteration number of the loop
            temp_evt_name = "{i}{en_id}_clean_temp{ind}.fits"

            # If it doesn't already exist then we will create commands to generate it
            # TODO Need to decide which file to check for here to see whether the command has already been run
            # Make the temporary directory (it shouldn't already exist but doing this to be safe)
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # If we've got to this point then merging will definitely occur, so we start with the setup command,
            #  which just moves us to the working directory - its in a list because said list will contain all
            #  the stages of this command, and will be joined at the end of this process into a single bash
            #  command.
            cur_merge_cmds = [setup_cmd.format(d=temp_dir)]
            # Now we can iterate through the files for merging - using enumerate so we get an index for the current
            #  event path, which we can add one to to retrieve the next event list along - i.e. what we will be
            #  merging into. This is why we slice the event file list so that we only iterate up to the penultimate
            #  file, because that file will be accessed by adding one to the last evt_ind.
            # Frankly I probably should have used a while loop here, but ah well
            for evt_ind, evt_path in enumerate(to_combine[oi][:-1]):

                # If we haven't iterated yet then we use the currently access event list name as the first event list.
                if evt_ind == 0:
                    first_evt = evt_path[0]
                # However if we HAVE iterated before, then the first event list should actually be the output of the
                #  last merging step, not the CURRENT value of evt_path (as that has already been added into the
                #  merged list).
                else:
                    # This is a bit cheeky, but this will never be used before its defined - it will always use the
                    #  value defined in the last iteration around
                    first_evt = cur_t_name
                # The output of the merge has to be given a temporary name, as the merge command won't allow it to
                #  have the same name as an existing file
                cur_t_name = temp_evt_name.format(i=inst, en_id=to_combine[oi][0][1], ind=evt_ind)
                # This populated the command with the event list paths and output path (note where we add 1 to the
                #  evt_ind value).
                cur_cmd = merge_cmd.format(e_one=first_evt, e_two=to_combine[oi][evt_ind+1][0], e_fin=cur_t_name)
                # Then the command is added to the command list
                cur_merge_cmds.append(cur_cmd)

            # The final command added to the cmd list is a cleanup step, removing the temporary working directory
            #  (and all the transient part merged event lists that might have been created along the way).
            cur_merge_cmds.append(cleanup_cmd.format(ft=cur_t_name, fe=final_evt_name, d=temp_dir))
            # Finally the list of commands is all joined together so it is one, like the commands of the rest of the
            #  SAS wrapper functions
            cmd = '; '.join(cur_merge_cmds)

            # # Now store the bash command, the path, and extra info in the dictionaries
            miss_cmds[miss.name][obs_id+inst] = cmd
            miss_final_paths[miss.name][obs_id+inst] = final_path
            miss_extras[miss.name][obs_id+inst] = {'final_evt': final_path}

        # This is just used for populating a progress bar during the process run
        process_message = 'Generating final PN/MOS event lists'

        return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout
