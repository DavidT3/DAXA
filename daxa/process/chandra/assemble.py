#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 11/11/2024, 23:26. Copyright (c) The Contributors

import os
from random import randint
from warnings import warn

import numpy as np
from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.chandra._common import _ciao_process_setup, ciao_call

# All the ASCA system grade IDs that can be chosen
ASCA_SYSTEM_GRADES = [0, 2, 3, 4, 6, 1, 5, 7]
# The default grades selected by chandra_repro - need to show a warning if they select any that aren't here
REPRO_GRADES = [0, 2, 3, 4, 6]


@ciao_call
def chandra_repro(obs_archive: Archive, destreak: bool = True, check_very_faint: bool = False, pix_adj: str = 'default',
                  asol_update: bool = True, grating_pi_filter: bool = True, num_cores: int = NUM_CORES,
                  disable_progress: bool = False, timeout: Quantity = None):
    """
    The DAXA implementation of the CIAO ('chandra_repro'; https://cxc.cfa.harvard.edu/ciao/ahelp/chandra_repro.html)
    tool, which takes downloaded Chandra observations and re-reduces them, with the latest calibrations applied. It
    is the first stage of most Chandra analyses, and we decided to implement a wrapper rather than re-creating the
    whole process from the ground up. We provide most of the same configurations as the CIAO command-line tool.

    This process requires that the 'prepare_chandra_info' function has been run on the observation archive, as without
    it, we cannot know what data there are to process.

    :param Archive obs_archive: An Archive instance containing a Chandra mission instance. This function will fail
        if no Chandra missions are present in the archive.
    :param bool destreak: Controls whether a 'destreaking' technique is applied to ACIS data, to account for a flaw
        in the readout of one of the CCDs. Default is True, in which case streak events are filtered out of the
        events list.
    :param bool check_very_faint: Sets whether the ACIS particle background for 'very faint mode' observations is
        cleaned, default is False. Setting to True can lead to good events being removed in observations with
        modestly bright point sources.
    :param str pix_adj: Controls the pixel randomization applied to ACIS data to avoid spatial aliasing effects. The
        default is 'default' but 'edser', 'none', 'randomize' may also be passed - read more on the CIAO documentation
        website (https://cxc.harvard.edu/ciao/why/acispixrand.html).
    :param bool asol_update: If True (the default) then a boresight correction will be applied to the observation to
        update the location of the aimpoint.
    :param bool grating_pi_filter: This parameter controls whether an optional grating filter should be used, default
        is True - the filter is meant to suppress the background.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-instrument processes.
    """
    # Runs standard checks, makes directories, returns CIAO versions, etc.
    ciao_vers, caldb_vers, chan_miss = _ciao_process_setup(obs_archive)

    # We're not going to use the handy CIAO Python-wrapped version of 'chandra_repro' - just because all the
    #  infrastructure of DAXA is set up to wrap cmd-line tools itself, and this way we don't rely on there being
    #  a correctly-installed-to-Python version of CIAO (even though that should be quite simple). We are
    #  going to use the chandra_repro command to process Chandra data though, as it works so well, so we
    #  create the command template.
    # Note that we don't need to make a local copy of PFILES here, because that will be added in when the command
    #  is run
    crp_cmd = ("cd {d}; chandra_repro indir={in_f} outdir={out_f} root={rn} badpixel='yes' process_events='yes' "
               "destreak={ds} set_ardlib='no' check_vf_pha={cvf} pix_adj={pa} tg_zo_position='evt2' "
               "asol_update={as_up} pi_filter={pf} cleanup='no' verbose=5; mv {oge} {fe}; mv {oggti} {fgti}; "
               "mv {ogbp} {fbp}; mv {ogfov} {ffov}; ")

    # HRC has one slight addition to the command, so that re retrieve the dead time correction file produced for
    #  those instruments
    hrc_crp_cmd = ("cd {d}; chandra_repro indir={in_f} outdir={out_f} root={rn} badpixel='yes' process_events='yes' "
                   "destreak={ds} set_ardlib='no' check_vf_pha={cvf} pix_adj={pa} tg_zo_position='evt2' "
                   "asol_update={as_up} pi_filter={pf} cleanup='no' verbose=5; mv {oge} {fe}; mv {oggti} {fgti}; "
                   "mv {ogbp} {fbp}; mv {ogfov} {ffov}; mv {ogdtf} {fdtf}; ")

    # The aspect solution file problem requires a little bash if-else
    asol_mv = ("if ls {rpasol} 1> /dev/null 2>&1; then mv {rpasol} {fasol}; else mv {altasol} {fasol}; fi; "
               "cd ..; rm -r {d}")
    # asol_mv = ("if ls {rpasol} 3>&1 4>&2 1>/dev/null 2>&1; then mv {rpasol} {fasol} 1>&3 2>&4; else mv {altasol} "
    #            "{fasol} 1>&3 2>&4; fi; "
    #            "")

    # The file patterns that should exist after the chandra_repro command has finished running - not just the event
    #  list but some other files as well
    prod_evt_list_name = "{rn}_repro_evt2.fits"
    # The GTI table - there also seems to sometimes be a file with 'flt2' in the name, but nowhere mentions it and
    #  thus I will go with the 'flt1' file
    prod_gti_name = "{rn}_repro_flt1.fits"
    # Newly made bad-pixel file (remember we have chandr_repro configured so that we will ALWAYS make this
    prod_bad_pix_name = "{rn}_repro_bpix1.fits"
    # And finally the 'FOV' file, which provides regions that describe the CCDs I think? - don't actually know if
    #  this will be of any use to us, but we'll keep it for now
    prod_fov_name = "{rn}_repro_fov1.fits"
    # Dead time correction file for HRC data - don't think it'll be present in ACIS reprocessing - also relying on
    #  there ONLY BEING ONE, which I think should be true. The bash will also file if there multiple matches
    prod_dtf_name = "hrcf{oi}_*_dtf1.fits"
    # The aspect solution file is rather important, tells us where the telescope was actually looking throughout
    #  the observation - many other processes require this - IRRITATINGLY IT SEEMS THERE CAN BE MULTIPLE ASPECT
    #  SOLUTION FILES. The asol.lis file contains a list of them, but that is hardly useful for us right now, to
    #  move and rename the asol files in the cmd line
    prod_asol_name = "pcadf{oi}_repro_asol1.fits"
    prod_asol_alt_name = "pcadf{oi}_0*N00*_asol1.fits"

    # These represent the final names and resting places of the event lists (note that we include the energy bound
    #  identifier in the filename, but include no bounds because none are applied right now
    evt_list_name = "obsid{o}-inst{i}-subexp{se}-en-events.fits"
    # Now we do the same for the other file types we're pulling out of this command
    gti_name = "obsid{o}-inst{i}-subexp{se}-defaultGTI.fits"
    bad_pix_name = "obsid{o}-inst{i}-subexp{se}-badpix.fits"
    fov_name = "obsid{o}-inst{i}-subexp{se}-FOVreg.fits"
    # Now adding 'final' names for the aspect solution file (for ACIS AND HRC, definitely need it) and the dead time
    #  file, which is only for HRC as ACIS stores DTC differently I think
    asol_name = "obsid{o}-inst{i}-subexp{se}-aspectsolution.fits"
    dtf_name = "obsid{o}-inst{i}-subexp{se}-deadtimefile.fits"

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Make sure that destreak is the right type of object - then we convert to the string 'yes' or 'no' that
    #  the command line chandra_repro expects
    if not isinstance(destreak, bool):
        raise TypeError("The 'destreak' argument must be a boolean value; you passed {bv}".format(bv=str(destreak)))
    elif destreak:
        destreak = 'yes'
    else:
        destreak = 'no'

    # Now we do the same thing for the 'check_very_faint' argument, which maps to the 'check_vf_pha' chandra_repro
    #  parameter - again it is a boolean choice
    if not isinstance(check_very_faint, bool):
        raise TypeError("The 'check_very_faint' argument must be a boolean value; you passed "
                        "{bv}".format(bv=str(check_very_faint)))
    elif check_very_faint:
        check_very_faint = 'yes'
    else:
        check_very_faint = 'no'

    # The 'pix_adj' argument is not boolean, it has several allowed string values, so we check that the user has
    #  not passed something daft before we blindly pass it on the tool command
    if pix_adj not in ['default', 'edser', 'none', 'randomize']:
        raise ValueError("'pix_adj' must be either; 'default', 'edser', 'none', or 'randomize'.")

    # Rinse and repeat for another boolean variable - 'asol_update'
    if not isinstance(asol_update, bool):
        raise TypeError("The 'asol_update' argument must be a boolean value; you passed "
                        "{bv}".format(bv=str(asol_update)))
    elif asol_update:
        asol_update = 'yes'
    else:
        asol_update = 'no'

    # Rinse and repeat for another boolean variable - 'grating_pi_filter'
    if not isinstance(grating_pi_filter, bool):
        raise TypeError("The 'grating_pi_filter' argument must be a boolean value; you passed "
                        "{bv}".format(bv=str(grating_pi_filter)))
    elif grating_pi_filter:
        grating_pi_filter = 'yes'
    else:
        grating_pi_filter = 'no'
    # ---------------------------------------------------------------------------------------------------------

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # We are iterating through Chandra missions, though only one type exists in DAXA and I don't see that changing
    for miss in chan_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        all_obs = obs_archive.get_obs_to_process(miss.name)
        all_ois = np.array([en[0] for en in all_obs])

        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_ois, 'prepare_chandra_info',
                                                            no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'prepare_chandra_info' runs, so "
                                           "chandra_repro cannot be run.")

        # TODO DEPENDING HOW THEY ACTUALLY DEAL WITH MULTI-OBI OBSERVATIONS, THIS SETUP MAY NOT WORK FOR THEM AS
        #  IS, BUT WE'LL DEAL WITH THAT LATER
        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst-sub-exposure
            #  (though for Chandra the sub-exposure ID matters very VERY rarely) combo
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # Create the variable that points to the 'raw' data for this ObsID of this Chandra mission
            obs_data_path = miss.raw_data_path + obs_id + '/'

            # This path is guaranteed to exist, as it was set up in _ciao_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

            # Also make a root prefix for the files output by chandra_repro, with the random int above and
            #  the ObsID + instrument identifier
            root_prefix = val_id + "_" + str(r_id)

            # ------------------------------ Creating final names for output files ------------------------------
            # First where do we expect them to be before we move and rename them
            evt_out_path = os.path.join(temp_dir, prod_evt_list_name.format(rn=root_prefix))
            gti_out_path = os.path.join(temp_dir, prod_gti_name.format(rn=root_prefix))
            badpix_out_path = os.path.join(temp_dir, prod_bad_pix_name.format(rn=root_prefix))
            fov_out_path = os.path.join(temp_dir, prod_fov_name.format(rn=root_prefix))
            # Only for HRC, the dead time file - note that these files ignore the root prefix, and so are back to
            #  filling with zeros at the beginning of the ObsID to get up to five characters
            dtf_out_path = os.path.join(temp_dir, prod_dtf_name.format(oi=obs_id.zfill(5)))

            # We treat the aspect solution files different, because sometimes there can be multiple, which is deeply
            #  annoying - also according to the OIF files, sometimes there are no shipped asol files at all - then
            #  things fall over, so I have currently marked all without ASPSOL or ASPSOLOBI entries their 'oif.fits' as
            #  unusable
            asol_repro_path = os.path.join(temp_dir, prod_asol_name.format(oi=obs_id.zfill(5)))
            asol_alt_path = os.path.join(temp_dir, prod_asol_alt_name.format(oi=obs_id.zfill(5)))

            # This is where the final output event list file will be stored - after moving and renaming
            evt_final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, se=exp_id, i=inst))
            # And then all the others
            gti_final_path = os.path.join(dest_dir, 'cleaning', gti_name.format(o=obs_id, se=exp_id, i=inst))
            badpix_final_path = os.path.join(dest_dir, 'misc', bad_pix_name.format(o=obs_id, se=exp_id, i=inst))
            fov_final_path = os.path.join(dest_dir, 'misc', fov_name.format(o=obs_id, se=exp_id, i=inst))
            # Only for HRC, the moved dead time file final name
            if inst == 'HRC':
                dtf_final_path = os.path.join(dest_dir, 'misc', dtf_name.format(o=obs_id, se=exp_id, i=inst))
            else:
                dtf_final_path = None
            asol_final_path = os.path.join(dest_dir, 'misc', asol_name.format(o=obs_id, se=exp_id, i=inst))

            # ---------------------------------------------------------------------------------------------------

            # If it doesn't already exist then we will create commands to generate it
            if ('chandra_repro' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['chandra_repro']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                if inst == 'ACIS':
                    # Fill out the template, and generate the command that we will run through subprocess
                    cmd = crp_cmd.format(d=temp_dir, in_f=obs_data_path, out_f=temp_dir, rn=root_prefix, ds=destreak,
                                         cvf=check_very_faint, pa=pix_adj, as_up=asol_update, pf=grating_pi_filter,
                                         oge=evt_out_path, fe=evt_final_path, oggti=gti_out_path, fgti=gti_final_path,
                                         ogbp=badpix_out_path, fbp=badpix_final_path, ogfov=fov_out_path,
                                         ffov=fov_final_path)

                else:
                    # Fill out the template, and generate the command that we will run through subprocess
                    cmd = hrc_crp_cmd.format(d=temp_dir, in_f=obs_data_path, out_f=temp_dir, rn=root_prefix,
                                             ds=destreak, cvf=check_very_faint, pa=pix_adj, as_up=asol_update,
                                             pf=grating_pi_filter, oge=evt_out_path, fe=evt_final_path,
                                             oggti=gti_out_path, fgti=gti_final_path, ogbp=badpix_out_path,
                                             fbp=badpix_final_path, ogfov=fov_out_path, ffov=fov_final_path,
                                             ogdtf=dtf_out_path, fdtf=dtf_final_path)

                # Now add the bash if-else that determines which aspect solution file name to try to move out
                cmd += asol_mv.format(rpasol=asol_repro_path, altasol=asol_alt_path, fasol=asol_final_path, d=temp_dir)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                # TODO CONSIDER WHAT FILES TO CHECK FOR ALTERNATING EXPOSURE AND MULTI-OBI MODES
                miss_final_paths[miss.name][val_id] = evt_final_path
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir, 'evt_list': evt_final_path,
                                                  'default_gti': gti_final_path, 'badpix': badpix_final_path,
                                                  'fov_reg': fov_final_path, 'asol_file': asol_final_path}
                # Only store if an HRC observation
                if dtf_final_path is not None:
                    miss_extras[miss.name][val_id]['dead_time_file'] = dtf_final_path

    # This is just used for populating a progress bar during the process run
    process_message = 'Reprocessing data'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout


@ciao_call
def cleaned_chandra_evts(obs_archive: Archive, lo_en: Quantity = None, hi_en: Quantity = None,
                         allowed_grades: list = None, num_cores: int = NUM_CORES, disable_progress: bool = False,
                         timeout: Quantity = None):
    """
    This function is used to apply the soft-proton filtering (along with any other filtering you may desire, including
    the setting of energy limits) to Chandra event lists, resulting in the creation of sets of cleaned event lists
    which are ready to be analysed. We require that the 'deflare' function has been run before
    running 'cleaned_chandra_evts'.

    Note that a STATUS=0 cut is applied to all cleaned event lists.

    :param Archive obs_archive: An Archive instance containing a Chandra mission instance. This function will fail
        if no Chandra missions are present in the archive.
    :param Quantity lo_en: The lower bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'lo_en' is set to an Astropy Quantity, then 'hi_en' must be as well. Default is None, in which case no
        energy filter is applied. Note that no energy filter can be applied to HRC data.
    :param Quantity hi_en: The upper bound of an energy filter to be applied to the cleaned, filtered, event lists. If
        'hi_en' is set to an Astropy Quantity, then 'lo_en' must be as well. Default is None, in which case no
        energy filter is applied. Note that no energy filter can be applied to HRC data.
    :param list allowed_grades: A list of event grades that should be kept in the final cleaned event list. Default
        is None, as the 'chandra_repro' step already imposes a filter of grades [0,2,3,4,6], any more conservative
        grade filter should be passed as a list of integers. Note that passed values should be in the ASCA grade
        system, NOT THE ACIS FLIGHT GRADE SYSTEM.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the CIAO generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-Inst processes.
    """
    # Run the setup for Chandra processes, which checks that CIAO is installed (as well as CALDB), and checks that the
    #  archive has at least one Chandra mission in it, and
    ciao_vers, caldb_vers, chan_miss = _ciao_process_setup(obs_archive)

    # Setting up the simple commands to make the cleaned event lists from the user arguments - have to make two
    #  commands because we can't apply any energy filtering to HRC event lists. Also, the user won't necessarily
    #  want to apply energy filtering to ACIS data.
    en_clevt_cmd = ('cd {d}; dmcopy infile="{ef}[EVENTS][energy={lo_en}:{hi_en},grade={gr},status=0]" outfile={iev} '
                    'verbose=5; punlearn dmcopy; dmcopy infile="{iev}[EVENTS][@{fgti}]" outfile={fev} verbose=5; '
                    'cd ..; rm -r {d}')

    no_en_clevt_cmd = ('cd {d}; dmcopy infile="{ef}[EVENTS][grade={gr},status=0]" outfile={iev} verbose=5; '
                       'punlearn dmcopy; dmcopy infile="{iev}[EVENTS][@{fgti}]" outfile={fev} verbose=5; '
                       'cd ..; rm -r {d}')

    # HRC strikes again, doesn't have event grades like ACIS, so needs a whole separate command again
    hrc_clevt_cmd = ('cd {d}; dmcopy infile="{ef}[EVENTS][status=0]" outfile={iev} verbose=5; punlearn dmcopy; '
                     'dmcopy infile="{iev}[EVENTS][@{fgti}]" outfile={fev} verbose=5; cd ..; rm -r {d}')

    # Interim event name template - for the midway point where the initial filtering has been applied, but
    #  not yet the flaring GTIs
    int_evt_name = "obsid{o}-inst{i}-subexp{se}-en{en_id}-interimevents.fits"

    # Final name template for the cleaned event lists
    cl_evt_name = "obsid{o}-inst{i}-subexp{se}-en{en_id}-cleanevents.fits"

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Here we are making sure that the input energy limits are legal and sensible
    en_check = [en is not None for en in [lo_en, hi_en]]
    # Both lo_en and hi_en have to be set
    if not all(en_check) and any(en_check):
        raise ValueError("If one energy limit is set (e.g. 'lo_en') then the other energy limit must also be set.")
    elif (lo_en is not None and not lo_en.unit.is_equivalent('eV')) or \
            (hi_en is not None and not hi_en.unit.is_equivalent('eV')):
        raise UnitConversionError("The lo_en and hi_en arguments must be astropy quantities in units "
                                  "that can be converted to eV.")
    # Obviously the upper limit can't be lower than the lower limit, or equal to it.
    elif hi_en is not None and lo_en is not None and hi_en <= lo_en:
        raise ValueError("The hi_en argument must be larger than the lo_en argument.")

    # Make sure we're converted to the right units
    if all(en_check):
        # First make sure we're in keV for energy identifier
        lo_en = lo_en.to('keV')
        hi_en = hi_en.to('keV')
        # This is added into the filtered event list name, but only if energy limits are applied
        en_ident = '_{l}_{h}keV'.format(l=lo_en.value, h=hi_en.value)
        # Then make sure we're in eV for the filling in of the template command later
        lo_en = lo_en.to('eV').astype(int)
        hi_en = hi_en.to('eV').astype(int)
    else:
        en_ident = ''

    # Now we check the input acceptable event grades
    if allowed_grades is None:
        # Set up the default values in case chandra_repro ever changes so that it doesn't enforce them
        allowed_grades = REPRO_GRADES
    # Make sure that any passed values are actually valid in the ASCA grade system
    elif not all([ag in ASCA_SYSTEM_GRADES for ag in allowed_grades]):
        raise ValueError("An invalid event grade has been passed to 'allowed_grades' - only grades from the ASCA "
                         "system may be passed; {asc}".format(asc=", ".join([str(acg) for acg in ASCA_SYSTEM_GRADES])))
    elif any([ag not in REPRO_GRADES for ag in allowed_grades]):
        warn("An event grade that is not selected by chandra_repro has been passed to 'allowed_grades', this will "
             "likely have no effect, or will result in no events left after filtering.", stacklevel=2)

    allowed_grades = ",".join([str(ag) for ag in allowed_grades])
    # ---------------------------------------------------------------------------------------------------------

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # We are iterating through Chandra missions, though only one type exists in DAXA and I don't see that changing.
    #  Much of this code is boilerplate that you'll see throughout the Chandra functions (and similar code in many of
    #  the other telescope processing functions), but never mind - it doesn't need to be that different, so why
    #  should we make it so?
    for miss in chan_miss:
        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        # Getting all the ObsIDs that have been flagged as being able to be processed
        all_obs = obs_archive.get_obs_to_process(miss.name)
        # Then filtering those based on which of them successfully passed the dependency function
        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_obs, 'deflare', no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'deflare' runs, so "
                                           "'cleaned_chandra_evts' cannot be run.")

        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst-sub-exposure
            #  (though for Chandra the sub-exposure ID matters very VERY rarely) combo
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # We will need the event list created by the 'chandra_repro' run, so the path must be retrieved
            rel_evt = obs_archive.process_extra_info[miss.name]['chandra_repro'][val_id]['evt_list']
            # We will also need the flaring GTI produced by 'deflare', and again that is in the process extra info
            rel_flare_gti = obs_archive.process_extra_info[miss.name]['deflare'][val_id]['flaring_gti']

            # This path is guaranteed to exist, as it was set up in _ciao_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

            # ------------------------------ Creating final name for output evt file ------------------------------
            # Also need to set up the name for the interim event list, where filtering expressions. Note that this
            #  lives in the temporary directory, and will be lost to the ages when that directory is deleted at
            #  the end of the process.
            int_evt_final_path = os.path.join(temp_dir, int_evt_name.format(o=obs_id, se=exp_id, i=inst,
                                                                            en_id=en_ident))

            # This is where the final 'clean' event list will live, hopefully devoid of flares and unpleasant events
            cl_evt_final_path = os.path.join(dest_dir, 'events', cl_evt_name.format(o=obs_id, se=exp_id, i=inst,
                                                                                    en_id=en_ident))
            # -----------------------------------------------------------------------------------------------------

            # If it doesn't already exist then we will create commands to generate it
            if ('cleaned_chandra_evts' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['cleaned_chandra_evts']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Slightly different commands based on whether the user wants us to apply an energy cut or not (or
                #  if an energy cut can't be applied - HRC data)
                if lo_en is not None:
                    # Fill out the template, and generate the command that we will run through subprocess
                    cmd = en_clevt_cmd.format(d=temp_dir, ef=rel_evt, lo_en=lo_en.value, hi_en=hi_en.value,
                                              gr=allowed_grades, iev=int_evt_final_path, fgti=rel_flare_gti,
                                              fev=cl_evt_final_path)
                # Here we have no energy cut
                elif lo_en is None and inst != 'HRC':
                    cmd = no_en_clevt_cmd.format(d=temp_dir, ef=rel_evt, gr=allowed_grades, iev=int_evt_final_path,
                                                 fgti=rel_flare_gti, fev=cl_evt_final_path)
                # And here we have an energy-cut-averse instrument (HRC) that also has fundamentally different
                #  information in the event lists
                else:
                    cmd = hrc_clevt_cmd.format(d=temp_dir, ef=rel_evt, iev=int_evt_final_path, fgti=rel_flare_gti,
                                               fev=cl_evt_final_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                miss_final_paths[miss.name][val_id] = cl_evt_final_path
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir, 'cleaned_events': cl_evt_final_path,
                                                  'en_key': en_ident}

    # This is just used for populating a progress bar during the process run
    process_message = 'Assembling cleaned event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout




