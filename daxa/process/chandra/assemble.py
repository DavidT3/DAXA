#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 21/10/2024, 16:39. Copyright (c) The Contributors
import os
from random import randint

import numpy as np
from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.chandra._common import _ciao_process_setup, ciao_call


@ciao_call
def chandra_repro(obs_archive: Archive, destreak: bool = True, check_very_faint: bool = False, pix_adj: str = 'default',
                  asol_update: bool = True, grating_pi_filter: bool = True, num_cores: int = NUM_CORES,
                  disable_progress: bool = False, timeout: Quantity = None):
    """

    NOTES:
    - We will always produce new bad-pixel files, as we don't currently allow the user to pass their own
    - We will also always produce a new event file with 'process_events', so we don't allow any choice. We want a new
      calibrated level 1 (and from there level 2) event files in all cases.
    - QUESTION FOR ME - set_ardlib controls whether the observation bad-pixel file is stored in the ardlib and (I
      think) doesn't have to be supplied to other analyses. This won't play well with multi-processing I think, and I
      do remember they mentioned ARDLIB in the multi-processing part of their docs. I'll either have to do it that way
      or see whether the bad pixel file can be passed manually for each analysis we might do (which honestly is
      probably the way because XGA is gonna be doing all that stuff). 'set_ardlib=FALSE' FOR NOW!!
    - Following CIAO docs advice for the 'check_vf_pha' parameter and having it set to False by
      default (https://cxc.harvard.edu/ciao/why/aciscleanvf.html.)
    - Don't fully understand 'pix_adj' yet, but should probably include if I don't understand it well enough to not
      be able to argue against it
    - 'tg_zo_position' - this defines the coordinate of the target to be reduced in a grating observation, the 'zero
      point' - we run into the same problem as processing RGS that we just want to prep the data without making
      spectra, ARF, RMF, etc. - as this tool just prepares the data, and the user might not want to ultimately look
      at the brightest source in the field. NOT SURE WHAT TO DO YET - STARTED BY LEAVING IT ON THE DEFAULT BEHAVIOUR
    - 'asol_update' - again don't know why you wouldn't want to do this, but we'll leave the choice in
    - 'pi_filter' - for the grating spectra, a low-cost way of lowering the background it seems? I'll leave the
      choice and it'll be on by default
    - I will initially set verbose to 5, to store the maximum amount of data for debugging
    - The 'root' option, which I think essentially controls the prefix on the generated files, will be set to the
      ObsID + instrument + randomly generated ID I think, just to be totally sure which ones we made (also while I
      don't fully understand the behaviours of chandra_repro in practise I think it'll be a good way of tracking what
      it is doing

    :param Archive obs_archive:
    :param bool destreak:
    :param bool check_very_faint:
    :param str pix_adj:
    :param bool asol_update:
    :param bool grating_pi_filter:
    :param int num_cores:
    :param bool disable_progress:
    :param Quantity timeout:
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
               "mv {ogbp} {fbp}; mv {ogfov} {ffov}")
    # cd ..; rm -r {d};


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

    # These represent the final names and resting places of the event lists
    evt_list_name = "obsid{o}-inst{i}-subexp{se}-events.fits"
    # Now we do the same for the other file types we're pulling out of this command
    gti_name = "obsid{o}-inst{i}-subexp{se}-defaultGTI.fits"
    bad_pix_name = "obsid{o}-inst{i}-subexp{se}-badpix.fits"
    fov_name = "obsid{o}-inst{i}-subexp{se}-FOVreg.fits"

    # TODO THINK THE asol1 AND eph1 AND dtf1 (FOR HRC) FILES MIGHT NEED TO BE LIFTED OUT AS WELL - IN FACT WE MIGHT
    #  NEED TO SOME SUBTLY DIFFERENT BEHAVIOURS HERE FOR HRC AND ACIS

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

            # This path is guaranteed to exist, as it was set up in _sas_process_setup. This is where output
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

            # This is where the final output event list file will be stored - after moving and renaming
            evt_final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, se=exp_id, i=inst))
            # And then all the others
            gti_final_path = os.path.join(dest_dir, 'misc', gti_name.format(o=obs_id, se=exp_id, i=inst))
            badpix_final_path = os.path.join(dest_dir, 'misc', bad_pix_name.format(o=obs_id, se=exp_id, i=inst))
            fov_final_path = os.path.join(dest_dir, 'misc', fov_name.format(o=obs_id, se=exp_id, i=inst))
            # ---------------------------------------------------------------------------------------------------

            # If it doesn't already exist then we will create commands to generate it
            if ('chandra_repro' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['chandra_repro']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # Fill out the template, and generate the command that we will run through subprocess
                cmd = crp_cmd.format(d=temp_dir, in_f=obs_data_path, out_f=temp_dir, rn=root_prefix, ds=destreak,
                                     cvf=check_very_faint, pa=pix_adj, as_up=asol_update, pf=grating_pi_filter,
                                     oge=evt_out_path, fe=evt_final_path, oggti=gti_out_path, fgti=gti_final_path,
                                     ogbp=badpix_out_path, fbp=badpix_final_path, ogfov=fov_out_path,
                                     ffov=fov_final_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                # TODO CONSIDER WHAT FILES TO CHECK FOR ALTERNATING EXPOSURE AND MULTI-OBI MODES
                miss_final_paths[miss.name][val_id] = evt_final_path
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir}

            # This is just used for populating a progress bar during the process run
        process_message = 'Reprocessing Chandra data'

        return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout








