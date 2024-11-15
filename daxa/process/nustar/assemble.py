#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 14/11/2024, 23:52. Copyright (c) The Contributors
import os
from random import randint
from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.nustar._common import _nustardas_process_setup, nustardas_call


@nustardas_call
def nupipeline_calibrate(obs_archive: Archive, hp_time_bin: Quantity = Quantity(600, 's'),
                         hp_cell_bin: Union[Quantity, int] = Quantity(5, 'pix'), hp_imp: float = 1.,
                         hp_log_pos: float = -6., hp_bck_thr: int = 6, asp_ab_corr: bool = True,
                         num_cores: int = NUM_CORES, disable_progress: bool = False,
                         timeout: Quantity = None):
    """
    The DAXA wrapper for stage one of the NuSTARDAS tool 'nupipeline', which prepares and calibrates NuSTAR raw
    data - it also calculates aspect solutions, and the mast attitude. The main output are the
    unfiltered, calibrated, event lists - but many other important files are created

    :param Archive obs_archive: An Archive instance containing a NuSTAR mission instance. This function will fail
        if no NuSTAR missions are present in the archive.
    :param Quantity hp_time_bin: Time bin size for the generation of images to search for hot pixels. Default is
        600 seconds.
    :param Quantity/int hp_cell_bin: Spatial cell size to use when searching for hot pixels. Default is 5 pixels, and
        the input must be an odd number of pixels (greater than one).
    :param float hp_imp: The value used to compute the background level (input for the incomplete Gamma
        function) for hot pixel search. Default is 1.
    :param float hp_log_pos: Logarithm of the Poisson probability threshold for rejecting a hot pixel, default
        is -6., and the value must be negative.
    :param int hp_bck_thr: Background threshold used if the candidate hot/flickering pixel's neighborhood
        has zero counts. Default is 6.
    :param bool asp_ab_corr: Controls whether aberration is included in aspecting calculations. Default is True.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the NuSTARDAS generation progress bar.
    :param Quantity timeout: The amount of time each individual process is allowed to run for, the default is None.
        Please note that this is not a timeout for the entire process, but a timeout for individual
        ObsID-Inst processes.
    """
    # Runs standard checks, makes directories, returns NuSTARDAS versions, etc.
    nudas_vers, caldb_vers, nustar_miss = _nustardas_process_setup(obs_archive)

    # --------------------------------- Setting up command and file templates ---------------------------------

    stg_one_cmd = ("cd {d}; nupipeline indir='{in_d}' outdir='outputs' steminputs='nu{oi}' obsmode='{om}' "
                   "instrument='{inst}' entrystage=1 exitstage=1 hpbinsize={hp_tb} hpcellsize={hp_cb} impfac={hp_imp} "
                   "logpos={hp_lp} bthresh={hp_bt} aberration={asp_ab}; mv {oge} {fe}; mv {oghp} {fhp}; "
                   "mv {ogbp} {fbp}; mv {ogrp} {frp}; mv {oga} {fa}; mv {ogm} {fm}; mv {ogo} {fo}; mv {ogps} {fps}; "
                   "mv {ogcps} {fcps}; cd ..; rm -r {d}")

    # TODO MAYBE ADD A LITTLE BASH CHECK FOR THE EXISTENCE OF THE SHARED FILES THAT ARE GENERATED NOT FOR SPECIFIC
    #  INSTRUMENTS - DON'T WANT TO BE COPYING THEM OUT AND HAVE THEM COLLIDE

    # The file patterns that should exist after the first stage of NuSTAR processing has finished running - they
    #  will all be in the 'outputs' directory, as that is what we specified in the command above
    prod_evt_list_name = "nu{oi}{si}_uf.evt"
    # The hot and bad pixels identified by the processing
    prod_hotpix_name = "nu{oi}{si}_hp.fits"
    prod_badpix_name = "nu{oi}{si}_bp.fits"
    # Not going to lie, not entirely sure what this is yet
    prod_detref_name = "nu{oi}{si}_det1.fits"

    # The above were instrument specific, but these are generated regardless of whether FPMA or B is processed - as
    #  such running each instrument separately is a bit wasteful, but it works better with the way DAXA is designed,
    #  as it will make tracking of process failures for specific instruments much easier
    # Attitude file for spacecraft
    prod_att_name = "nu{oi}_att.fits"
    # Mast movement file (I assume)
    prod_mast_name = "nu{oi}_mast.fits"
    prod_obeb_name = "nu{oi}_obeb.hk"
    # Position sensing detector and corrected position sensing detector files
    prod_psd_name = "nu{oi}_psd.fits"
    prod_psdcorr_name = "nu{oi}_psdcorr.fits"

    # These represent the final names and resting places of the event lists (note that we include the energy bound
    #  identifier in the filename, but include no bounds because none are applied right now
    evt_list_name = "obsid{o}-inst{i}-subexpALL-en-events.fits"
    # Now we do the same for the other file types we're pulling out of this command
    bad_pix_name = "obsid{o}-inst{i}-subexpALL-badpix.fits"
    hot_pix_name = "obsid{o}-inst{i}-subexpALL-hotpix.fits"
    det_ref_name = "obsid{o}-inst{i}-subexpALL-refpixel.fits"
    # Now the files that are for the overall ObsID
    att_name = "obsid{o}-attitude.fits"
    mast_name = "obsid{o}-mast.fits"
    obeb_name = "obsid{o}-obeb.fits"
    psd_name = "obsid{o}-psd.fits"
    psdcorr_name = "obsid{o}-psd.fits"
    # ---------------------------------------------------------------------------------------------------------

    # ---------------------------------- Checking and converting user inputs ----------------------------------
    # Checking that the hot-pixel search time bin is the right type of variable and in the right units
    if not isinstance(hp_time_bin, Quantity):
        raise TypeError("The 'hp_time_bin' argument must be a Quantity.")
    elif isinstance(hp_time_bin, Quantity) and not hp_time_bin.unit.is_equivalent('s'):
        raise UnitConversionError("The 'hp_time_bin' argument must be in units convertible to seconds.")
    else:
        hp_time_bin = hp_time_bin.to('s').astype(int)

    # Check the hot-pixel search spatial cell size - must be in pixels but we also allow an integer to be
    #  passed, in which case we convert to a quantity
    if not isinstance(hp_cell_bin, (Quantity, int)):
        raise TypeError("The 'hp_cell_bin' argument must be an astropy quantity or an integer.")
    elif isinstance(hp_cell_bin, Quantity) and not hp_cell_bin.unit.is_equivalent('pix'):
        raise UnitConversionError("The 'hp_cell_bin' argument must be in units convertible to pixels.")
    elif isinstance(hp_cell_bin, int):
        hp_cell_bin = Quantity(hp_cell_bin, 'pix')
    # Make sure it is an integer
    hp_cell_bin = hp_cell_bin.astype(int)
    # Final check (yes this is slightly inelegant but oh well)
    if hp_cell_bin < Quantity(1, 'pix'):
        raise ValueError("The 'hp_cell_bin' argument must be greater than or equal to one.")
    elif hp_cell_bin.value % 2 == 0:
        raise ValueError("The 'hp_cell_bin' argument must be an odd number of pixels.")

    # This is the log of the Poisson probability threshold for rejecting a hot pixel, must be negative
    if hp_log_pos >= 0:
        raise ValueError("The 'hp_log_pos' argument must be negative.")

    # Make sure the background threshold is an integer
    hp_bck_thr = int(hp_bck_thr)

    # Aspect file aberration correction - can be turned off if the user wants
    if asp_ab_corr:
        asp_ab_corr = 'yes'
    else:
        asp_ab_corr = 'no'
    # ---------------------------------------------------------------------------------------------------------

    # Sets up storage dictionaries for bash commands, final file paths (to check they exist at the end), and any
    #  extra information
    miss_cmds = {}
    miss_final_paths = {}
    miss_extras = {}

    # We are iterating through NuSTAR missions - there are pointed and slewing observations so it is possible
    #  that there will be multiple
    for miss in nustar_miss:

        # Changes the type of data to process depending on the mission - though there may be more to this?
        # TODO FIGURE OUT HOW NUSTAR SLEW DATA WORKS
        if 'point' in miss.name:
            obs_mode = 'SCIENCE'
        else:
            obs_mode = 'SLEW'

        # Sets up the top level keys (mission name) in our storage dictionaries
        miss_cmds[miss.name] = {}
        miss_final_paths[miss.name] = {}
        miss_extras[miss.name] = {}

        all_obs = obs_archive.get_obs_to_process(miss.name)
        all_ois = np.array([en[0] for en in all_obs])

        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_ois, 'prepare_nustar_info',
                                                            no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'prepare_nustar_info' runs, so "
                                           "nupipeline_calibrate cannot be run.")

        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst
            #  combination - there are no sub-exposures in NuSTAR like there are in XMM
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst = obs_info

            # Create the variable that points to the 'raw' data for this ObsID of this NuSTAR mission
            obs_data_path = miss.raw_data_path + obs_id + '/'

            # This path is guaranteed to exist, as it was set up in _nustardas_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

            # ------------------------------ Creating final names for output files ------------------------------
            # First where do we expect them to be before we move and rename them
            evt_out_path = os.path.join(temp_dir, 'outputs', prod_evt_list_name.format(oi=obs_id, si=inst[-1]))
            hotpix_out_path = os.path.join(temp_dir, 'outputs', prod_hotpix_name.format(oi=obs_id, si=inst[-1]))
            badpix_out_path = os.path.join(temp_dir, 'outputs', prod_badpix_name.format(oi=obs_id, si=inst[-1]))
            detref_out_path = os.path.join(temp_dir, 'outputs', prod_detref_name.format(oi=obs_id, si=inst[-1]))
            # Then the non-instrument specific ones
            att_out_path = os.path.join(temp_dir, 'outputs', prod_att_name.format(oi=obs_id))
            mast_out_path = os.path.join(temp_dir, 'outputs', prod_mast_name.format(oi=obs_id))
            obeb_out_path = os.path.join(temp_dir, 'outputs', prod_obeb_name.format(oi=obs_id))
            psd_out_path = os.path.join(temp_dir, 'outputs', prod_psd_name.format(oi=obs_id))
            psdcorr_out_path = os.path.join(temp_dir, 'outputs', prod_psdcorr_name.format(oi=obs_id))

            # This is where the final output event list file will be stored - after moving and renaming
            evt_final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, i=inst))
            hotpix_final_path = os.path.join(dest_dir, 'misc', hot_pix_name.format(o=obs_id, i=inst))
            badpix_final_path = os.path.join(dest_dir, 'misc', bad_pix_name.format(o=obs_id, i=inst))
            detref_final_path = os.path.join(dest_dir, 'misc', det_ref_name.format(o=obs_id, i=inst))
            att_final_path = os.path.join(dest_dir, 'misc', att_name.format(o=obs_id))
            mast_final_path = os.path.join(dest_dir, 'misc', mast_name.format(o=obs_id))
            obeb_final_path = os.path.join(dest_dir, 'misc', obeb_name.format(o=obs_id))
            psd_final_path = os.path.join(dest_dir, 'misc', psd_name.format(o=obs_id))
            psdcorr_final_path = os.path.join(dest_dir, 'misc', psdcorr_name.format(o=obs_id))
            # ---------------------------------------------------------------------------------------------------

            # If it doesn't already exist then we will create commands to generate it
            if ('nupipeline_calibrate' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['nupipeline_calibrate']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                cmd = stg_one_cmd.format(d=temp_dir, in_d=obs_data_path, oi=obs_id, inst=inst, om=obs_mode,
                                         hp_tb=hp_time_bin.value, hp_cb=hp_cell_bin.value, hp_imp=hp_imp,
                                         hp_lp=hp_log_pos, hp_bt=hp_bck_thr, asp_ab=asp_ab_corr, oge=evt_out_path,
                                         fe=evt_final_path, oghp=hotpix_out_path, fhp=hotpix_final_path,
                                         ogbp=badpix_out_path, fbp=badpix_final_path, ogrp=detref_out_path,
                                         frp=detref_final_path, oga=att_out_path, fa=att_final_path, ogm=mast_out_path,
                                         fm=mast_final_path, ogo=obeb_out_path, fo=obeb_final_path, ogps=psd_out_path,
                                         fps=psd_final_path, ogcps=psdcorr_out_path, fcps=psdcorr_final_path)

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                # Not much rhyme or reason to why we're only testing for some of the output files
                miss_final_paths[miss.name][val_id] = [evt_final_path, hotpix_final_path, badpix_final_path,
                                                       detref_final_path]
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir, 'evt_list': evt_final_path,
                                                  'hot_pix': hotpix_final_path, 'bad_pix': badpix_final_path,
                                                  'ref_pix': detref_final_path, 'attitude': att_final_path,
                                                  'mast': mast_final_path, 'obeb': obeb_final_path,
                                                  'psd': psd_final_path, 'psdcorr': psdcorr_final_path}

    # This is just used for populating a progress bar during the process run
    process_message = 'Calibrating observations'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout