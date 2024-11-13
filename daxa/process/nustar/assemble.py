#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 12/11/2024, 21:57. Copyright (c) The Contributors
from random import randint

import numpy as np
from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.nustar._common import _nustardas_process_setup, nustardas_call


@nustardas_call
def nupipeline_calibrate(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                         timeout: Quantity = None):

    # Runs standard checks, makes directories, returns NuSTARDAS versions, etc.
    nudas_vers, caldb_vers, nustar_miss = _nustardas_process_setup(obs_archive)




    # --------------------------------- Setting up command and file templates ---------------------------------

    # fpma_infile={evt_a} fpmb_infile={evt_b} attfile={att} "
    #                    "fpma_hkfile={hk_a} fpmb_hkfile={hk_b} cebhkfile={hk_ceb} inobebhkfile={hk_obeb}
    stg_one_cmd = ("cd {d}; nupipeline indir={arch_d} obsmode={obsmode} entrystage=1 exitstage=2 "
                   "hpbinsize={hp_tbin} hpcellsize={hp_cbin} impfac={hp_imp} logpos={hp_logpos} bthresh={hp_bthr}"
                   "aberration={asp_ab}"
                   "obebhkfile={out_hk_obeb} outattfile={out_att} outpsdfile={out_psd} outpsdfilecor={out_corr_psd} "
                   "mastaspectfile={out_mask_asp} fpma_outbpfile={out_bp_a} fpmb_outbpfile={out_bp_b} "
                   "fpma_outhpfile={out_hp_a} fpmb_outhpfile={out_hp_b}")
    # ---------------------------------------------------------------------------------------------------------


    # ---------------------------------- Checking and converting user inputs ----------------------------------

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

        print(good_obs)
        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'prepare_nustar_info' runs, so "
                                           "nupipeline_calibrate cannot be run.")

        # TODO DEPENDING HOW THEY ACTUALLY DEAL WITH MULTI-OBI OBSERVATIONS, THIS SETUP MAY NOT WORK FOR THEM AS
        #  IS, BUT WE'LL DEAL WITH THAT LATER
        for obs_info in good_obs:
            # This is the valid id that allows us to retrieve the specific product for this ObsID-Inst
            #  combination - there are no sub-exposures in NuSTAR like there are in XMM
            val_id = ''.join(obs_info)
            # Split out the information in obs_info
            obs_id, inst, exp_id = obs_info

            # Create the variable that points to the 'raw' data for this ObsID of this NuSTAR mission
            obs_data_path = miss.raw_data_path + obs_id + '/'

            # This path is guaranteed to exist, as it was set up in _ciao_process_setup. This is where output
            #  files will be written to.
            dest_dir = obs_archive.construct_processed_data_path(miss, obs_id)

            # Set up a temporary directory to work in (probably not really necessary in this case, but will be
            #  in other processing functions).
            r_id = randint(0, int(1e+8))
            temp_name = "tempdir_{}".format(r_id)
            temp_dir = dest_dir + temp_name + "/"

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
    process_message = 'Calibrating data'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout