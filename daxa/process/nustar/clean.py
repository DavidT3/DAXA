#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 18/11/2024, 10:52. Copyright (c) The Contributors

import os
from random import randint

import numpy as np
from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.exceptions import NoDependencyProcessError
from daxa.process.nustar._common import _nustardas_process_setup, nustardas_call


@nustardas_call
def nupipeline_clean(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                     timeout: Quantity = None):
    # Runs standard checks, makes directories, returns NuSTARDAS versions, etc.
    nudas_vers, caldb_vers, nustar_miss = _nustardas_process_setup(obs_archive)

    # --------------------------------- Setting up command and file templates ---------------------------------

    # indir='{in_d}'    steminputs='nu{oi}'
    stg_two_cmd_a = ("cd {d}; nupipeline fpma_infile='{ef}' outdir='outputs' obsmode='{om}' "
                    "instrument='A' entrystage=2 exitstage=2 inmastaspectfile={ma} fpma_inoptaxisfile={oa} "
                    "fpma_indet1reffile={dr} inpsdfilecor={pc}")

    # cd ..; rm -r {d}

    # File name templates for things produced by this task that we want to keep
    # TODO CHANGE OBVIOUSLY
    prod_evt_list_name = "nu{oi}{si}_uf.evt"

    # The final file names we'll assign to the files that we want to keep
    evt_list_name = "obsid{o}-inst{i}-subexpALL-en-events.fits"
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

        good_obs_sel = obs_archive.check_dependence_success(miss.name, all_obs, 'nupipeline_calibrate',
                                                            no_success_error=False)
        good_obs = np.array(all_obs)[good_obs_sel]

        # Have to check that there is something for us to work with here!
        if len(good_obs) == 0:
            raise NoDependencyProcessError("No observations have had successful 'nupipeline_calibrate' runs, so "
                                           "nupipeline_clean cannot be run.")

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

            # This is where the final output event list file will be stored - after moving and renaming
            evt_final_path = os.path.join(dest_dir, 'events', evt_list_name.format(o=obs_id, i=inst))
            # ---------------------------------------------------------------------------------------------------

            # ----------------------------- Retrieving files from the previous stage ----------------------------
            # We need many of the files that were created in the first stage of processing ('nupipeline_calibrate')
            # Firstly, the calibrated (but not yet cleaned!) event list
            rel_evt = obs_archive.process_extra_info[miss.name]['nupipeline_calibrate'][val_id]['evt_list']

            # Then the mast aspect file (which accounts for any flexing or deformation of the mast (I think?)
            rel_mast = obs_archive.process_extra_info[miss.name]['nupipeline_calibrate'][val_id]['mast']

            # The file that describes the pointing of each telescope (optical axis) as a function of time
            rel_optax = obs_archive.process_extra_info[miss.name]['nupipeline_calibrate'][val_id]['opt_axis']

            # Detector reference pixel file
            rel_detref = obs_archive.process_extra_info[miss.name]['nupipeline_calibrate'][val_id]['ref_pix']

            # This is the 'corrected' position sensing detector file - these track the positions of the two laser
            #  points on the 'position sensing detectors', which are used to calculate the mast aspect file
            rel_psdcorr = obs_archive.process_extra_info[miss.name]['nupipeline_calibrate'][val_id]['psdcorr']
            # ---------------------------------------------------------------------------------------------------


            # If it doesn't already exist then we will create commands to generate it
            if ('nupipeline_clean' not in obs_archive.process_success[miss.name] or
                    val_id not in obs_archive.process_success[miss.name]['nupipeline_clean']):
                # Make the temporary directory for processing - this (along with the temporary PFILES that
                #  the execute_cmd function will create) should help avoid any file collisions
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)

                # stg_two_cmd_a = ("cd {d}; nupipeline fpma_infile='{ef}' outdir='outputs' obsmode='{om}' "
                #                     "instrument='A' entrystage=2 exitstage=2 inmastaspectfile={ma} fpma_inoptaxisfile={oa} "
                #                     "fpma_indet1reffile={dr} inpsdfilecor={pc}")

                # We have two slightly different templates for the two FPMs - simply because the input parameter
                #  names are instrument specific, the setup and processes run are the same
                if inst == 'FPMA':
                    cmd = stg_two_cmd_a.format(d=temp_dir, oi=obs_id, om=obs_mode, ef=rel_evt, ma=rel_mast,
                                               oa=rel_optax, dr=rel_detref, pc=rel_psdcorr)
                elif inst == 'FPMB':
                    raise NotImplementedError("Nope")

                # Now store the bash command, the path, and extra info in the dictionaries
                miss_cmds[miss.name][val_id] = cmd
                # Not much rhyme or reason to why we're only testing for some of the output files
                miss_final_paths[miss.name][val_id] = "evt_final_path"
                miss_extras[miss.name][val_id] = {'working_dir': temp_dir}

    # This is just used for populating a progress bar during the process run
    process_message = 'Producing science-ready event lists'

    return miss_cmds, miss_final_paths, miss_extras, process_message, num_cores, disable_progress, timeout