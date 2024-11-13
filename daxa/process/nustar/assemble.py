#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 12/11/2024, 21:40. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.process.nustar._common import _nustardas_process_setup, nustardas_call


@nustardas_call
def nupipeline_calibrate(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                         timeout: Quantity = None):

    # Runs standard checks, makes directories, returns NuSTARDAS versions, etc.
    nudas_vers, caldb_vers, nustar_miss = _nustardas_process_setup(obs_archive)


    # fpma_infile={evt_a} fpmb_infile={evt_b} attfile={att} "
    #                    "fpma_hkfile={hk_a} fpmb_hkfile={hk_b} cebhkfile={hk_ceb} inobebhkfile={hk_obeb}

    stg_one_cmd = ("cd {d}; nupipeline indir={arch_d} obsmode='SCIENCE' entrystage=1 exitstage=2 "
                   "hpbinsize={hp_tbin} hpcellsize={hp_cbin} impfac={hp_imp} logpos={hp_logpos} bthresh={hp_bthr}"
                   "aberration={asp_ab}"
                   "obebhkfile={out_hk_obeb} outattfile={out_att} outpsdfile={out_psd} outpsdfilecor={out_corr_psd} "
                   "mastaspectfile={out_mask_asp} fpma_outbpfile={out_bp_a} fpmb_outbpfile={out_bp_b} "
                   "fpma_outhpfile={out_hp_a} fpmb_outhpfile={out_hp_b}")

    # return