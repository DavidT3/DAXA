#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 12/11/2024, 09:20. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.process.nustar._common import _nustardas_process_setup


def nupipeline_calibrate(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                         timeout: Quantity = None):

    # Runs standard checks, makes directories, returns NuSTARDAS versions, etc.
    nudas_vers, caldb_vers, nustar_miss = _nustardas_process_setup(obs_archive)

    stg_one_cmd = "cd {d}; nupipeline"