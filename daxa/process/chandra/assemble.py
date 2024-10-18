#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 18/10/2024, 16:03. Copyright (c) The Contributors

from astropy.units import Quantity

from daxa import NUM_CORES
from daxa.archive import Archive
from daxa.process.chandra._common import ciao_call


@ciao_call
def chandra_repro(obs_archive: Archive, num_cores: int = NUM_CORES, disable_progress: bool = False,
                  timeout: Quantity = None):
    pass
