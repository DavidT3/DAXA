#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 26/07/2023, 03:53. Copyright (c) The Contributors

from .base import BaseMission
from .chandra import Chandra
from .erosita import eROSITACalPV
from .nustar import NuSTARPointed
from .rosat import ROSATAllSky
from .xmm import XMMPointed

# This just links the internal DAXA names of missions to their class
MISS_INDEX = {'xmm_pointed': XMMPointed, 'nustar_pointed': NuSTARPointed, 'erosita_calpv': eROSITACalPV,
              'chandra': Chandra, 'rosat_all_sky': ROSATAllSky}



