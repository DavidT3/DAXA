#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 03/02/2025, 13:48. Copyright (c) The Contributors

from .asca import ASCA
from .base import BaseMission
from .chandra import Chandra
from .erosita import eROSITACalPV, eRASS1DE
from .integral import INTEGRALPointed
from .nustar import NuSTARPointed
from .rosat import ROSATAllSky, ROSATPointed
from .suzaku import Suzaku
from .swift import Swift
from .xmm import XMMPointed
from .xrism import XRISMPointed

# This just links the internal DAXA names of missions to their class
MISS_INDEX = {'xmm_pointed': XMMPointed, 'nustar_pointed': NuSTARPointed, 'erosita_calpv': eROSITACalPV,
              'erosita_all_sky_de_dr1': eRASS1DE, 'chandra': Chandra, 'rosat_all_sky': ROSATAllSky,
              'rosat_pointed': ROSATPointed, 'swift': Swift, 'suzaku': Suzaku, 'asca': ASCA,
              'integral_pointed': INTEGRALPointed, 'xrism_pointed': XRISMPointed}



