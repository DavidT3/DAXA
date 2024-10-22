#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 21/10/2024, 11:15. Copyright (c) The Contributors

from . import _version
from .config import daxa_conf, OUTPUT, NUM_CORES, sb_rate, PFILES_PATH
from .mission.xmm import *

__version__ = _version.get_versions()['version']
