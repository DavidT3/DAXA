#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 09/08/2024, 15:37. Copyright (c) The Contributors

from . import _version
from .config import daxa_conf, OUTPUT, NUM_CORES, sb_rate
from .mission.xmm import *

__version__ = _version.get_versions()['version']
