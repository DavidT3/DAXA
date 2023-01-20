#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 05/12/2022, 14:30. Copyright (c) The Contributors

import os
from subprocess import Popen, PIPE

from packaging.version import Version

from ..exceptions import SASNotFoundError, SASVersionError


def find_sas() -> Version:
    """
    This function checks to ensure the presence of SAS on the host system, and it will be called before performing
    any data processing/reduction of XMM data. An error will be thrown if SAS (or SAS calibration files) cannot
    be identified on the system.

    :return: The SAS version that has been successfully identified, as an instance of the 'packaging'
        module's Version class.
    :rtype: Version
    """
    # Here we check to see whether SAS is installed (along with all the necessary paths)
    sas_version = None
    if "SAS_DIR" not in os.environ:
        raise SASNotFoundError("SAS_DIR environment variable is not set, unable to verify SAS is present on "
                               "system, as such XMM raw data (ODFs) cannot be processed.")
        sas_version = None
        sas_avail = False
    else:
        sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
        sas_version = Version(sas_out.decode("UTF-8").strip("]\n").split('-')[-1])
        sas_avail = True

    # This checks for the CCF path, which is required to use cifbuild, which is required to do basically
    #  anything with SAS (including processing ODFs)
    if sas_avail and "SAS_CCFPATH" not in os.environ:
        raise SASNotFoundError("SAS_CCFPATH environment variable is not set, this is required to generate "
                               "calibration files. As such XMM raw data (ODFs) cannot be processed.")

    if sas_version < Version('14.0.0'):
        raise SASVersionError("The detected SAS installation is of too low a version ({v}), please use version 14 or "
                              "later.".format(v=sas_version))

    return sas_version
