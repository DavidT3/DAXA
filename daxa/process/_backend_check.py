#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 31/03/2023, 17:22. Copyright (c) The Contributors

import os
from shutil import which
from subprocess import Popen, PIPE

from packaging.version import Version

from ..exceptions import SASNotFoundError, SASVersionError, BackendSoftwareError


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


def find_lcurve() -> Version:
    """
    This function searches for the lcurve tool (makes light curves) and raises an exception of it cannot be found
    on the system. lcurve is distributed as part of HEASoft but, as HEASoft can be downloaded in multiple
    configurations, it is prudent to check for it.

    :return: The version of HEASoft (lcurve does not have an individual verson).
    :rtype: Version
    """

    # Use the shutil interface with the Unix 'which' command to check whether the lcurve binary is on the path
    if which('lcurve') is not None:
        # If it is then we can run a quick version check and parse the output
        lc_out, lc_err = Popen("lcurve --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
        lc_version = Version(lc_out.decode("UTF-8").split('Version ')[-1].split(' ')[0])
    else:
        # If we cannot find lcurve on the path then we raise a (hopefully useful) exception
        raise BackendSoftwareError("The lcurve package (included in the XRONOS section of HEASoft) cannot be "
                                   "found, you may not have installed HEASoft with the right software selections.")

    return lc_version

