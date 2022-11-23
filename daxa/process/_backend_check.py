#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 23/11/2022, 16:42. Copyright (c) The Contributors

import os
from subprocess import Popen, PIPE
from warnings import warn


def find_sas():
    # Here we check to see whether SAS is installed (along with all the necessary paths)
    sas_version = None
    if "SAS_DIR" not in os.environ:
        warn("SAS_DIR environment variable is not set, unable to verify SAS is present on system, as such "
             "all functions in xga.sas will not work.")
        sas_version = None
        sas_avail = False
    else:
        # This way, the user can just import the sas_version from this utils code
        sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
        sas_version = sas_out.decode("UTF-8").strip("]\n").split('-')[-1]
        sas_avail = True

    # This checks for the CCF path, which is required to use cifbuild, which is required to do basically
    #  anything with SAS
    if sas_avail and "SAS_CCFPATH" not in os.environ:
        warn("SAS_CCFPATH environment variable is not set, this is required to generate calibration files. As such "
             "functions in xga.sas will not work.")
        sas_avail = False

    return sas_avail, sas_version
