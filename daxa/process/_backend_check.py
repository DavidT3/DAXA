#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 05/12/2022, 14:30. Copyright (c) The Contributors

import os
from subprocess import Popen, PIPE

from packaging.version import Version
from shutil import which
from typing import Bool 

from ..exceptions import SASNotFoundError, SASVersionError, eSASSNotFoundError


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

def find_esass() -> Bool:
    """
     This function checks to ensure the presence of either eSASS on the host system, or for an installation of Docker with a running Docker daemon. 
     It will be called before performing any data processing/reduction of eROSITA data. 
     An error will be thrown if eSASS (or Docker with a running Docker daemon) cannot be identified on the system.

    :return: A bool indicating whether or not eSASS is being used via Docker or not, set to True if Docker is being used. 
    :rtype: Bool
    """
    # Defining the Booleans to check whether eSASS can be used
    docker_installed = False
    docker_daemon_running = False
    esass_outside_docker = False

    # Performing the Docker checks
    # Firstly checking whether it is installed by seeing if 'docker' is on PATH and is marked as executable
    if which('docker') is not None:
        docker_installed = True
    # Then seeing if a Docker daemon is running, aka can a container be run
    cmd = 'docker run hello-world'
    # Running this command in the terminal to see if it is possible to run a container
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    # Decodes the stdout and stderr from the binary encoding it currently exists in. The errors='ignore' flag
    #  means that it doesn't throw errors if there is a character it doesn't recognize
    err = err.decode("UTF-8", errors='ignore')
    # If this doesnt raise an error, then the eSASS env. is enabled and working
    if len(err) == 0:
        docker_daemon_running = True
    
    # Performing eSASS installation checks for eSASS outside of Docker
    cmd = 'evtool'
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    err = err.decode("UTF-8", errors='ignore')
    # If this doesnt raise an error, then the eSASS env. is enabled and working
    if len(err) == 0:
        esass_outside_docker = True

    # Raising errors 
    if not (docker_installed or esass_outside_docker):
        raise eSASSNotFoundError("No version of eSASS has been detected on this system.")

    if docker_installed and not docker_daemon_running and not esass_outside_docker:
        raise eSASSNotFoundError("Please start the Docker daemon so that the eSASS container may be run."
                                " If you are using the desktop application of Docker, this error may arise"
                                " if the application is installed, but not open.") 
    
    # Doing the returns
    if docker_daemon_running and not esass_outside_docker:
        return True
    if not docker_daemon_running and esass_outside_docker:
        return False
    if docker_daemon_running and esass_outside_docker:
        # If docker and esass are both present, use esass outside of docker 
        return False