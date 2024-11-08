#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 07/11/2024, 22:55. Copyright (c) The Contributors

import os
from shutil import which
from subprocess import Popen, PIPE
from typing import Tuple

from packaging.version import Version

from ..exceptions import (SASNotFoundError, SASVersionError, eSASSNotFoundError, BackendSoftwareError,
                          CIAONotFoundError, NuSTARDASNotFoundError)


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
    elif sas_avail and not os.path.exists(os.environ['SAS_CCFPATH']):
        raise SASNotFoundError("Though the SAS_CCFPATH environment variable is set, the path ({p}) does not "
                               "exist".format(p=os.environ['SAS_CCFPATH']))

    if sas_version < Version('14.0.0'):
        raise SASVersionError("The detected SAS installation is too low of a version ({v}), please use version 14 or "
                              "later.".format(v=sas_version))

    return sas_version


def find_esass() -> bool:
    """
     This function checks to ensure the presence of either eSASS on the host system, or for an installation of Docker
     with a running Docker daemon. It will be called before performing any data processing/reduction of eROSITA data.
     An error will be thrown if eSASS (or Docker with a running Docker daemon) cannot be identified on the system.

    :return: A bool indicating whether or not eSASS is being used via Docker or not, set to True if Docker
        is being used.
    :rtype: Bool
    """
    # Defining the Booleans to check whether eSASS can be used
    docker_installed = False
    docker_daemon_running = False
    esass_outside_docker = False

    # Performing the Docker checks
    # Firstly checking whether it is installed by seeing if 'docker' is on PATH and is marked as executable
    if which('docker') is not None:
        docker_installed = True

        # Then seeing if a Docker daemon is running, aka can a container be run
        cmd = 'docker run hello-world'
        # Running this command in the terminal to see if it is possible to run a container
        out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
        # Decodes the stdout and stderr from the binary encoding it currently exists in. The errors='ignore' flag
        #  means that it doesn't throw errors if there is a character it doesn't recognize
        err = err.decode("UTF-8", errors='ignore')
        # If this doesn't raise an error, then the eSASS env. is enabled and working
        if len(err) == 0:
            docker_daemon_running = True
    
    # Performing eSASS installation checks for eSASS outside of Docker
    # Checking whether it is installed by seeing if 'evtool' is on PATH and is marked as executable
    if which('evtool') is not None:
        esass_outside_docker = True
        
    # Raising errors 
    if not (docker_installed or esass_outside_docker):
        raise eSASSNotFoundError("No version of eSASS has been detected on this system.")

    if docker_installed and not docker_daemon_running and not esass_outside_docker:
        raise eSASSNotFoundError("Please start the Docker daemon so that the eSASS container may be run."
                                 " If you are using the desktop application of Docker, this error may arise"
                                 " if the application is installed, but not open.")
        
    if docker_daemon_running and not esass_outside_docker:
        raise NotImplementedError("DAXA currently only supports eSASS via direct installation, and not via Docker.")
      
    # Doing the returns
    if docker_daemon_running and not esass_outside_docker:
        return True
    if not docker_daemon_running and esass_outside_docker:
        return False
    if docker_daemon_running and esass_outside_docker:
        # If docker and esass are both present, use esass outside of docker
        return False


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


def find_ciao() -> Tuple[Version, Version]:
    """
    This function checks to ensure the presence of CIAO on the host system, and it will be called before performing
    any data processing/reduction of Chandra data. An error will be thrown if CIAO (or CalDB Chandra calibration
    files) cannot be identified on the system.

    :return: The CIAO version that has been successfully identified, as an instance of the 'packaging'
        module's Version class, and the CALDB version as another instance of the Version class.
    :rtype: Tuple[Version, Version]
    """
    # Here we check to see whether CIAO is installed at all, and get outputs from the terminal which tell us about
    #  the versions - CIAO does have a handy Python implementation, but I think we'll avoid using it for now just
    #  in case there is a CIAO implementation on the system that isn't installed to the environment
    ciao_out, ciao_err = Popen("ciaover -v", stdout=PIPE, stderr=PIPE, shell=True).communicate()
    # Just turn those pesky byte outputs into strings
    ciao_out = ciao_out.decode("UTF-8")
    ciao_err = ciao_err.decode("UTF-8")

    # We initially check to see if our ciaover command ran at all, if it did then there is clearly a CIAO
    #  installation, if not then we throw an exception, as it will be fatal to any hope of processing Chandra data
    if "ciaover: command not found" in ciao_err:
        raise CIAONotFoundError("CIAO cannot be identified on your system, and Chandra data cannot be processed.")
    else:
        # The ciaover output is over a series of lines, with different info on each - this is a little bit of a hard
        #  code cheesy method to do this, but we'll split them on lines and selected the 2nd line to get
        #  the ciao version
        split_out = [en.strip(' ') for en in ciao_out.split('\n')]
        # Strip the CIAO version out of the ciaover output
        ciao_version = Version(split_out[1].split(':')[-1].split('CIAO')[-1].strip(' ').split(' ')[0])

    # If we've got to this point, then we know CIAO is installed - now we must make sure CALDB is present (and
    #  determine the version) - we'll use the split_out variable again seeing as we already split the information lines
    if 'not installed' in split_out[5].lower():
        raise CIAONotFoundError("A Chandra CALDB installation cannot be identified on your system, and as such "
                                "Chandra data cannot be processed.")
    else:
        # Strip out the CALDB version
        caldb_version = Version(split_out[5].split(':')[-1].strip())

    return ciao_version, caldb_version


def find_nustardas() -> Tuple[Version, Version]:
    """
    This function checks to ensure the presence of the NuSTARDAS software on the host system, and it will be called
    before performing any data processing/reduction of NuSTAR data. An error will be thrown if CIAO (or CalDB NuSTAR
    calibration files) cannot be identified on the system.

    :return: The NuSTARDAS version that has been successfully identified, as an instance of the 'packaging'
        module's Version class, and the NuSTAR CALDB version as another instance of the Version class.
    :rtype: Tuple[Version, Version]
    """
    # Here we check to see whether NuSTARDAS is installed - if the user has a full HEASoft installation then it will
    #  be - we're going to check with the 'nuversion' command that they created for this very purpose (it will also
    #  give us the NuSTARDAS version, but not the NuSTAR CALDB version)
    nu_out, nu_err = Popen("nuversion", stdout=PIPE, stderr=PIPE, shell=True).communicate()
    # Just turn those pesky byte outputs into strings
    nu_out = nu_out.decode("UTF-8")
    nu_err = nu_err.decode("UTF-8")

    # We initially check to see if our nuversion command ran at all, if it did then there is clearly a NuSTARDAS
    #  installation, if not then we throw an exception, as it will be fatal to any hope of processing NuSTAR data
    if "command not found" in nu_err:
        raise NuSTARDASNotFoundError("NuSTARDAS cannot be identified on your system, and NuSTAR data cannot be "
                                     "processed.")
    else:
        # If there was an output, it is very simple to strip the version out
        nudas_version = Version(nu_out.split('_')[-1])

    # If we've got to this point, then we know NuSTARDAS is installed - but we still need to check for the NuSTAR
    #  calibration files being present. Unfortunately, there doesn't appear to be a single handy command to pull out
    #  the NuSTAR CALDB version. We instead get the date that the NuSTAR CALDB was last updated
    nu_cal_out, nu_cal_err = Popen("readlink $CALDB/data/nustar/fpm/caldb.indx", stdout=PIPE, stderr=PIPE,
                                   shell=True).communicate()
    # Just turn those pesky byte outputs into strings
    nu_cal_out = nu_cal_out.decode("UTF-8")
    nu_cal_err = nu_cal_err.decode("UTF-8")
    if nu_cal_out == "":
        raise NuSTARDASNotFoundError("A NuSTAR CALDB installation cannot be identified on your system, and as such "
                                     "NuSTAR data cannot be processed.")
    else:
        # Strip out the CALDB version
        caldb_version = Version("v" + nu_cal_out.split('indx')[-1])

    return nudas_version, caldb_version