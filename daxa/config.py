#  This code is a part of the Democratising Archival X-ray Astronomy (DAXA) module.
#  Last modified by David J Turner (turne540@msu.edu) 15/04/2024, 14:49. Copyright (c) The Contributors

import os
from configparser import ConfigParser
from warnings import warn

import pandas as pd
import pkg_resources
from astropy.units import def_unit, ct, deg, s
from numpy import floor

from .exceptions import DAXAConfigError

# If XDG_CONFIG_HOME is set, then use that, otherwise use this default config path
CONFIG_PATH = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'daxa'))
# DAXA config file path
CONFIG_FILE = os.path.join(CONFIG_PATH, 'daxa.cfg')
# Section of the config file for setting up the DAXA module
DAXA_CONFIG = {"daxa_save_path": "daxa_output/",
               "num_cores": -1}

if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)

# If first DAXA run, creates default config file
if not os.path.exists(CONFIG_FILE):
    daxa_default = ConfigParser()
    daxa_default.add_section("DAXA_SETUP")
    daxa_default["DAXA_SETUP"] = DAXA_CONFIG
    with open(CONFIG_FILE, 'w') as new_cfg:
        daxa_default.write(new_cfg)

    # First time run triggers this message
    warn("A configuration file has been created ({}); you can use it to control where DAXA "
         "stores data by default.".format(CONFIG_FILE), stacklevel=2)

daxa_conf = ConfigParser()
# It would be nice to do configparser interpolation, but it wouldn't handle the lists of energy values
daxa_conf.read(CONFIG_FILE)

try:
    cfg_cores = daxa_conf['DAXA_SETUP'].getint('num_cores')
except ValueError:
    raise DAXAConfigError("The 'num_cores' configuration parameter must be an integer, with -1 corresponding "
                          "to a null value and meaning that DAXA will determine the number of cores to use itself.")

# As it turns out, the ConfigParser class is a pain to work with, so we're converting to a dict here
# Addressing works just the same
daxa_conf = {str(sect): dict(daxa_conf[str(sect)]) for sect in daxa_conf}

if cfg_cores != -1 and cfg_cores <= os.cpu_count():
    # If the user has set a number of cores in the config file then we'll use that.
    NUM_CORES = int(daxa_conf["DAXA_SETUP"]["num_cores"])
elif cfg_cores != -1:
    raise DAXAConfigError("You have set a num_cores values that is greater than the number of cores available in"
                          " the current system ({}).".format(os.cpu_count()))
else:
    # Going to allow multi-core processing to use 90% of available cores by default, but
    # this can be over-ridden in individual SAS calls.
    NUM_CORES = max(int(floor(os.cpu_count() * 0.9)), 1)  # Makes sure that at least one core is used


# This is the default output directory for archives setup by DAXA, though it can be overridden on
#  a mission level
OUTPUT = os.path.abspath(daxa_conf["DAXA_SETUP"]["daxa_save_path"]) + "/"

# Here we read in files that list the errors and warnings in SAS
errors = pd.read_csv(pkg_resources.resource_filename(__name__, "files/sas_errors.csv"), header="infer")
warnings = pd.read_csv(pkg_resources.resource_filename(__name__, "files/sas_warnings.csv"), header="infer")
# Just the names of the errors in two handy constants
SASERROR_LIST = errors["ErrName"].values
SASWARNING_LIST = warnings["WarnName"].values

# Reading in the file with information on the eROSITA observations that were made available in the
#  eROSITA CalPV release
EROSITA_CALPV_INFO = pd.read_csv(pkg_resources.resource_filename(__name__, "files/erosita_calpv_info.csv"),
                                 header="infer", dtype={'ObsID': str})
# TODO This may end up changing when we get access to the DR1 release - it could be in a format that makes this
#  a bad way of doing it
# Then doing the same thing, but for the German eRASS:1 release
ERASS_DE_DR1_INFO = pd.read_csv(pkg_resources.resource_filename(__name__, "files/erass_de_dr1_info.csv"),
                                header="infer", dtype={'ObsID': str, 'FIELD1': str, 'FIELD2': str, 'FIELD3': str,
                                                       'FIELD4': str, 'FIELD5': str, 'FIELD6': str, 'FIELD7': str,
                                                       'FIELD8': str, 'FIELD9': str})

# We define a surface brightness rate astropy unit for use in flaregti to measure thresholds in
sb_rate = def_unit('sb_rate', ct / (deg**2 * s))
